"""Offline bottlenecks and compressed full-model prefill/decode timing.

Writes into the existing quantization-profile.json/log artifacts. No original
checkpoints are modified. CUDA event intervals include stream idle/launch gaps;
they are not claimed to be sums of GPU kernel execution times.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from types import SimpleNamespace

from accelerate import init_empty_weights
from safetensors import safe_open
import torch
import quiptools_cuda
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from lib import codebook, utils
from lib.algo import quip
from lib.linear.quantized_linear import QuantizedLinear
from lib.linear.activation_quantization import (activation_config, activation_overrides,
    add_activation_args, configure_activation_quantization)
from lib.utils.matmul_fft import matmul_rfft, matmul_rfft_inverse
from lib.utils.matmul_had import matmul_hadU_cuda
from model.llama import LlamaForCausalLM
from quantize_llama.reproduce_incoherence import evaluate, restore


GROUPS = [('qkv', ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']),
          ('o', ['self_attn.o_proj']),
          ('up', ['mlp.up_proj', 'mlp.gate_proj']),
          ('down', ['mlp.down_proj'])]
CODEBOOKS = {2: 'E8P12', 3: 'E8P12RVQ3B', 4: 'E8P12RVQ4B'}


class StreamTimer:
    def __init__(self):
        self.enabled = False
        self.events = []

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((name, start, end, time.perf_counter() - begin))

    def reset(self):
        self.events.clear()

    def collect(self):
        torch.cuda.synchronize()
        sums = defaultdict(lambda: dict(stream_ms=0., cpu_enqueue_ms=0., calls=0))
        for name, start, end, wall in self.events:
            item = sums[name]
            item['stream_ms'] += start.elapsed_time(end)
            item['cpu_enqueue_ms'] += wall * 1000
            item['calls'] += 1
        for item in sums.values():
            item['mean_stream_ms'] = item['stream_ms'] / item['calls']
        return dict(sums)


def save(cli, result):
    if getattr(cli, 'experiment_key', None):
        cli.root_result.setdefault('model_comparisons', {})[cli.experiment_key] = result
        result = cli.root_result
    temporary = cli.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(cli.output)


def other_gpu_processes():
    output = subprocess.check_output(['nvidia-smi', '--id=0',
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return sorted(int(line.strip()) for line in output.splitlines()
                  if line.strip().isdigit() and int(line.strip()) != os.getpid())


def wait_for_exclusive_gpu():
    idle = 0
    while idle < 3:
        others = other_gpu_processes()
        idle = 0 if others else idle + 1
        if others:
            print('WAIT_GPU other_compute_pids', others, flush=True)
        if idle < 3:
            time.sleep(10)


def read_weight(cli, index, name):
    with safe_open(cli.base_model / index[name], framework='pt', device='cpu') as f:
        return f.get_tensor(name)


def offline(cli, result, index):
    cb = codebook.get_codebook('E8P12')
    timer = StreamTimer()
    selector_original = type(cb).quantize
    ldl_original = quip.LDLQ_buffered
    measured = {}
    active_ldl = False
    active_preprocess = False

    def selector(self, *a, **kw):
        if active_ldl:
            with timer.section('codeword_selection'):
                return selector_original(self, *a, **kw)
        return selector_original(self, *a, **kw)

    def ldl(*a, **kw):
        nonlocal active_ldl
        active_ldl = True
        try:
            with timer.section('ldlq_total'):
                return ldl_original(*a, **kw)
        finally:
            active_ldl = False

    patches = []

    def patch(obj, name, replacement):
        patches.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    def timed(obj, name, label):
        original = getattr(obj, name)

        def call(*a, **kw):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            value = original(*a, **kw)
            torch.cuda.synchronize()
            measured[label] = measured.get(label, 0.) + time.perf_counter() - begin
            return value
        patch(obj, name, call)

    patch(type(cb), 'quantize', selector)
    patch(quip, 'LDLQ_buffered', ldl)
    preprocess_original = quip.incoherence_preprocess
    cholesky_original = torch.linalg.cholesky

    def preprocess(*a, **kw):
        nonlocal active_preprocess
        active_preprocess = True
        try:
            return preprocess_original(*a, **kw)
        finally:
            active_preprocess = False

    def cholesky(*a, **kw):
        if not active_preprocess:
            return cholesky_original(*a, **kw)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        value = cholesky_original(*a, **kw)
        torch.cuda.synchronize()
        measured['preprocess_cholesky'] = time.perf_counter() - begin
        return value

    patch(quip, 'incoherence_preprocess', preprocess)
    patch(torch.linalg, 'cholesky', cholesky)
    timed(quip, 'incoherence_preprocess', 'rotation_preprocessing_including_cholesky')
    timed(utils, 'block_LDL', 'block_ldl')
    timed(quip, 'LDLQ_buffered', 'ldlq_total')
    timed(quip, 'incoherence_process', 'inverse_weight_rotation')
    timed(type(cb), 'maybe_pack_idxs', 'index_packing')
    timed(utils, 'show_metrics', 'cpu_error_diagnostics')
    timed(torch, 'save', 'checkpoint_write')
    quant_original = quip.quantize
    quant_bounds = {}

    def quant(*a, **kw):
        quant_bounds['start'] = time.perf_counter()
        value = quant_original(*a, **kw)
        quant_bounds['end'] = time.perf_counter()
        return value
    patch(quip, 'quantize', quant)
    result['offline'] = []
    try:
        with tempfile.TemporaryDirectory(prefix='quip-profile-') as td:
            for mode in ('had', 'fft'):
                for group_idx, (group, attrs) in enumerate(GROUPS):
                    weights = [read_weight(cli, index,
                               f'model.layers.{cli.layer}.{a}.weight') for a in attrs]
                    args = SimpleNamespace(incoh_mode=mode, use_fp64=False,
                        rescale_WH=False, lora_rank=0, full_svd=False,
                        sigma_reg=.01, sigma_reg2=.01, scale_override=.9,
                        resid_scale_override=-1, no_use_buffered=False,
                        lowmem_ldlq=False, quip_tune_iters=10, save_pfx=td)
                    times = {}
                    saved_indices = None
                    # Same RNG, weights and algorithm: isolate instrumentation effects.
                    for instrument in (False, True):
                        timer.reset()
                        timer.enabled = instrument
                        measured.clear()
                        torch.manual_seed(cli.layer + group_idx)
                        path = Path(td) / f'{mode}-{group}-{instrument}.pt'
                        torch.cuda.synchronize()
                        begin = time.perf_counter()
                        quip.quantize_linear(weights, str(path),
                            str(cli.work_dir / 'hessians' / f'{cli.layer}_{group}.pt'),
                            cb, args, 'cuda:0')
                        torch.cuda.synchronize()
                        wall = time.perf_counter() - begin
                        current = torch.load(path, weights_only=True)
                        if not instrument:
                            saved_indices = current['Qidxs']
                            times['no_fine_events_wall_seconds'] = wall
                        else:
                            assert torch.equal(saved_indices, current['Qidxs'])
                            stages = dict(measured)
                            stages['rotation_and_phase_generation'] = (
                                stages.pop('rotation_preprocessing_including_cholesky')
                                - stages['preprocess_cholesky'])
                            stages['hessian_and_weight_preparation'] = quant_bounds['start'] - begin
                            stages['other_transfers_cleanup_and_bookkeeping'] = wall - sum(stages.values())
                            fine = timer.collect()
                            expected_calls = 11 * weights[0].shape[1] // 8
                            assert fine['codeword_selection']['calls'] == expected_calls
                            fine['error_compensation_and_buffer_updates'] = dict(
                                stream_ms=fine['ldlq_total']['stream_ms'] - fine['codeword_selection']['stream_ms'],
                                cpu_enqueue_ms=fine['ldlq_total']['cpu_enqueue_ms'] - fine['codeword_selection']['cpu_enqueue_ms'],
                                calls=expected_calls)
                            times.update(profiled_wall_seconds=wall,
                                fine_event_wall_ratio=wall / times['no_fine_events_wall_seconds'],
                                stage_wall_seconds=stages, ldlq_detail=fine,
                                identical_packed_indices=True)
                    timer.enabled = False
                    record = dict(mode=mode, layer=cli.layer, group=group,
                                  shape=[sum(w.shape[0] for w in weights), weights[0].shape[1]], **times)
                    result['offline'].append(record)
                    save(cli, result)
                    print('OFFLINE', json.dumps(record), flush=True)
                    del weights, current, saved_indices
                    utils.clean()
    finally:
        for obj, name, original in reversed(patches):
            setattr(obj, name, original)


class PackedForward(nn.Module):
    """Native RHT operation order; same E8 kernels with an RFFT rotation adapter."""
    def __init__(self, mode, timer, codebook_name='E8P12'):
        super().__init__()
        self.mode = mode
        self.timer = timer
        self.codebook_name = codebook_name
        self.codebook = codebook.codebook_id[codebook_name][1](inference=True).cuda().half()

    def split_indices(self, idxs):
        if self.codebook_name == 'E8P12':
            return [idxs]
        split = idxs.shape[-1] * 2 // 3 if self.codebook_name == 'E8P12RVQ3B' else idxs.shape[-1] // 2
        return [idxs[:, :split].contiguous(), idxs[:, split:].contiguous()]

    def forward(self, input, Qidxs_list, SU, SV, had_left_T, had_right,
                K_left, K_right, **kwargs):
        n, m = len(SU), len(SV)
        x = input.view(-1, n).float()
        with self.timer.section('activation_rotation'):
            if self.mode == 'had':
                x = matmul_hadU_cuda(x * SU, had_left_T, K_left) / 32
            else:
                x = matmul_rfft(x, SU.float()) / 32
        if kwargs.get('activation_quantizer') is not None:
            if self.mode != 'had':
                raise ValueError('Activation QDQ currently supports the Hadamard path only')
            with self.timer.section('activation_quantization'):
                x = kwargs['activation_quantizer'](x)
        idxs = Qidxs_list[0].view(m // 16, n // 64, 8, 4)
        residual_scale = getattr(self.codebook, 'opt_resid_scale', None)
        if x.shape[0] == 1:
            with self.timer.section('fused_weight_decode_matvec'):
                x16 = x.half()
                if self.codebook_name == 'E8P12RVQ3B':
                    # Preserve the reference 3-bit path: 1-bit residual uses k8 lookup.
                    padded = torch.zeros(8, n, dtype=torch.float16, device=x.device)
                    padded[0] = x16[0]
                    residual = torch.zeros(8, m, dtype=torch.float32, device=x.device)
                    quiptools_cuda.lookupmatmul_e81b_k8(
                        padded / residual_scale, Qidxs_list[1], self.codebook.e81b_grid, residual)
                    x = quiptools_cuda.decode_matvec_e8p(
                        x16[0], idxs, self.codebook.grid_packed_abs).float() + residual[0]
                else:
                    x = torch.ops.quip_lib.decode_matvec_e8p(
                        x16[0], idxs, self.codebook.grid_packed_abs, m, n).float()
                    if self.codebook_name == 'E8P12RVQ4B':
                        x = x + torch.ops.quip_lib.decode_matvec_e8p(
                            x16[0] / residual_scale,
                            Qidxs_list[1].view(m // 16, n // 64, 8, 4),
                            self.codebook.grid_packed_abs, m, n).float()
        else:
            with self.timer.section('weight_decode'):
                weight = torch.ops.quip_lib.decompress_packed_e8p(
                    idxs, self.codebook.grid_packed_abs, m, n)
                if self.codebook_name == 'E8P12RVQ3B':
                    residual = torch.zeros(m, n, dtype=torch.float16, device=x.device)
                    quiptools_cuda.decompress_e81b_packed(
                        Qidxs_list[1], self.codebook.e81b_grid, residual)
                    weight = weight + residual / residual_scale
                elif self.codebook_name == 'E8P12RVQ4B':
                    weight = weight + torch.ops.quip_lib.decompress_packed_e8p(
                        Qidxs_list[1].view(m // 16, n // 64, 8, 4),
                        self.codebook.grid_packed_abs, m, n) / residual_scale
            with self.timer.section('linear_matmul'):
                x = (x.half() @ weight.T).float()
        with self.timer.section('activation_rotation'):
            if self.mode == 'had':
                x = matmul_hadU_cuda(x, had_right, K_right) * SV * 32
            else:
                x = matmul_rfft_inverse(x, SV.float()) * 32
        return x.view(*input.shape[:-1], m)


def build_compressed(cli, mode, timer, index, codebook_name='E8P12', checkpoint_dir=None,
                     native=False):
    times = {}
    start = time.perf_counter()
    config = AutoConfig.from_pretrained(cli.base_model, local_files_only=True)
    config._attn_implementation = 'sdpa'
    cb = codebook.get_codebook(codebook_name)
    checkpoint_dir = checkpoint_dir or cli.work_dir / mode
    config.quip_params = dict(codesz=cb.codesz, packsz=cb.packsz,
        pack_out=cb.pack_out, idx_dtype=str(cb.idx_dtype), codebook_version=cb.version,
        lora_rank=0, rescale_WH=False)
    with init_empty_weights():
        model = LlamaForCausalLM(config).half()
    for name, _ in list(model.named_parameters()):
        value = read_weight(cli, index, name)
        parent, attr = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), attr, nn.Parameter(value, requires_grad=False))
    for i, layer in enumerate(model.model.layers):
        for group, attr in [('qkv', 'self_attn.qkv_proj'), ('o', 'self_attn.o_proj'),
                            ('up', 'mlp.upgate_proj'), ('down', 'mlp.down_proj')]:
            saved = torch.load(checkpoint_dir / f'{i}_{group}.pt', weights_only=True)
            module = layer.get_submodule(attr)
            utils.unpack_quip(module, saved, codebook.get_id(codebook_name), cb.codesz)
            if saved['fused']:
                module.fuse_scales.copy_(torch.tensor([float(s) for s in saved['scales']]))
    times['construct_and_checkpoint_read_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    model.cuda().eval()
    torch.cuda.synchronize()
    if native:
        if mode != 'had':
            raise ValueError('Native compressed model requires Hadamard checkpoints')
    else:
        shared = PackedForward(mode, timer, codebook_name)
        for module in model.modules():
            if isinstance(module, QuantizedLinear):
                module.Qidxs_list = shared.split_indices(module.Qidxs)
                if len(module.Qidxs_list) > 1:
                    del module.Qidxs
                module.codebook_class = shared
                module.built_codebook_class = True
    times['h2d_and_kernel_setup_seconds'] = time.perf_counter() - start
    # Full-size actual checkpoint validation for prefill-like and decode-like shapes.
    decoder = codebook.get_codebook(codebook_name).cuda()
    errors = []
    for group, attr in [('qkv', 'self_attn.qkv_proj'), ('o', 'self_attn.o_proj'),
                        ('up', 'mlp.upgate_proj'), ('down', 'mlp.down_proj')]:
        module = model.model.layers[cli.layer].get_submodule(attr)
        ws = restore(checkpoint_dir / f'{cli.layer}_{group}.pt', decoder,
                     SimpleNamespace(incoh_mode=mode, rescale_WH=False), 'cuda:0')
        for rows in (1, 4):
            x = torch.randn(rows, module.in_features, device='cuda', dtype=torch.float16)
            actual = module(x)
            actual = torch.cat(actual, dim=-1) if isinstance(actual, tuple) else actual
            expected = torch.cat([x.float() @ w.cuda().float().T for w in ws], dim=-1)
            error = ((actual.float() - expected).norm() / expected.norm()).item()
            assert error < .01, (mode, group, rows, error)
            errors.append(dict(group=group, rows=rows, relative_error=error))
    del decoder, ws
    utils.clean()
    return model, times, errors


def online(cli, result, index, prepared=None, output_records=None):
    tokenizer = AutoTokenizer.from_pretrained(cli.base_model, local_files_only=True, use_fast=False)
    import pandas as pd
    text = '\n\n'.join(pd.read_parquet(cli.work_dir / 'data/wikitext-2-raw-test.parquet')['text'])
    ids = tokenizer(text, return_tensors='pt').input_ids[:, :cli.context].cuda()
    if output_records is None:
        output_records = result.setdefault('online', {})
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    original_cache_update = DynamicCache.update
    for mode in cli.online_modes:
        timer = StreamTimer()
        start = time.perf_counter()
        if prepared is not None:
            model, startup, errors, timer = prepared[mode]
        elif mode == 'baseline':
            model = AutoModelForCausalLM.from_pretrained(cli.base_model,
                local_files_only=True, torch_dtype=torch.float16,
                attn_implementation='sdpa', low_cpu_mem_usage=True).cuda().eval()
            torch.cuda.synchronize()
            startup = dict(load_and_h2d_seconds=time.perf_counter() - start)
            errors = []
        else:
            model, startup, errors = build_compressed(cli, mode, timer, index)

        def sdpa(*a, **kw):
            with timer.section('attention_sdpa'):
                return original_sdpa(*a, **kw)
        torch.nn.functional.scaled_dot_product_attention = sdpa

        def cache_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            # Legacy-cache reconstruction appends existing tensors without GPU work.
            if cache_kwargs is None:
                return original_cache_update(self, key_states, value_states, layer_idx, cache_kwargs)
            with timer.section('kv_cache_update'):
                return original_cache_update(self, key_states, value_states, layer_idx, cache_kwargs)
        DynamicCache.update = cache_update
        # Dense baseline projections and the FP16 language head are timed alike.
        dense_originals = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                original = module.forward
                def dense(*a, _original=original, **kw):
                    with timer.section('linear_matmul'):
                        return _original(*a, **kw)
                dense_originals.append((module, original))
                module.forward = dense

        def prefill():
            output = model(ids, use_cache=True)
            assert torch.isfinite(output.logits[:, -1]).all()
            return output

        def decode_run(steps, profile=False):
            prefix = cli.context - cli.decode_steps
            output = model(ids[:, :prefix], use_cache=True)
            cache = output.past_key_values
            del output
            torch.cuda.synchronize()
            timer.reset()
            timer.enabled = profile
            latencies = []
            start = time.perf_counter()
            for j in range(steps):
                # Synchronize per token only at the externally observable latency boundary.
                begin = time.perf_counter()
                output = model(ids[:, prefix+j:prefix+j+1], past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - begin) * 1000)
            wall = time.perf_counter() - start
            assert torch.isfinite(output.logits).all()
            timer.enabled = False
            detail = timer.collect() if profile else None
            del output, cache
            return dict(wall_seconds=wall, ms_per_token_mean=statistics.mean(latencies),
                        ms_per_token_median=statistics.median(latencies),
                        output_tokens_per_second=steps/wall, token_latencies_ms=latencies,
                        detail=detail)

        try:
            # Initialize kernels, caches and allocator outside the measured runs.
            output = prefill()
            del output
            torch.cuda.synchronize()
            decode_run(4)
            prefill_times = []
            for _ in range(cli.repeats):
                start = time.perf_counter()
                output = prefill()
                torch.cuda.synchronize()
                prefill_times.append(time.perf_counter() - start)
                del output
            decodes = [decode_run(cli.decode_steps) for _ in range(cli.repeats)]
            timer.reset()
            timer.enabled = True
            start = time.perf_counter()
            output = prefill()
            torch.cuda.synchronize()
            profiled_prefill_wall = time.perf_counter() - start
            timer.enabled = False
            prefill_detail = timer.collect()
            del output
            profiled_decode = decode_run(cli.decode_steps, profile=True)
            record = dict(startup=startup, checkpoint_operator_validation=errors,
                prefill_wall_seconds=prefill_times,
                prefill_mean_seconds=statistics.mean(prefill_times),
                prefill_input_tokens_per_second=cli.context/statistics.mean(prefill_times),
                decode_runs=decodes, profiled_prefill_wall_seconds=profiled_prefill_wall,
                prefill_detail=prefill_detail, profiled_decode=profiled_decode,
                prefill_instrumentation_wall_ratio=profiled_prefill_wall/statistics.mean(prefill_times),
                decode_instrumentation_wall_ratio=profiled_decode['wall_seconds']/statistics.mean(r['wall_seconds'] for r in decodes),
                peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            output_records[mode] = record
            save(cli, result)
            print('ONLINE', mode, 'prefill_s', record['prefill_mean_seconds'],
                  'decode_ms', statistics.mean(r['ms_per_token_mean'] for r in decodes),
                  'prefill_detail', prefill_detail,
                  'decode_detail', profiled_decode['detail'], flush=True)
        finally:
            torch.nn.functional.scaled_dot_product_attention = original_sdpa
            DynamicCache.update = original_cache_update
            for module, original in dense_originals:
                module.forward = original
            # Remove closure/list references before the next full model is loaded.
            dense_originals.clear()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def write_model_comparison_reports(cli, result):
    """Update only the model-specific section, preserving the original 7B table."""
    name = result.get('model_name', cli.experiment_key)
    baseline = result.get('activation_baseline', {})
    reference = baseline.get('ppl') if baseline.get('status') == 'complete' else None
    order = [(2, 0), (3, 0), (4, 0), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]
    sweep = result.get('weight_activation_ppl', {})
    quality = ['| 配置 | 权重码本 | 激活处理 | WikiText2 PPL ↓ | 相对同位宽 A16 的 ΔPPL | 测评流程耗时（s） |',
               '| --- | --- | --- | ---: | ---: | ---: |']
    timing = ['| 配置 | 完整 PPL 测评流程（s） | 相对同位宽 A16 的流程耗时倍数 |',
              '| --- | ---: | ---: |']
    completed = int(reference is not None)
    if reference:
        assert baseline['settings']['base_model'] == str(cli.base_model)
        assert reference['blocks'] == 83 and reference['context_length'] == 4096
        assert reference['scored_tokens'] == 339885
        quality.append(f'| FP16（基线） | — | FP16 | {reference["ppl"]:.6f} | — | {baseline["ppl_evaluation_seconds"]:.2f} |')
        timing.append(f'| FP16 | {baseline["ppl_evaluation_seconds"]:.2f} | — |')
    else:
        quality.append('| FP16（基线） | — | FP16 | 待测 | — | — |')
        timing.append('| FP16 | 待测 | — |')
    for w, a in order:
        parent = sweep.get(str(w), {})
        run = parent.get('activation', {}).get(str(a), {})
        control = parent.get('activation', {}).get('0', {})
        label = f'W{w}A{a or 16}'
        activation = f'{CODEBOOKS[a]} QDQ' if a else 'FP16'
        if run.get('status') != 'complete':
            quality.append(f'| {label} | {CODEBOOKS[w]} | {activation} | 待测 | — | — |')
            timing.append(f'| {label} | 待测 | — |')
            continue
        assert reference is not None
        assert run['timing_valid_without_detected_contention']
        assert not run['other_gpu_processes_after_ppl']
        for key in ('blocks', 'test_tokens', 'scored_tokens', 'dropped_tail_tokens',
                    'context_length', 'test_parquet_sha256'):
            assert run['ppl'][key] == reference[key], (w, a, key)
        assert run['activation_quantization'] == activation_config(
            dict(bits=a, group_size=128, chunk_size=65536))
        settings = parent['settings']
        assert settings['base_model'] == baseline['settings']['base_model']
        assert settings['seed'] == 0 and settings['ldlq_local_search_passes'] == 10
        assert 'native packed QuIP#' in run['ppl']['evaluation']
        delta = run['ppl']['ppl'] - control['ppl']['ppl']
        seconds = run['ppl_evaluation_seconds']
        quality.append(f'| {label} | {CODEBOOKS[w]} | {activation} | {run["ppl"]["ppl"]:.6f} | {delta:+.6f} | {seconds:.2f} |')
        timing.append(f'| {label} | {seconds:.2f} | {seconds / control["ppl_evaluation_seconds"]:.2f}× |')
        completed += 1
    common = (f'{name}；除模型及对应 Hessian 外，沿用上方 7B 配置：Hadamard、no FT、'
        'WikiText2 test、context=4096、batch=1、seed=0、8 CPU 线程、TF32 关闭；'
        '83 区块、339885 计分 token。原生压缩 QuIP# 线性层、SDPA、eager、无 KV cache。'
        'A2/A3/A4 使用旋转后 E8/RVQ QDQ、每 token 每 128 通道 RMS、chunk=65536；'
        '沿用相同码本尺度。权重使用初次 BlockLDLQ + 10 轮逐块局部搜索。')
    timer_note = ('测评时间沿用上方 `evaluate()` wall time 口径；模型加载和 8-token warmup 在计时前完成。'
        '离线量化单列，同一 W 位宽的所有激活配置复用同一次量化权重；下载不计入量化或 PPL 时间。'
        '每次测评后校验数据 SHA256、区块数及 GPU 争用记录。')
    offline = []
    for bits, parent in sorted(sweep.items()):
        count = len(parent.get('quantized_matrices', []))
        offline.append(f'W{bits}：已量化 {count}/160 个矩阵，累计 {parent.get("quantization_seconds", 0):.2f} s')
    progress = ('完整测评已完成 10/10。' if completed == 10 else
                f'完整测评已完成 {completed}/10；待测项不代表实验结果。')
    sources = result.get('sources', {})
    provenance = ('模型来源与校验记录：`quantization-profile.json` → '
        f'`model_comparisons.{cli.experiment_key}`；日志继续使用 `quantization-profile.log`。')
    if sources:
        provenance += (f'模型 `{sources["model"]["repo"]}`（revision `{sources["model"]["revision"]}`）；'
            f'Hessian `{sources["hessian"]["repo"]}`（revision `{sources["hessian"]["revision"]}`），'
            '下载后校验文件大小与 LFS SHA256。')
    for filename, table in [('RESULTS.md', quality), ('TIMING_ANALYSIS.md', timing)]:
        path = cli.work_dir / filename
        existing = path.read_text()
        heading = f'## {name}：与上方 7B 相同配置的完整对比'
        content = '\n\n'.join([heading, progress, common, '\n'.join(table), timer_note,
                               '；'.join(offline) + '。' if offline else '离线量化尚未开始。', provenance])
        if heading in existing:
            start = existing.index(heading)
            end = existing.index('### 128-token 流程检查', start)
        else:
            start = end = existing.index('### 128-token 流程检查')
        temporary = path.with_suffix('.md.tmp')
        temporary.write_text(existing[:start] + content + '\n\n' + existing[end:])
        temporary.replace(path)


def write_activation_reports(cli, result):
    """Publish the requested ten-row comparison after validating complete records."""
    if getattr(cli, 'experiment_key', None):
        write_model_comparison_reports(cli, result)
        return
    baseline = result.get('activation_baseline', {})
    if baseline.get('status') != 'complete':
        return
    runs, sources = {}, {}
    for a in (0, 2, 3, 4):
        matches = [(key, r) for key, r in result.get('activation_comparison', {}).items()
            if r.get('status') == 'complete' and r['settings']['weight_bits'] == 2
            and r['settings']['context'] == 4096 and r['settings']['ppl_blocks'] == 0
            and r['settings']['activation_quantization']['bits'] == a
            and (a == 0 or r['settings']['activation_quantization']['chunk_size'] == 65536)]
        if not matches:
            return
        assert len(matches) == 1, (a, [key for key, _ in matches])
        sources[a], runs[2, a] = matches[0]
    for w, activations in ((3, (0, 3, 4)), (4, (0, 4))):
        parent = result.get('weight_activation_ppl', {}).get(str(w), {})
        if parent.get('status') != 'complete':
            return
        for a in activations:
            record = parent.get('activation', {}).get(str(a), {})
            if record.get('status') != 'complete':
                return
            runs[w, a] = record
    reference = baseline['ppl']
    metric_keys = ('blocks', 'test_tokens', 'scored_tokens', 'dropped_tail_tokens',
                   'context_length', 'test_parquet_sha256')
    assert reference['blocks'] == 83 and reference['scored_tokens'] == 339885
    assert reference['context_length'] == 4096
    for (w, a), record in runs.items():
        assert record['timing_valid_without_detected_contention']
        assert not record['other_gpu_processes_after_ppl']
        assert all(record['ppl'][key] == reference[key] for key in metric_keys)
        config = record.get('activation_quantization') or record['settings']['activation_quantization']
        assert config == activation_config(dict(bits=a, group_size=128,
            chunk_size=65536 if a or w != 2 else 4096))
        assert 'native packed QuIP#' in record['ppl']['evaluation']
        settings = (record['settings'] if w == 2 else
                    result['weight_activation_ppl'][str(w)]['settings'])
        assert settings['base_model'] == baseline['settings']['base_model']
        assert settings['seed'] == baseline['settings']['seed'] == 0
    rebuild = result['weight_activation_ppl']['3']['checkpoint_rebuilds'][-1]
    validation = rebuild['a16_rebuild_validation']
    assert validation['absolute_ppl_difference'] <= validation['tolerance'] == 1e-6
    order = [(2, 0), (3, 0), (4, 0), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]
    quality = [
        '| 配置 | 权重码本 | 激活处理 | WikiText2 PPL ↓ | 相对同位宽 A16 的 ΔPPL | 测评流程耗时（s） |',
        '| --- | --- | --- | ---: | ---: | ---: |',
        f'| FP16（基线，本次统一口径补测） | — | FP16 | {reference["ppl"]:.6f} | — | {baseline["ppl_evaluation_seconds"]:.2f} |']
    timing = ['| 配置 | 完整 PPL 测评流程（s） | 相对同位宽 A16 的流程耗时倍数 |',
        '| --- | ---: | ---: |',
        f'| FP16（本次统一口径补测） | {baseline["ppl_evaluation_seconds"]:.2f} | — |']
    for w, a in order:
        record = runs[w, a]
        name = f'W{w}A{a or 16}'
        ppl, seconds = record['ppl']['ppl'], record['ppl_evaluation_seconds']
        delta = ppl - runs[w, 0]['ppl']['ppl']
        activation = f'{CODEBOOKS[a]} QDQ' if a else 'FP16（激活量化关闭）'
        quality.append(f'| {name} | {CODEBOOKS[w]} | {activation} | {ppl:.6f} | {delta:+.6f} | {seconds:.2f} |')
        ratio = seconds / runs[w, 0]['ppl_evaluation_seconds']
        timing.append(f'| {name} | {seconds:.2f} | {ratio:.2f}× |')
    common = ('Llama2-7B、Hadamard、no FT；WikiText2 test、context=4096、batch=1、'
        '83 个完整区块、339885 个计分 token。测试集共 341469 token，尾部 1501 token 丢弃；'
        '所有配置数据 SHA256 一致。九种量化配置均使用原生压缩 QuIP# 线性层和 SDPA，'
        'A16 表示关闭激活量化（`bits=0`）。eager、无 CUDA Graph/compile、无 KV cache；'
        '固定 seed=0、8 个 CPU 线程、关闭 TF32。')
    activation_note = ('A2/A3/A4 均在 decoder 线性层输入的 Hadamard 旋转后执行 E8/RVQ 量化后反量化；'
        '每 token 每 128 通道动态 RMS，搜索批次统一为 65536 个 8D 向量。'
        '各码本默认 rho=1.03/0.98/1.03，A3/A4 残差尺度=2.04/3.45，不额外调参。'
        'A2 为单级 E8P，A3 为 2+1 bit RVQ，A4 为 2+2 bit RVQ。'
        'embedding、LM head、attention 内部计算及 KV cache 未启用激活量化。'
        '计算前重建浮点，尚无低比特激活计算内核。A16 不执行激活搜索；'
        '历史 W2A16 的 chunk=4096 不影响该关闭路径，其余完整配置为 65536。')
    timer_note = ('时间统一统计 `evaluate` 的 wall time：包括 Parquet 读取、tokenization、'
        '`model.cuda().eval()`、83 次前向、交叉熵及结果统计。所有配置在计时前完成模型/检查点构建加载、'
        'tokenizer 初始化、模型搬至 GPU、8-token warmup 与 CUDA 同步；不含离线量化/重建准备。'
        'FP16 本次按相同口径补测，替换统一表中的历史时间，`baseline/metrics.json` 原记录保留。'
        '不同日期运行仍有正常波动；整段 PPL 流程时间不是独立 prefill/decode 或纯算子延迟。'
        '测评前后检查 GPU 计算进程，发现争用则等待/重试，检查之间的短暂争用仍无法完全排除。')
    weight_note = ('W2A16/A2/A3/A4 复用同一已有 W2 检查点。W3A3 按原 seed、预计算 Hessian、'
        '初次 BlockLDLQ + 10 轮逐块局部搜索、默认码本尺度重建临时权重；'
        f'新增测评前完整复测 W3A16，PPL={validation["ppl"]["ppl"]:.9f}，'
        f'与原对照绝对差={validation["absolute_ppl_difference"]:.9g}（容差 1e-6），数据 SHA256 一致。'
        'W3/W4 原有 A16/A4 结果沿用；原始离线计时保留，重建另存 `checkpoint_rebuilds`，'
        '不计入 PPL 测评。临时检查点完成后清理。')
    source_note = ('原始记录均在既有 `quantization-profile.json`：`activation_comparison` '
        f'（W2A16=`{sources[0]}`、W2A2=`{sources[2]}`、W2A3=`{sources[3]}`、W2A4=`{sources[4]}`）、'
        '`weight_activation_ppl`（W3/W4）、`activation_baseline`（本次 FP16）。日志沿用 '
        '`quantization-profile.log`。表格覆盖 FP16 加九种量化配置；W3A2、W4A2、W4A3 尚未完整测评。')
    history_note = ('下文历史 weight-only PPL（8.222629 / 5.604216 / 5.267964）使用提前还原 FP16 '
        '权重后的 HF SDPA 路径，其约 30 秒不含还原；本表 A16 使用原生压缩线性层，约 51–58 秒。'
        '两组执行路径不同，不能混用时间或用作激活量化的精确对照。机制与运行方式见 '
        '[功能模块文档](../quip-sharp/FUNCTIONAL_MODULES.md#4-weight-only-与-weight-activation-ptq)。')
    offline_note = ('离线费用单列：W2 复用已有检查点；历史 W3/W4 128 个矩阵的有效量化时间累计 '
        f'1266.56 / 1797.86 秒。本次 W3 重建累计 {rebuild["quantization_seconds"]:.2f} 秒，'
        '另存 `checkpoint_rebuilds`；不计入上表，也不为每种激活位宽重复计费。'
        '下文 LDLQ/局部搜索分项仍来自历史独立测试。')
    analysis_note = ('激活侧不执行 BlockLDLQ 或 10 轮逐块局部搜索。新增在线工作主要为码字选择及浮点重建；'
        'A4 是两级完整 E8P 搜索，A3 是 E8P 加较小残差码本。完整模型未逐算子拆分，不能把相对 '
        'A16 的全部差额都归为纯码字选择或纯 E8 解码。此前单张量 QDQ 微基准中的候选评分、矩乘和 '
        'argmax 占比不能直接套用于整模型。')
    sections = [
        ('RESULTS.md', '## FP16、Weight-only', '\n\n'.join([
            '## FP16、Weight-only 与 W2/W3/W4 A2/A3/A4 完整对比',
            '更新日期：2026-10-23。新增 W2A2、W2A3、W3A3，补测相同预热/计时口径的 FP16 基线。',
            common, '\n'.join(quality), activation_note, weight_note, timer_note, source_note, history_note])),
        ('TIMING_ANALYSIS.md', '## W2/W3/W4 A16', '\n\n'.join([
            '## W2/W3/W4 A16 与 A2/A3/A4：完整 WikiText2 测评耗时',
            '更新日期：2026-10-23。完整 PPL 大表见 [RESULTS.md](RESULTS.md)。',
            common, '\n'.join(timing), timer_note, activation_note, analysis_note,
            offline_note, weight_note, source_note]))]
    for filename, heading, content in sections:
        path = cli.work_dir / filename
        existing = path.read_text()
        start = existing.index(heading)
        end = existing.index('### 128-token 流程检查', start)
        updated = existing[:start] + content + '\n\n' + existing[end:]
        updated = updated.replace('A2/A3 完整测试尚未执行，因此目前不能给出完整 W2A2/W2A3 PPL。',
            '完整 W2A2/W2A3 PPL 已补测，见本文首表。')
        updated = updated.replace('A2/A3 完整测试尚未执行。',
            '完整 W2A2/W2A3/W3A3 已补测，见本文首表。')
        path.write_text(updated)
    functional = Path(__file__).resolve().parents[1] / 'FUNCTIONAL_MODULES.md'
    existing = functional.read_text().replace('核查日期：2026-10-22。', '核查日期：2026-10-23。')
    existing = existing.replace('尤其 A2/A3 的完整 PPL 尚需验证。',
        'W2A2/W2A3/W3A3 已完成完整 PPL，但默认尺度未证明最优。')
    existing = existing.replace('七种配置（含 FP16）的统一大表', '十种配置（含 FP16）的统一大表')
    existing = existing.replace('A2/A3 完整 PPL 尚未测试；',
        f'2026-10-23 补测完整 W2A2/W2A3/W3A3，PPL 分别为 {runs[2, 2]["ppl"]["ppl"]:.6f} / '
        f'{runs[2, 3]["ppl"]["ppl"]:.6f} / {runs[3, 3]["ppl"]["ppl"]:.6f}；'
        'W3 重建通过完整 A16 PPL 对照校验，并按相同预热/计时口径补测 FP16。'
        'W3A2/W4A2/W4A3 尚未完整测评；')
    functional.write_text(existing)
    print('UPDATED existing activation comparison reports after full validation', flush=True)


def activation_baseline(cli, result):
    """FP16 control with the same warmup and evaluate timer as native QDQ runs."""
    if cli.context != 4096 or cli.ppl_blocks:
        raise ValueError('The FP16 control requires full WikiText2 with context=4096')
    if result.get('activation_baseline', {}).get('status') == 'complete':
        print('FP16_ACTIVATION_BASELINE already complete', flush=True)
        write_activation_reports(cli, result)
        return
    wait_for_exclusive_gpu()
    model = AutoModelForCausalLM.from_pretrained(cli.base_model,
        local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation='sdpa').cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(cli.base_model, local_files_only=True, use_fast=False)
    model(torch.ones(1, 8, device='cuda', dtype=torch.long), use_cache=False)
    torch.cuda.synchronize()
    record = dict(status='running', settings=dict(base_model=str(cli.base_model),
        context=cli.context, batch_size=1, seed=0, dtype='float16',
        attention='sdpa', warmup_tokens=8, use_cache=False),
        discarded_contended_attempts=[])
    result['activation_baseline'] = record
    save(cli, result)
    while True:
        if other_gpu_processes():
            wait_for_exclusive_gpu()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        record['ppl'] = evaluate(model, tokenizer,
            cli.work_dir / 'data/wikitext-2-raw-test.parquet', cli.context,
            evaluation='original FP16 weights -> HF SDPA; warmup before evaluate timer')
        torch.cuda.synchronize()
        record['ppl_evaluation_seconds'] = time.perf_counter() - start
        others = other_gpu_processes()
        if not others:
            break
        record['discarded_contended_attempts'].append(dict(
            seconds=record['ppl_evaluation_seconds'], other_pids=others))
        save(cli, result)
    record.update(status='complete', other_gpu_processes_after_ppl=others,
        timing_valid_without_detected_contention=True,
        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
    save(cli, result)
    print('FP16_ACTIVATION_BASELINE_COMPLETE', record, flush=True)
    write_activation_reports(cli, result)


def activation_comparison(cli, result, index):
    """PPL uses native compressed linears, including online activation QDQ."""
    config = activation_config(activation_overrides(cli))
    checkpoint_dir = cli.checkpoint_dir or cli.work_dir / 'had'
    cb_name = CODEBOOKS[cli.weight_bits]
    manifest = checkpoint_dir / 'config.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved.get('incoh_mode', 'had') != 'had' or saved.get('codebook', cb_name) != cb_name:
            raise ValueError('Checkpoint rotation/codebook does not match the requested Hadamard weight bits')
        if saved.get('rescale_WH', False) or saved.get('resid_scale_override', -1) > 0:
            raise ValueError('This benchmark requires checkpoints without rescale_WH or residual-scale overrides')
    elif cli.checkpoint_dir is None and cli.weight_bits != 2:
        raise ValueError('Supply --checkpoint-dir containing the requested 3/4-bit checkpoints')
    for layer in range(AutoConfig.from_pretrained(cli.base_model).num_hidden_layers):
        for group, _ in GROUPS:
            if not (checkpoint_dir / f'{layer}_{group}.pt').is_file():
                raise ValueError(f'Missing packed checkpoint: {layer}_{group}.pt')
    settings = dict(base_model=str(cli.base_model), checkpoint_dir=str(checkpoint_dir.resolve()),
                    weight_bits=cli.weight_bits, activation_quantization=config,
                    context=cli.context, ppl_blocks=cli.ppl_blocks, seed=0)
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    previous = result.get('activation_comparison', {}).get(key)
    if previous and previous.get('status') == 'complete':
        print('ACTIVATION already complete:', key, flush=True)
        write_activation_reports(cli, result)
        return
    record = dict(settings=settings, status='running')
    result.setdefault('activation_comparison', {})[key] = record
    save(cli, result)
    wait_for_exclusive_gpu()
    timer = StreamTimer()
    model, startup, errors = build_compressed(cli, 'had', timer, index, cb_name,
                                             checkpoint_dir, native=True)
    record['startup'] = startup
    record['weight_only_operator_validation'] = errors
    configure_activation_quantization(model, config)
    tokenizer = AutoTokenizer.from_pretrained(cli.base_model, local_files_only=True, use_fast=False)
    # Initialize lazy codebooks and kernels before timing PPL.
    warmup = torch.ones(1, min(8, cli.context), device='cuda', dtype=torch.long)
    model(warmup, use_cache=False)
    torch.cuda.synchronize()
    record['discarded_contended_attempts'] = []
    while True:
        if other_gpu_processes():
            wait_for_exclusive_gpu()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        record['ppl'] = evaluate(model, tokenizer, cli.work_dir / 'data/wikitext-2-raw-test.parquet',
            cli.context, max_blocks=cli.ppl_blocks or None,
            evaluation='native packed QuIP# linears + rotated E8/RVQ activation QDQ -> SDPA'
                       if config['bits'] else 'native packed QuIP# weight-only linears -> SDPA')
        torch.cuda.synchronize()
        record['ppl_evaluation_seconds'] = time.perf_counter() - start
        others = other_gpu_processes()
        if not others:
            break
        record['discarded_contended_attempts'].append(dict(
            seconds=record['ppl_evaluation_seconds'], other_pids=others))
        save(cli, result)
    record['other_gpu_processes_after_ppl'] = others
    record['timing_valid_without_detected_contention'] = not record['other_gpu_processes_after_ppl']
    record['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    record['status'] = 'complete'
    save(cli, result)
    print('ACTIVATION_COMPLETE', key, record, flush=True)
    del model
    utils.clean()
    write_activation_reports(cli, result)


def activation_bit_sweep(cli, result, index):
    """Recreate checkpoints transiently and add requested native activation PPL."""
    if cli.context != 4096 or cli.ppl_blocks:
        raise ValueError('The activation bit sweep requires full WikiText2 with context=4096')
    sweep = result.setdefault('weight_activation_ppl', {})
    tokenizer = AutoTokenizer.from_pretrained(cli.base_model, use_fast=False, local_files_only=True)
    requested_activations = tuple(dict.fromkeys(cli.comparison_activation_bits))
    for bits in cli.comparison_bits:
        cb_name = CODEBOOKS[bits]
        settings = dict(base_model=str(cli.base_model), weight_bits=bits, rotation='had',
            context=cli.context, seed=0, ft_epochs=0, ldlq_local_search_passes=10,
            activation_bits=list(requested_activations), activation_group_size=128,
            activation_chunk_size=cli.activation_chunk_size or 4096,
            weight_scale_override={2: .9, 3: .98, 4: 1.03}[bits],
            weight_residual_scale={2: None, 3: 2.04, 4: 3.45}[bits])
        record = sweep.setdefault(str(bits), dict(settings=settings, status='pending',
            quantized_matrices=[], activation={}))
        if ({k: v for k, v in record['settings'].items() if k != 'activation_bits'} !=
                {k: v for k, v in settings.items() if k != 'activation_bits'}):
            raise ValueError(f'Existing W{bits} native activation PPL uses different settings')
        record['settings']['activation_bits'] = sorted(set(
            record['settings']['activation_bits']) | set(requested_activations))
        if all(record.get('activation', {}).get(str(a), {}).get('status') == 'complete'
               for a in requested_activations):
            print('NATIVE_ACTIVATION_SWEEP already complete:', bits, flush=True)
            continue
        # Retain the original offline measurements when rebuilding deleted checkpoints.
        if record.get('quantized_matrices'):
            quantization_record = dict(quantized_matrices=[], quantization_seconds=0.)
            record.setdefault('checkpoint_rebuilds', []).append(quantization_record)
        else:
            quantization_record = record
            record.update(quantized_matrices=[], quantization_seconds=0.)
        record['status'] = 'quantizing'
        save(cli, result)
        wait_for_exclusive_gpu()
        model = AutoModelForCausalLM.from_pretrained(cli.base_model,
            local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            attn_implementation='sdpa').eval()
        cb = codebook.get_codebook(cb_name)
        try:
            with tempfile.TemporaryDirectory(prefix=f'quip-w{bits}-native-ppl-', dir='/dev/shm') as td:
                checkpoint_dir = Path(td)
                args = SimpleNamespace(incoh_mode='had', use_fp64=False, rescale_WH=False,
                    lora_rank=0, full_svd=False, sigma_reg=.01, sigma_reg2=.01,
                    scale_override=.9 if bits == 2 else -1, resid_scale_override=-1,
                    no_use_buffered=False, lowmem_ldlq=False, quip_tune_iters=10, save_pfx=td)
                for i, layer in enumerate(model.model.layers):
                    torch.manual_seed(i)
                    for group, attrs in GROUPS:
                        modules = [layer.get_submodule(attr) for attr in attrs]
                        path = checkpoint_dir / f'{i}_{group}.pt'
                        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                        while True:
                            if other_gpu_processes():
                                wait_for_exclusive_gpu()
                            torch.set_rng_state(cpu_rng)
                            torch.cuda.set_rng_state(cuda_rng)
                            path.unlink(missing_ok=True)
                            start = time.perf_counter()
                            quip.quantize_linear([m.weight for m in modules],
                                str(path), str(cli.hessian_dir / path.name), cb, args, 'cuda:0')
                            torch.cuda.synchronize()
                            seconds = time.perf_counter() - start
                            others = other_gpu_processes()
                            if not others:
                                break
                            record.setdefault('discarded_contended_attempts', []).append(dict(
                                stage='quantization', layer=i, group=group,
                                quantization_seconds=seconds, other_pids=others))
                            save(cli, result)
                        quantization_record['quantized_matrices'].append(dict(layer=i, group=group, seconds=seconds))
                        quantization_record['quantization_seconds'] += seconds
                        save(cli, result)
                        print(f'NATIVE_ACTIVATION_MATRIX W{bits} {len(quantization_record["quantized_matrices"])}/{len(model.model.layers)*len(GROUPS)} '
                              f'layer={i} group={group} seconds={seconds:.2f}', flush=True)
                del modules, layer, cb, model
                utils.clean()
                record['status'] = 'building_native_model'
                save(cli, result)
                timer = StreamTimer()
                packed_model, startup, errors = build_compressed(cli, 'had', timer, index, cb_name,
                    checkpoint_dir, native=True)
                record['startup'] = startup
                record['weight_only_operator_validation'] = errors
                record['status'] = 'evaluating_native_ppl'
                save(cli, result)
                # Check that a rebuild reproduces the existing full native A16 control.
                reference = record['activation'].get('0', {})
                if reference.get('status') == 'complete':
                    configure_activation_quantization(packed_model, dict(
                        bits=0, group_size=128, chunk_size=cli.activation_chunk_size or 4096))
                    packed_model(torch.ones(1, 8, device='cuda', dtype=torch.long), use_cache=False)
                    torch.cuda.synchronize()
                    control = evaluate(packed_model, tokenizer,
                        cli.work_dir / 'data/wikitext-2-raw-test.parquet', cli.context,
                        evaluation='native packed QuIP# weight-only linears -> SDPA')
                    difference = abs(control['ppl'] - reference['ppl']['ppl'])
                    quantization_record['a16_rebuild_validation'] = dict(
                        ppl=control, reference_ppl=reference['ppl']['ppl'],
                        absolute_ppl_difference=difference, tolerance=1e-6)
                    save(cli, result)
                    if (difference > 1e-6 or
                            control['test_parquet_sha256'] != reference['ppl']['test_parquet_sha256']):
                        raise ValueError(f'W{bits} rebuilt weights do not reproduce the native A16 control')
                    print(f'NATIVE_REBUILD_VALIDATED W{bits} A16 PPL={control["ppl"]:.9f} '
                          f'difference={difference:.9g}', flush=True)
                for activation_bits in requested_activations:
                    activation_key = str(activation_bits)
                    if record['activation'].get(activation_key, {}).get('status') == 'complete':
                        continue
                    chunk = cli.activation_chunk_size or 4096
                    config = configure_activation_quantization(packed_model, dict(
                        bits=activation_bits, group_size=128, chunk_size=chunk))
                    activation_record = dict(status='running',
                        activation_quantization=config, discarded_contended_attempts=[])
                    record['activation'][activation_key] = activation_record
                    save(cli, result)
                    warmup = torch.ones(1, 8, device='cuda', dtype=torch.long)
                    packed_model(warmup, use_cache=False)
                    torch.cuda.synchronize()
                    while True:
                        if other_gpu_processes():
                            wait_for_exclusive_gpu()
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        activation_record['ppl'] = evaluate(packed_model, tokenizer,
                            cli.work_dir / 'data/wikitext-2-raw-test.parquet', cli.context,
                            evaluation='native packed QuIP# linears + rotated E8/RVQ activation QDQ -> SDPA'
                                if activation_bits else 'native packed QuIP# weight-only linears -> SDPA')
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - start
                        others = other_gpu_processes()
                        if not others:
                            break
                        activation_record['discarded_contended_attempts'].append(
                            dict(seconds=seconds, other_pids=others))
                        save(cli, result)
                    activation_record.update(status='complete', ppl_evaluation_seconds=seconds,
                        other_gpu_processes_after_ppl=others,
                        timing_valid_without_detected_contention=not others,
                        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
                    save(cli, result)
                    write_activation_reports(cli, result)
                    print(f'NATIVE_ACTIVATION_PPL W{bits}A{activation_bits} '
                          f'{activation_record["ppl"]["ppl"]:.6f} seconds={seconds:.2f}', flush=True)
                record['status'] = 'complete'
                save(cli, result)
                del packed_model
                utils.clean()
        except Exception as exc:
            record['status'] = 'failed'
            record['error'] = repr(exc)
            save(cli, result)
            raise
    write_activation_reports(cli, result)


def bit_comparison(cli, result, index):
    """Full 128-matrix quantization, common reconstructed-weight PPL and packed online timing."""
    comparison = result.setdefault('bit_comparison', {})
    config = dict(base_model=str(cli.base_model), context=cli.context,
        bits=list(CODEBOOKS), seed=0, incoherence='had', ft_epochs=0,
        # Historical JSON key: blockwise local-search passes after initial BlockLDLQ.
        refinement_iterations=10, hessian_samples=6144,
        scales={'2': .9, '3': .98, '4': 1.03}, residual_scales={'3': 2.04, '4': 3.45},
        repeats=cli.repeats, decode_steps=cli.decode_steps,
        checkpoint_storage='temporary RAM-backed /dev/shm; deleted after each bit width',
        offline_timing='All 128 quantize_linear calls, including Hessian read, diagnostics and RAM-backed checkpoint writes; LDLQ measured with synchronized coarse boundaries only',
        ppl_path='Packed indices reconstructed to FP16 weights, identical HF SDPA evaluation for all bit widths; no cache',
        online_timing='Separate full-model packed-weight prefill/decode; same reference operations for each codebook; CUDA events include stream idle/launch gaps',
        gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__)
    if 'config' in comparison and comparison['config'] != config:
        raise ValueError('Existing bit comparison has different settings; preserve or reconcile it before rerunning')
    comparison['config'] = config
    runs = comparison.setdefault('runs', {})
    tokenizer = AutoTokenizer.from_pretrained(cli.base_model, use_fast=False, local_files_only=True)
    for bits in cli.comparison_bits:
        if runs.get(str(bits), {}).get('status') == 'complete':
            print('BIT_COMPARISON already complete:', bits, flush=True)
            continue
        begin = time.perf_counter()
        cb_name = CODEBOOKS[bits]
        record = dict(status='quantizing', bits=bits, codebook=cb_name, matrices=[],
            gpu_exclusivity_checks='GPU-0 compute-process checks before/after each matrix, PPL and online suite; failed attempts retried',
            discarded_contended_attempts=[])
        runs[str(bits)] = record
        save(cli, result)
        model = AutoModelForCausalLM.from_pretrained(cli.base_model,
            local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            attn_implementation='sdpa').eval()
        cb = codebook.get_codebook(cb_name)
        decoder = codebook.get_codebook(cb_name).cuda()
        record['initial_model_and_codebook_load_seconds'] = time.perf_counter() - begin
        patches = []
        measured = {}

        def timed_ldl(original):
            def call(*a, **kw):
                torch.cuda.synchronize()
                start = time.perf_counter()
                value = original(*a, **kw)
                torch.cuda.synchronize()
                measured['ldlq_seconds'] = measured.get('ldlq_seconds', 0.) + time.perf_counter() - start
                return value
            return call

        with tempfile.TemporaryDirectory(prefix=f'quip-{bits}bit-', dir='/dev/shm') as td:
            checkpoint_dir = Path(td)
            args = SimpleNamespace(incoh_mode='had', use_fp64=False, rescale_WH=False,
                lora_rank=0, full_svd=False, sigma_reg=.01, sigma_reg2=.01,
                scale_override=.9 if bits == 2 else -1, resid_scale_override=-1,
                no_use_buffered=False, lowmem_ldlq=False, quip_tune_iters=10, save_pfx=td)
            for name in ('LDLQ', 'LDLQ_buffered', 'LDLQ_buffered_lowmem'):
                original = getattr(quip, name)
                patches.append((name, original))
                setattr(quip, name, timed_ldl(original))
            offline_start = time.perf_counter()
            try:
                for i, layer in enumerate(model.model.layers):
                    torch.manual_seed(i)
                    for group, attrs in GROUPS:
                        modules = [layer.get_submodule(attr) for attr in attrs]
                        path = checkpoint_dir / f'{i}_{group}.pt'
                        cpu_rng = torch.get_rng_state()
                        cuda_rng = torch.cuda.get_rng_state()
                        while True:
                            if other_gpu_processes():
                                wait_for_exclusive_gpu()
                            torch.set_rng_state(cpu_rng)
                            torch.cuda.set_rng_state(cuda_rng)
                            path.unlink(missing_ok=True)
                            measured.clear()
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            quip.quantize_linear([m.weight for m in modules], str(path),
                                str(cli.work_dir / 'hessians' / path.name), cb, args, 'cuda:0')
                            torch.cuda.synchronize()
                            quant_seconds = time.perf_counter() - start
                            others = other_gpu_processes()
                            if not others:
                                break
                            record['discarded_contended_attempts'].append(dict(
                                stage='quantization', layer=i, group=group,
                                quantization_seconds=quant_seconds, other_pids=others))
                            save(cli, result)
                            print('BIT_RETRY_GPU', bits, i, group, others, flush=True)
                        start = time.perf_counter()
                        for module, weight in zip(modules, restore(path, decoder, args, 'cuda:0')):
                            module.weight.copy_(weight)
                        torch.cuda.synchronize()
                        reconstruction_seconds = time.perf_counter() - start
                        utils.clean()
                        entry = dict(layer=i, group=group,
                            shape=[sum(m.weight.shape[0] for m in modules), modules[0].weight.shape[1]],
                            quantization_seconds=quant_seconds,
                            ldlq_seconds=measured['ldlq_seconds'],
                            reconstruction_for_ppl_seconds=reconstruction_seconds)
                        record['matrices'].append(entry)
                        save(cli, result)
                        print(f'BIT_MATRIX bits={bits} completed={len(record["matrices"])}/128 '
                              f'layer={i} group={group} quant_s={quant_seconds:.3f} '
                              f'ldlq_s={entry["ldlq_seconds"]:.3f}', flush=True)
            finally:
                for name, original in patches:
                    setattr(quip, name, original)
            record['pre_ppl_wall_seconds'] = time.perf_counter() - offline_start
            record['quantization_seconds'] = sum(x['quantization_seconds'] for x in record['matrices'])
            record['ldlq_seconds'] = sum(x['ldlq_seconds'] for x in record['matrices'])
            record['reconstruction_for_ppl_seconds'] = sum(x['reconstruction_for_ppl_seconds'] for x in record['matrices'])
            record['checkpoint_bytes'] = sum(p.stat().st_size for p in checkpoint_dir.glob('*.pt'))
            del layer, modules, module, weight
            del decoder, cb
            utils.clean()
            record['status'] = 'evaluating_ppl'
            save(cli, result)
            print('BIT_PPL_START', bits, flush=True)
            torch.cuda.reset_peak_memory_stats()
            while True:
                if other_gpu_processes():
                    wait_for_exclusive_gpu()
                start = time.perf_counter()
                record['ppl'] = evaluate(model, tokenizer,
                    cli.work_dir / 'data/wikitext-2-raw-test.parquet', cli.context)
                torch.cuda.synchronize()
                record['ppl_evaluation_seconds'] = time.perf_counter() - start
                others = other_gpu_processes()
                if not others:
                    break
                record['discarded_contended_attempts'].append(dict(
                    stage='ppl', seconds=record['ppl_evaluation_seconds'], other_pids=others))
                save(cli, result)
            record['workflow_through_ppl_seconds'] = time.perf_counter() - begin
            record['ppl_peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
            record['status'] = 'profiling_online'
            save(cli, result)
            print('BIT_PPL', bits, record['ppl']['ppl'], 'offline_s', record['quantization_seconds'],
                  'ldlq_s', record['ldlq_seconds'], 'ppl_s', record['ppl_evaluation_seconds'], flush=True)
            del model
            utils.clean()
            torch.cuda.reset_peak_memory_stats()
            timer = StreamTimer()
            packed_model, startup, errors = build_compressed(
                cli, 'had', timer, index, cb_name, checkpoint_dir)
            online_cli = SimpleNamespace(**{**vars(cli), 'online_modes': ['had']})
            while True:
                if other_gpu_processes():
                    wait_for_exclusive_gpu()
                online(online_cli, result, index,
                    prepared={'had': (packed_model, startup, errors, timer)},
                    output_records=record.setdefault('online', {}))
                others = other_gpu_processes()
                if not others:
                    break
                record['discarded_contended_attempts'].append(dict(
                    stage='online', other_pids=others))
                save(cli, result)
            del packed_model
            utils.clean()
            record['status'] = 'complete'
            record['experiment_including_validation_and_profiling_seconds'] = time.perf_counter() - begin
            save(cli, result)
            print('BIT_COMPLETE', bits, 'PPL', record['ppl']['ppl'], flush=True)


def retime_2bit(cli, result, index):
    """Replace interrupted timings; require bitwise-identical existing checkpoints."""
    record = result['bit_comparison']['runs']['2']
    assert record['status'] == 'complete'
    assert len(record['matrices']) == 128
    assert 0 <= cli.retime_from_layer < 32
    history = record.setdefault('timing_remeasurement', dict(
        reason='GPU contention and SIGSTOP/SIGCONT waiting interrupted the original timing',
        original_quantization_seconds=record['quantization_seconds'],
        original_ldlq_seconds=record['ldlq_seconds'],
        original_matrices=[dict(m) for m in record['matrices']], matrices=[]))
    history['status'] = 'running'
    record['original_workflow_time_valid_for_comparison'] = False
    save(cli, result)
    cb = codebook.get_codebook('E8P12')
    measured = {}
    original = quip.LDLQ_buffered

    def timed_ldl(*a, **kw):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = original(*a, **kw)
        torch.cuda.synchronize()
        measured['ldlq_seconds'] = time.perf_counter() - start
        return value

    quip.LDLQ_buffered = timed_ldl
    try:
        with tempfile.TemporaryDirectory(prefix='quip-2bit-retime-', dir='/dev/shm') as td:
            args = SimpleNamespace(incoh_mode='had', use_fp64=False, rescale_WH=False,
                lora_rank=0, full_svd=False, sigma_reg=.01, sigma_reg2=.01,
                scale_override=.9, resid_scale_override=-1, no_use_buffered=False,
                lowmem_ldlq=False, quip_tune_iters=10, save_pfx=td)
            for layer in range(cli.retime_from_layer, 32):
                wait_for_exclusive_gpu()
                torch.manual_seed(layer)
                for group, attrs in GROUPS:
                    weights = [read_weight(cli, index, f'model.layers.{layer}.{attr}.weight')
                               for attr in attrs]
                    path = Path(td) / f'{layer}_{group}.pt'
                    cpu_rng = torch.get_rng_state()
                    cuda_rng = torch.cuda.get_rng_state()
                    # A new competing process invalidates this attempt; retry with the same RNG.
                    while True:
                        if other_gpu_processes():
                            wait_for_exclusive_gpu()
                        torch.set_rng_state(cpu_rng)
                        torch.cuda.set_rng_state(cuda_rng)
                        path.unlink(missing_ok=True)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        quip.quantize_linear(weights, str(path),
                            str(cli.work_dir / 'hessians' / path.name), cb, args, 'cuda:0')
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        if not other_gpu_processes():
                            break
                        print('RETIME_RETRY contention', layer, group, flush=True)
                    actual = torch.load(path, weights_only=True, map_location='cpu')
                    expected = torch.load(cli.work_dir / 'had' / path.name,
                                          weights_only=True, map_location='cpu')
                    for key in ('Qidxs', 'SU', 'SV'):
                        assert torch.equal(actual[key], expected[key]), (layer, group, key)
                    entry = next(m for m in record['matrices']
                                 if m['layer'] == layer and m['group'] == group)
                    entry.update(quantization_seconds=elapsed,
                                 ldlq_seconds=measured['ldlq_seconds'])
                    history['matrices'] = [m for m in history['matrices']
                        if (m['layer'], m['group']) != (layer, group)]
                    history['matrices'].append(dict(layer=layer, group=group,
                        quantization_seconds=elapsed, ldlq_seconds=measured['ldlq_seconds'],
                        indices_and_rotation_scales_identical_to_original=True))
                    for key in ('quantization_seconds', 'ldlq_seconds'):
                        record[key] = sum(m[key] for m in record['matrices'])
                    save(cli, result)
                    print('RETIME_MATRIX', layer, group, elapsed, measured['ldlq_seconds'], flush=True)
                    del weights, actual, expected
                    utils.clean()
    finally:
        quip.LDLQ_buffered = original
    history['status'] = 'complete'
    save(cli, result)


def retime_2bit_evaluation(cli, result, index):
    """Repeat legacy 2-bit PPL/online measurements with GPU contention checks."""
    record = result['bit_comparison']['runs']['2']
    assert record['status'] == 'complete'
    history = record.setdefault('evaluation_remeasurement', dict(
        reason='Legacy 2-bit PPL/online run lacked GPU exclusivity checks',
        original_ppl_evaluation_seconds=record['ppl_evaluation_seconds'],
        original_online=json.loads(json.dumps(record['online']))))
    history['status'] = 'running'
    save(cli, result)
    wait_for_exclusive_gpu()
    model = AutoModelForCausalLM.from_pretrained(cli.base_model,
        local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation='sdpa').eval()
    decoder = codebook.get_codebook('E8P12').cuda()
    args = SimpleNamespace(incoh_mode='had', rescale_WH=False)
    for i, layer in enumerate(model.model.layers):
        if other_gpu_processes():
            wait_for_exclusive_gpu()
        for group, attrs in GROUPS:
            modules = [layer.get_submodule(attr) for attr in attrs]
            for module, weight in zip(modules, restore(
                    cli.work_dir/'had'/f'{i}_{group}.pt', decoder, args, 'cuda:0')):
                module.weight.copy_(weight)
    del decoder, layer, modules, module, weight
    utils.clean()
    tokenizer = AutoTokenizer.from_pretrained(cli.base_model, local_files_only=True, use_fast=False)
    torch.cuda.reset_peak_memory_stats()
    while True:
        if other_gpu_processes():
            wait_for_exclusive_gpu()
        start = time.perf_counter()
        measured_ppl = evaluate(model, tokenizer,
            cli.work_dir/'data/wikitext-2-raw-test.parquet', cli.context)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        if not other_gpu_processes():
            break
        print('RETIME_PPL_RETRY_GPU', flush=True)
    assert abs(measured_ppl['ppl']-record['ppl']['ppl']) < 1e-4
    record['ppl'] = measured_ppl
    record['ppl_evaluation_seconds'] = elapsed
    record['ppl_peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
    history['ppl_completed'] = True
    save(cli, result)
    del model
    utils.clean()
    timer = StreamTimer()
    packed_model, startup, errors = build_compressed(cli, 'had', timer, index)
    online_cli = SimpleNamespace(**{**vars(cli), 'online_modes': ['had']})
    while True:
        if other_gpu_processes():
            wait_for_exclusive_gpu()
        online(online_cli, result, index,
            prepared={'had': (packed_model, startup, errors, timer)},
            output_records=record['online'])
        if not other_gpu_processes():
            break
        print('RETIME_ONLINE_RETRY_GPU', flush=True)
    del packed_model
    utils.clean()
    history['status'] = 'complete'
    history['gpu_checks'] = 'No other GPU-0 compute processes before/after accepted PPL and online suite'
    save(cli, result)
    write_bit_comparison_reports(cli, result)


def write_bit_comparison_reports(cli, result):
    """Update existing reports only after all complete, validated measurements exist."""
    runs = result['bit_comparison']['runs']
    if not all(runs.get(str(b), {}).get('status') == 'complete' for b in CODEBOOKS):
        return
    if runs['2'].get('evaluation_remeasurement', {}).get('status') == 'running':
        return
    ordered = [runs[str(b)] for b in CODEBOOKS]
    for r in ordered:
        assert len(r['matrices']) == 128
        assert len({(m['layer'], m['group']) for m in r['matrices']}) == 128
        assert 0 < r['ldlq_seconds'] <= r['quantization_seconds']
        assert 0 < r['ppl']['ppl'] < float('inf')
        for key in ('blocks', 'scored_tokens', 'context_length', 'test_parquet_sha256'):
            assert r['ppl'][key] == ordered[0]['ppl'][key]
        online_record = r['online']['had']
        assert len(online_record['checkpoint_operator_validation']) == 8
        assert all(e['relative_error'] < .01 for e in online_record['checkpoint_operator_validation'])
        assert online_record['prefill_detail']['weight_decode']['calls'] == 128
        assert online_record['prefill_detail']['linear_matmul']['calls'] == 129
    if runs['2'].get('original_workflow_time_valid_for_comparison') is False:
        assert runs['2']['timing_remeasurement']['status'] == 'complete'

    def table(headers, rows):
        return '\n'.join(['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join(['---'] + ['---:'] * (len(headers)-1)) + ' |'] +
            ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])

    def replace_section(path, start_heading, end_heading, content):
        existing = path.read_text()
        start = existing.index(start_heading)
        end = existing.index(end_heading, start + len(start_heading))
        updated = existing[:start] + content.rstrip() + '\n\n' + existing[end:]
        temporary = path.with_suffix('.md.tmp')
        temporary.write_text(updated)
        temporary.replace(path)

    quality = table(['权重位宽', 'WikiText2 PPL ↓', '全模型总量化（s）',
        '其中初次 BlockLDLQ + 10 轮逐块局部搜索（s）', '还原 FP16 权重后的 PPL 测评（s）'], [
        [r['bits'], f"{r['ppl']['ppl']:.6f}", f"{r['quantization_seconds']:.2f}",
         f"{r['ldlq_seconds']:.2f}", f"{r['ppl_evaluation_seconds']:.2f}"] for r in ordered])
    online_rows, prefill_rows, decode_rows, overhead_rows, other_rows = [], [], [], [], []
    for r in ordered:
        o = r['online']['had']
        p = o['prefill_detail']
        d = o['profiled_decode']['detail']
        steps = result['bit_comparison']['config']['decode_steps']
        decode_ms = statistics.mean(x['ms_per_token_mean'] for x in o['decode_runs'])
        online_rows.append([r['bits'], f"{o['prefill_mean_seconds']*1000:.2f}",
            f'{decode_ms:.2f}', f"{p['weight_decode']['stream_ms']:.2f}",
            f"{p['linear_matmul']['stream_ms']:.2f}"])
        pkeys = ['activation_rotation', 'weight_decode', 'linear_matmul',
                 'attention_sdpa', 'kv_cache_update']
        dkeys = ['activation_rotation', 'fused_weight_decode_matvec', 'linear_matmul',
                 'attention_sdpa', 'kv_cache_update']
        pw = o['profiled_prefill_wall_seconds']*1000
        dw = o['profiled_decode']['wall_seconds']*1000/steps
        prefill_rows.append([r['bits']] + [f"{p[k]['stream_ms']:.2f}" for k in pkeys] +
            [f"{pw-sum(p[k]['stream_ms'] for k in pkeys):.2f}", f'{pw:.2f}'])
        decode_rows.append([r['bits']] + [f"{d[k]['stream_ms']/steps:.2f}" for k in dkeys] +
            [f"{dw-sum(d[k]['stream_ms'] for k in dkeys)/steps:.2f}", f'{dw:.2f}'])
        overhead_rows.append([r['bits'], f"{o['prefill_instrumentation_wall_ratio']:.3f}",
            f"{o['decode_instrumentation_wall_ratio']:.3f}",
            f"{max(x['relative_error'] for x in o['checkpoint_operator_validation']):.8f}"])
        other_rows.append([r['bits'], f"{r['initial_model_and_codebook_load_seconds']:.2f}",
            f"{r['reconstruction_for_ppl_seconds']:.2f}",
            f"{r['quantization_seconds']-r['ldlq_seconds']:.2f}",
            f"{100*r['ldlq_seconds']/r['quantization_seconds']:.2f}%"])
    online_table = table(['位宽', '无细 events prefill（ms/4096 tokens）',
        '无细 events decode（ms/token）', 'Profile prefill 权重解码（ms）',
        'Profile prefill 线性矩乘（ms）'], online_rows)
    retime = runs['2'].get('timing_remeasurement', {})
    timing_note = (
        f"2 bit 原始进程受到外部 GPU 任务及 SIGSTOP/SIGCONT 暂停影响；空闲后重测了 "
        f"{len(retime.get('matrices', []))} 个矩阵，逐项验证 Qidxs、SU、SV 与原检查点完全一致。"
        "表中 2 bit 总量化和 LDLQ 是 128 个有效逐矩阵计时的累计（受影响项用重测值替换），"
        "不是含等待的连续进程 wall time。原始计时保存在 JSON 的 timing_remeasurement，"
        "原 workflow_through_ppl_seconds 含等待，明确不用于速度比较。"
        if retime else "各位宽总量化取 128 次 quantize_linear 的计时累计。")
    if runs['2'].get('evaluation_remeasurement', {}).get('status') == 'complete':
        timing_note += ('2 bit PPL 测评用时和在线 suite 也已在计算进程检查通过后重测；'
                        '原始受干扰记录保存在 evaluation_remeasurement，不参与表格对比。')
    common = (
        "Llama-2-7B，Hadamard，no FT，初次 BlockLDLQ 量化 + 10 轮逐块局部搜索；WikiText2 test、context=4096、batch=1、83 个完整区块。"
        "共用作者 6144 条校准序列的已有 Hessian，未重测 Hessian 生成。2 bit 使用 E8P12、scale=0.9；"
        "3 bit 使用 E8P12RVQ3B、默认 scale=0.98/residual=2.04；4 bit 使用 E8P12RVQ4B、默认 scale=1.03/residual=3.45。"
        "这是各码本推荐缩放配置的对比，不是相同 scale 的消融。")
    terminology = (
        '术语与计时口径：初次 BlockLDLQ 使用分解矩阵 L 反馈已量化块的误差；后续“逐块局部搜索”基于当前残差和 Hessian H 重新选择码字，不是重复执行完整 BlockLDLQ，也不做反向传播。\n\n'
        '10 轮来自当前仓库默认参数 `quip_tune_iters=10`，不是 QuIP# 论文明确给出的轮数。相关思想见前作 [QuIP](https://arxiv.org/abs/2307.13304) §4.2 的 Greedy local search、附录 B.2 的 Greedy Updates；当前代码使用逐块更新。QuIP# 算法 1、§4.1 未单独说明这些后续迭代，不能仅凭正文确认原表使用相同轮数。\n\n'
        '本文及 JSON 中沿用的 `ldlq_seconds` 等 LDLQ 计时简称均包含初次量化和后续局部搜索；历史字段 `refinement_iterations` 对应局部搜索轮数。'
    )
    semantics = (
        "“总量化”含全部 128 个矩阵的 Hessian 读取/预处理、旋转、LDLQ、逆旋转、打包、临时检查点保存、CPU 误差诊断与清理；"
        "不含原模型加载、PPL 所需的 FP16 权重还原或 PPL 测评。三者单独记录，不能合并解释为推理速度。"
        "临时检查点均写入 /dev/shm，测完即清理，不额外保留 3/4 bit 全模型文件。")
    online_note = (
        "PPL 均从打包索引还原 FP16 权重后走同一 HF SDPA 前向，无 KV cache；在线耗时另用完整 32 层压缩权重模型、FP16 激活、SDPA、eager。"
        "Prefill=4096 tokens；decode 使用 4064-token KV 前缀，teacher-forced 前向 32 步至 4096。预热后无细 events 测 3 次均值，另做一次 events profile。"
        "不含加载、tokenization、采样、文件 IO；无 CUDA Graph/compile。")
    detail_note = (
        "分项为 CUDA stream events 间隔，含 CPU launch/调度空隙，不是纯 kernel busy time；不能与另一次无 events 延迟相加或计算精确占比。"
        "3/4 bit 的 prefill 权重解码项包含两级残差解码及缩放合并。逐 token 路径采用项目原生的解码与矩阵向量计算组合，"
        "未强拆为纯 E8 解码与纯 GEMV；3 bit 残差沿用填充到 8 行的 lookupmatmul_e81b_k8，4 bit 调用两次 E8P decode-matvec。"
        "线性矩乘的 prefill 项含 FP16 lm-head；decode 的独立线性项仅为 lm-head，其余在线线性计算在解码/GEMV 路径内。")
    results_text = '\n\n'.join([
        '## Hadamard 2 / 3 / 4 bit 对比', '完成日期：' + time.strftime('%Y-%m-%d') + '。' + common,
        terminology, quality, semantics, timing_note, online_table, online_note, detail_note,
        '原始数据：`quantization-profile.json` 的 `bit_comparison`；日志：`quantization-profile.log`。'
        '完整分项及插桩开销见 [TIMING_ANALYSIS.md](TIMING_ANALYSIS.md)。以下保留先前 RHT/RFFT 实验。'])
    replace_section(cli.work_dir/'RESULTS.md', '## Hadamard 2 / 3 / 4 bit',
                    '## 已完成：2 bit RHT / RFFT', results_text)
    timing_text = '\n\n'.join([
        '## Hadamard 2 / 3 / 4 bit 完整对比', common, terminology, quality, semantics, timing_note,
        table(['位宽', '原模型/码本加载（s）', 'PPL 权重还原累计（s）', '非 LDLQ 量化余项（s）', 'LDLQ / 总量化'], other_rows),
        '加载与权重还原为另列准备时间，受文件缓存和运行状态影响，不作为在线延迟。总量化减去 LDLQ 的余项包括旋转、传输、诊断、保存等，不能称为纯旋转耗时。',
        online_table, online_note,
        '### Prefill 算子分项（ms / 4096 tokens）',
        table(['位宽', '旋转', '权重解码/合并', '线性矩乘', 'SDPA', 'KV 更新', '其他余项', '同次 profile wall'], prefill_rows),
        '### Decode 算子分项（ms/token）',
        table(['位宽', '旋转', '解码/GEMV 路径', '独立线性（lm-head）', 'SDPA', 'KV 更新', '其他余项', '同次 profile wall/token'], decode_rows),
        detail_note,
        table(['位宽', 'Prefill 有/无 events wall 比', 'Decode 有/无 events wall 比', '8 项算子对照最大相对误差'], overhead_rows),
        '其他余项来自同次 profile 总 wall 减去已测 stream 间隔，含框架、同步和插桩开销，不是独立 GPU 算子。'
        '所有位宽均完成 128 个矩阵、完整 WikiText2 PPL 及第 16 层四类矩阵各 1/4 行输入的 8 项还原对照。',
        '### 数据与复跑',
        '复用 `quantization-profile.json` 的 `bit_comparison` 与 `quantization-profile.log`。在 `quip-sharp` 目录运行：',
        '```bash\nPYTHONPATH=. ../.venv/bin/python -u quantize_llama/profile_incoherence.py --stage bit-comparison\n```',
        '已完成位宽会跳过；可用 `--comparison-bits 3 4` 指定未完成位宽。受中断的位宽会重新量化，临时检查点不承诺跨进程续跑。'
        '2 bit 计时补测入口为 `--stage retime-2bit --retime-from-layer 25`，要求该位宽已完成且 `had` 原始检查点仍在。'
        '后续量化/PPL/在线前后检查 GPU-0 的计算进程，检测到争用则等待并重试；不能排除两个检查之间的极短外部任务。'])
    replace_section(cli.work_dir/'TIMING_ANALYSIS.md', '## Hadamard 2 / 3 / 4 bit',
                    '## 已完成：2 bit RHT / RFFT', timing_text)
    functional = Path(__file__).resolve().parents[1]/'FUNCTIONAL_MODULES.md'
    if functional.exists():
        text = functional.read_text()
        text = text.replace('本机仅做小矩阵验证', '本机已完成 Llama-2-7B Hadamard、no FT 的全模型量化、WikiText2 PPL 及在线 profiling')
        text = text.replace('当前本机完整验证仅覆盖 Llama-2-7B、2 bit、no FT',
            '当前本机完整验证覆盖 Llama-2-7B Hadamard 2/3/4 bit 及 Fourier 2 bit、no FT')
        text = text.replace('本次配置为 E8P12 2 bit；',
            '代表性 RHT/RFFT 配置为 E8P12 2 bit；另有 `--stage bit-comparison` 完成 Hadamard 2/3/4 bit 全模型量化、PPL 和在线对比；')
        text = text.replace('尚未完成 3/4 bit 全模型、梯度微调、Mistral、C4 和零样本任务的本机端到端验证。',
            'Hadamard 3/4 bit 全模型量化、WikiText2 PPL 和在线 profiling 也已完成。尚未完成梯度微调、Mistral、C4 和零样本任务的本机端到端验证。')
        functional.write_text(text)
    print('UPDATED existing RESULTS.md, TIMING_ANALYSIS.md and FUNCTIONAL_MODULES.md', flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', type=Path, default=Path('/mnt/model/llama-2-7b-hf'))
    parser.add_argument('--work-dir', type=Path, default=Path('/home/jylu/QuIP-Sharp/reproduction'))
    parser.add_argument('--hessian-dir', type=Path, default=None)
    parser.add_argument('--experiment-key', default=None,
                        help='Isolate another model under model_comparisons in the existing JSON')
    parser.add_argument('--layer', type=int, default=16)
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--decode-steps', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--stage', choices=['all', 'offline', 'online', 'bit-comparison', 'activation-baseline', 'activation-comparison', 'activation-bit-sweep', 'retime-2bit', 'retime-2bit-evaluation'], default='all')
    add_activation_args(parser)
    parser.add_argument('--weight-bits', type=int, choices=[2, 3, 4], default=2)
    parser.add_argument('--checkpoint-dir', type=Path, default=None)
    parser.add_argument('--ppl-blocks', type=int, default=0,
                        help='Activation stage only: 0=full test, positive=limited smoke test')
    parser.add_argument('--retime-from-layer', type=int, default=0)
    parser.add_argument('--comparison-bits', nargs='+', type=int, choices=[2, 3, 4], default=[2, 3, 4])
    parser.add_argument('--comparison-activation-bits', nargs='+', type=int,
                        choices=[0, 2, 3, 4], default=[0, 4],
                        help='Activation bit sweep targets; 0 means A16 (default: 0 4)')
    parser.add_argument('--online-modes', nargs='+', choices=['baseline', 'had', 'fft'],
                        default=['baseline', 'had', 'fft'])
    cli = parser.parse_args()
    if cli.ppl_blocks < 0:
        parser.error('--ppl-blocks must be nonnegative')
    if activation_overrides(cli) and cli.stage not in ('activation-comparison', 'activation-bit-sweep'):
        parser.error('Activation overrides require an activation comparison stage')
    cli.output = cli.work_dir / 'quantization-profile.json'
    cli.hessian_dir = cli.hessian_dir or cli.work_dir / 'hessians'
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert 0 < cli.decode_steps < cli.context <= 4096
    assert cli.repeats > 0
    result = json.loads(cli.output.read_text()) if cli.output.exists() else {}
    if cli.experiment_key:
        if cli.stage not in ('activation-baseline', 'activation-bit-sweep'):
            parser.error('--experiment-key requires activation-baseline or activation-bit-sweep')
        cli.root_result = result
        result = result.setdefault('model_comparisons', {}).setdefault(cli.experiment_key, {})
    if cli.stage == 'activation-baseline':
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        activation_baseline(cli, result)
        return
    if cli.stage == 'activation-comparison':
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        index = json.loads((cli.base_model/'model.safetensors.index.json').read_text())['weight_map']
        activation_comparison(cli, result, index)
        return
    if cli.stage == 'activation-bit-sweep':
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        index = json.loads((cli.base_model/'model.safetensors.index.json').read_text())['weight_map']
        activation_bit_sweep(cli, result, index)
        return
    if 'had' in result and 'fft' in result:
        result['prior_coarse_profile'] = {key: result.pop(key)
            for key in ('had', 'fft', 'decoder_only', 'scope') if key in result}
    result['benchmark_config'] = dict(gpu=torch.cuda.get_device_name(0),
        torch_version=torch.__version__, context=cli.context, batch_size=1,
        decode_steps=cli.decode_steps, decode_prefix_length=cli.context-cli.decode_steps,
        repeats=cli.repeats, offline_layer=cli.layer,
        offline_scope='Four actual layer-16 matrix shapes, both RHT/RFFT; no full-model requantization',
        online_scope='Complete 32-layer model with packed weights; RHT uses native operation order and E8 kernels; RFFT uses the same kernels plus a new unfused FFT adapter',
        timer_semantics='CUDA stream event intervals, including idle/launch gaps. CPU enqueue times are a separate axis and must not be added. All fine runs report their wall-time overhead against runs without events.',
        decode_semantics='Teacher-forced next-token forward with real KV cache, context grows to 4096; no tokenizer, sampling, disk IO or startup included online.')
    index = json.loads((cli.base_model/'model.safetensors.index.json').read_text())['weight_map']
    if cli.stage == 'retime-2bit-evaluation':
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        retime_2bit_evaluation(cli, result, index)
        print('COMPLETE 2-bit evaluation remeasurement', cli.output, flush=True)
        return
    if cli.stage == 'retime-2bit':
        retime_2bit(cli, result, index)
        print('COMPLETE 2-bit timing remeasurement', cli.output, flush=True)
        return
    if cli.stage == 'bit-comparison':
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        bit_comparison(cli, result, index)
        write_bit_comparison_reports(cli, result)
        print('COMPLETE bit comparison', cli.output, flush=True)
        return
    if cli.stage in ('all', 'offline'):
        offline(cli, result, index)
    if cli.stage in ('all', 'online'):
        online(cli, result, index)
    save(cli, result)
    print('COMPLETE', cli.output, flush=True)


if __name__ == '__main__':
    main()
