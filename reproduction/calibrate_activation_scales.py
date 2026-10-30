"""Resumeable QuIP# W4A4 activation-only rho/residual calibration.

The supervisor owns no CUDA context. A contended worker exits and releases GPU
memory; the supervisor restarts it only after the GPU becomes idle again.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproduction/activation-scale-calibration'
sys.path.insert(0, str(ROOT / 'quip-sharp'))
MODELS = {'13b': ROOT / 'models/llama-2-13b-hf',
          '7b': Path('/mnt/model/llama-2-7b-hf')}
DEFAULT = (1.03, 3.45)
COARSE = list(itertools.product([.85, .95, 1.03, 1.10, 1.20],
                                [2.75, 3.10, 3.45, 3.80, 4.15]))
SETTINGS = dict(context=4096, activation_rms_group=128, codeword_dimension=8,
    activation_bits=4, weight_bits=4, rotation='full-dimensional Hadamard',
    weight_rho=1.03, weight_residual=3.45, seed=0, local_search_passes=10,
    calibration_documents=4, validation_documents=8, confirmation_documents=8,
    sampled_rows_per_document_per_linear=32, search_chunk=65536,
    coarse_grid=COARSE, refinement_steps=[.04, .175], shortlist=3,
    ranking='mean per-linear relative output squared error, includes fused scales',
    selection='lowest validation NLL among shortlist plus default; independent confirmation must improve NLL or retain default')


def now():
    return datetime.now(timezone.utc).isoformat()


def log(*args):
    print(now(), *args, flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def atomic_torch(path, value):
    import torch
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def gpu_pids():
    output = subprocess.check_output(['nvidia-smi', '--id=0',
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    return [int(line.strip()) for line in output.splitlines()
            if line.strip().isdigit() and int(line.strip()) != os.getpid()]


class Contended(Exception):
    pass


def guard():
    pids = gpu_pids()
    if pids:
        raise Contended(f'other GPU compute PIDs: {pids}')


def pair_key(pair):
    return f'{pair[0]:.5f}_{pair[1]:.5f}'


def source_fingerprint():
    paths = [Path(__file__), ROOT / 'quip-sharp/lib/algo/quip.py',
        ROOT / 'quip-sharp/lib/linear/activation_quantization.py',
        ROOT / 'quip-sharp/lib/linear/quantized_linear.py',
        ROOT / 'quip-sharp/lib/codebook/latticee8_padded12_rvq4bit.py',
        ROOT / 'quip-sharp/quantize_llama/profile_incoherence.py',
        ROOT / 'quip-sharp/model/llama.py',
        ROOT / 'quip-sharp/lib/utils/matmul_had.py']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def prepare_data(tokenizer):
    """One crop per unique document; the three document sets are disjoint."""
    import pyarrow.parquet as pq
    import torch
    path = OUT / 'fineweb-heldout.pt'
    if path.exists():
        value = torch.load(path, weights_only=True)
        assert value['tokens'].shape == (20, 4096)
        assert len(set(value['document_sha256'])) == 20
        return value
    source = Path('/home/jylu/WUSH/datasets/fineweb-edu/sample/10BT/013_00000.parquet')
    rng = random.Random(20261028)
    batches = pq.ParquetFile(source).iter_batches(batch_size=4096, columns=['text'])
    records, hashes, seen = [], [], set()
    for batch in batches:
        texts = batch.column(0).to_pylist()
        order = list(range(len(texts)))
        rng.shuffle(order)
        for row in order:
            text = texts[row]
            if len(text) < 12000:
                continue
            doc_hash = hashlib.sha256(text.encode()).hexdigest()
            if doc_hash in seen:
                continue
            seen.add(doc_hash)
            ids = tokenizer(text, return_tensors='pt').input_ids[0]
            if len(ids) < 4096:
                continue
            start = rng.randint(0, len(ids) - 4096)
            records.append(ids[start:start + 4096])
            hashes.append(doc_hash)
            if len(records) == 20:
                break
        if len(records) == 20:
            break
    if len(records) != 20:
        raise RuntimeError('Not enough unique long FineWeb documents')
    value = dict(tokens=torch.stack(records), document_sha256=hashes)
    atomic_torch(path, value)
    atomic_json(OUT / 'data.json', dict(source=str(source), source_bytes=source.stat().st_size,
        source_mtime_ns=source.stat().st_mtime_ns, seed=20261028,
        tokenizer_sha256=digest(MODELS['7b'] / 'tokenizer.model'),
        sampling='shuffled 4096-row batches, unique long documents, one random 4096-token crop/document',
        splits=dict(calibration=list(range(4)), validation=list(range(4, 12)),
                    confirmation=list(range(12, 20))),
        document_sha256=hashes, token_ids_sha256=hashlib.sha256(value['tokens'].numpy().tobytes()).hexdigest()))
    return value


def configure(model, pair=None):
    from lib.linear.activation_quantization import configure_activation_quantization
    overrides = dict(bits=0 if pair is None else 4, group_size=128, chunk_size=65536,
        scale_override=None if pair is None else pair[0],
        resid_scale_override=None if pair is None else pair[1])
    return configure_activation_quantization(model, overrides)


def checkpoint_build(name, record, save):
    import torch
    from transformers import AutoModelForCausalLM
    from lib import codebook, utils
    from lib.algo import quip
    from quantize_llama.profile_incoherence import GROUPS
    directory = OUT / name / 'weights'
    directory.mkdir(exist_ok=True)
    record['stage'] = 'rebuilding_w4_weights'
    save()
    guard()
    model = AutoModelForCausalLM.from_pretrained(MODELS[name], local_files_only=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation='sdpa').eval()
    cb = codebook.get_codebook('E8P12RVQ4B')
    args = SimpleNamespace(incoh_mode='had', use_fp64=False, rescale_WH=False,
        lora_rank=0, full_svd=False, sigma_reg=.01, sigma_reg2=.01,
        scale_override=-1, resid_scale_override=-1, no_use_buffered=False,
        lowmem_ldlq=False, quip_tune_iters=10, save_pfx=str(directory))
    hroot = ROOT / ('reproduction/hessians-13b' if name == '13b' else 'reproduction/hessians')
    for i, layer in enumerate(model.model.layers):
        torch.manual_seed(i)
        for group, attrs in GROUPS:
            guard()
            filename = f'{i}_{group}.pt'
            path = directory / filename
            state_path = directory / f'{i}_{group}.rng.pt'
            previous = record.setdefault('matrices', {}).get(filename)
            if previous and path.exists() and state_path.exists():
                if digest(path) != previous['sha256']:
                    raise RuntimeError(f'Checkpoint changed: {path}')
                states = torch.load(state_path, weights_only=True)
                torch.set_rng_state(states['cpu'])
                torch.cuda.set_rng_state(states['cuda'])
                continue
            partial = directory / f'{filename}.partial'
            partial.unlink(missing_ok=True)
            start = time.monotonic()
            quip.quantize_linear([layer.get_submodule(a).weight for a in attrs],
                str(partial), str(hroot / filename), cb, args, 'cuda:0')
            guard()
            partial.replace(path)
            atomic_torch(state_path, dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state()))
            record['matrices'][filename] = dict(seconds=time.monotonic() - start,
                sha256=digest(path), hessian_sha256=digest(hroot / filename))
            record['matrices_completed'] = len(record['matrices'])
            record['matrices_total'] = len(model.model.layers) * 4
            save()
            log(name, 'W4_MATRIX', record['matrices_completed'], '/', record['matrices_total'], filename)
    del model, cb
    utils.clean()
    return directory


def token_eval(model, tokens, metric, save):
    import torch
    metric.setdefault('block_losses', [])
    metric.setdefault('seconds', 0.)
    for i in range(len(metric['block_losses']), len(tokens)):
        guard()
        start = time.monotonic()
        x = tokens[i:i + 1].cuda()
        logits = model(x, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1)).item()
        del logits, x
        torch.cuda.synchronize()
        guard()
        if not math.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        metric['block_losses'].append(loss)
        metric['seconds'] += time.monotonic() - start
        save()
        log('PPL_BLOCK', i + 1, '/', len(tokens), math.exp(sum(metric['block_losses']) / (i + 1)))
    metric.update(nll=sum(metric['block_losses']) / len(tokens), blocks=len(tokens),
        scored_tokens=len(tokens) * (tokens.shape[1] - 1), context=tokens.shape[1], status='complete')
    metric['ppl'] = math.exp(metric['nll'])
    save()
    return metric


def collect_inputs(model, tokens, name, record, save):
    import torch
    from lib.linear.quantized_linear import QuantizedLinear
    path = OUT / name / 'linear-inputs.pt'
    if path.exists():
        return torch.load(path, weights_only=True)
    configure(model)
    inputs, handles = {}, []
    def capture(key):
        def hook(module, args):
            x = args[0].reshape(-1, module.in_features)
            positions = torch.linspace(0, len(x) - 1, 32, device=x.device).long()
            inputs.setdefault(key, []).append(x[positions].cpu())
        return hook
    for key, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            handles.append(module.register_forward_pre_hook(capture(key)))
    record['stage'] = 'collecting_calibration_inputs'
    save()
    try:
        for ids in tokens:
            guard()
            model(ids[None].cuda(), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    result = {key: torch.cat(rows) for key, rows in inputs.items()}
    assert all(len(x) == 128 for x in result.values())
    atomic_torch(path, result)
    return result


def distortion(model, inputs, pairs, record, save):
    import torch
    from lib.linear.activation_quantization import E8ActivationQuantizer
    scores = record.setdefault('distortion', {})
    for key, source in inputs.items():
        guard()
        pending = [p for p in pairs if key not in scores.get(pair_key(p), {}).get('linears', {})]
        if not pending:
            continue
        module = model.get_submodule(key)
        module.activation_quantizer = E8ActivationQuantizer(dict(bits=0))
        x = source.cuda()
        def flat(y):
            return torch.cat(y, dim=-1).float() if isinstance(y, tuple) else y.float()
        reference = flat(module(x))
        denominator = reference.square().sum().item()
        for pair in pending:
            guard()
            module.activation_quantizer = E8ActivationQuantizer(dict(bits=4, group_size=128,
                chunk_size=65536, scale_override=pair[0], resid_scale_override=pair[1]))
            actual = flat(module(x))
            relative = (actual - reference).square().sum().item() / max(denominator, 1e-30)
            if not math.isfinite(relative):
                raise RuntimeError('Nonfinite distortion')
            item = scores.setdefault(pair_key(pair), dict(pair=list(pair), linears={}))
            item['linears'][key] = relative
            del actual
        module.activation_quantizer = E8ActivationQuantizer(dict(bits=0))
        del reference, x
        save()
        log('DISTORTION_LINEAR', key, 'candidates', len(pairs))
    for item in scores.values():
        if len(item['linears']) == len(inputs):
            item['mean_relative_error'] = sum(item['linears'].values()) / len(inputs)
    save()


def report():
    sections = ['# QuIP# W4A4 激活尺度校准', '',
        '全维 Hadamard、激活 RMS 分组 128、8 维 E8P12RVQ4B；权重参数固定，仅调整激活 ρ 与 RVQ 残差系数。', '',
        'FineWeb-Edu：4 个校准文档筛选输出误差，8 个独立文档按 NLL 选参，另 8 个独立文档确认。所有文档上下文为 4096。', '',
        '先测 25 组粗网格，再对输出误差最小点做 3×3 局部细化；按输出误差取前三组及默认值进行验证。确认集上未优于默认值则保留默认值。这里只能确定所测候选中的最佳参数。', '',
        '| 模型 | 状态 | 激活 ρ | 激活残差系数 | WikiText2 PPL | 默认 PPL |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for name in MODELS:
        path = OUT / name / 'results.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        p = r.get('locked_pair', [None, None])
        sections.append(f'| {name} | {r.get("stage", "pending")} | {p[0] or "—"} | {p[1] or "—"} | '
            f'{r.get("test_selected", {}).get("ppl", "—")} | {r.get("test_default", {}).get("ppl", "—")} |')
    (OUT / 'RESULTS.md').write_text('\n'.join(sections) + '\n')


def publish(name, record):
    main = ROOT / 'reproduction/RESULTS.md'
    original = main.read_text()
    begin, end = '<!-- activation-scale-calibration:start -->', '<!-- activation-scale-calibration:end -->'
    text = [begin, '### QuIP# W4A4：激活尺度参数校准', '',
        '保留全维 Hadamard、激活 RMS 分组 128 和原权重参数；只调整激活 ρ 与 RVQ 残差系数。'
        '使用独立 FineWeb-Edu 校准、验证及确认文档选参后锁定，再测完整 WikiText2（context=4096、83 区块）。'
        '详见[校准流程和结果](activation-scale-calibration/RESULTS.md)。', '',
        '| 模型 | 激活 ρ | 激活残差系数 | 默认参数 PPL | 校准后 PPL | ΔPPL |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label in MODELS:
        p = OUT / label / 'results.json'
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        if r.get('status') != 'complete':
            continue
        rho, resid = r['locked_pair']
        default, selected = r['test_default']['ppl'], r['test_selected']['ppl']
        text.append(f'| Llama-2-{label.upper()} | {rho} | {resid} | {default:.6f} | {selected:.6f} | {selected-default:+.6f} |')
    text += ['', '各模型共享一组激活参数；这里的最优仅指所测候选。原始对照表保留默认参数结果。', end]
    block = '\n'.join(text)
    if begin in original:
        start = original.index(begin)
        stop = original.index(end, start) + len(end)
        updated = original[:start] + block + original[stop:]
    else:
        updated = original.rstrip() + '\n\n' + block + '\n'
    if main.read_text() != original:
        raise RuntimeError('RESULTS.md changed while publishing; retry')
    temp = main.with_suffix('.calibration.tmp')
    temp.write_text(updated)
    temp.replace(main)


def worker(name):
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    from lib import utils
    from quantize_llama.profile_incoherence import build_compressed, StreamTimer
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    # Match profile_incoherence.py: matmul TF32 is off, cuDNN keeps its
    # original PyTorch default. A prior run additionally disabled cuDNN TF32;
    # rerun A16 on the same saved weights to test whether this explains drift.
    torch.backends.cudnn.allow_tf32 = True
    directory = OUT / name
    directory.mkdir(exist_ok=True)
    path = directory / 'results.json'
    record = json.loads(path.read_text()) if path.exists() else dict(
        model=name, status='running', started_at=now(), settings=SETTINGS,
        source_sha256=source_fingerprint())
    if record['source_sha256'] != source_fingerprint():
        raise RuntimeError('Experiment sources changed since start')
    def save():
        record.update(updated_at=now(), worker_pid=os.getpid())
        atomic_json(path, record)
    if record.get('status') == 'complete':
        publish(name, record)
        return
    record['status'] = 'running'
    record.pop('error', None)
    record.pop('interruption', None)
    record['backend_settings'] = dict(matmul_allow_tf32=False,
                                     matmul_allow_fp16_reduced_precision_reduction=False,
                                     matmul_allow_bf16_reduced_precision_reduction=False,
                                     cudnn_allow_tf32=True)
    save()
    try:
        guard()
        tokenizer = AutoTokenizer.from_pretrained(MODELS[name], local_files_only=True, use_fast=False)
        assert digest(MODELS['7b'] / 'tokenizer.model') == digest(MODELS['13b'] / 'tokenizer.model')
        data = prepare_data(tokenizer)
        checkpoint_dir = checkpoint_build(name, record, save)
        guard()
        cli = SimpleNamespace(base_model=MODELS[name], work_dir=ROOT / 'reproduction', layer=16)
        index = json.loads((MODELS[name] / 'model.safetensors.index.json').read_text())['weight_map']
        record['stage'] = 'building_native_model'
        save()
        model, startup, errors = build_compressed(cli, 'had', StreamTimer(), index,
            'E8P12RVQ4B', checkpoint_dir, native=True)
        record.update(startup=startup, operator_validation=errors)
        configure(model)
        model(torch.ones(1, 8, device='cuda', dtype=torch.long), use_cache=False)
        test_path = ROOT / 'reproduction/data/wikitext-2-raw-test.parquet'
        ids = tokenizer('\n\n'.join(pd.read_parquet(test_path)['text'].tolist()), return_tensors='pt').input_ids
        test_tokens = ids[0, :83 * 4096].reshape(83, 4096)
        assert ids.numel() == 341469
        record['test_data'] = dict(parquet_sha256=digest(test_path),
            token_ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())
        record['stage'] = 'validating_rebuilt_weights_a16'
        save()
        control = token_eval(model, test_tokens, record.setdefault('a16_control', {}), save)
        previous = json.loads((ROOT / 'reproduction/quantization-profile.json').read_text())
        if name == '13b':
            previous = previous['model_comparisons']['llama2-13b']
        baseline = previous['weight_activation_ppl']['4']['activation']
        difference = abs(control['ppl'] - baseline['0']['ppl']['ppl'])
        record['a16_control_difference'] = difference
        if difference > 1e-6:
            record['a16_control_warning'] = dict(
                reference_ppl=baseline['0']['ppl']['ppl'],
                rebuilt_ppl=control['ppl'], tolerance=1e-6,
                reason='historical quantization-profile artifact differs from current deterministic rebuild')
            log('A16_BASELINE_WARNING', name, 'historical=', baseline['0']['ppl']['ppl'],
                'rebuilt=', control['ppl'], 'difference=', difference)
        inputs = collect_inputs(model, data['tokens'][:4], name, record, save)
        record['stage'] = 'coarse_grid_output_distortion'
        save()
        distortion(model, inputs, COARSE, record, save)
        best = min((record['distortion'][pair_key(p)] for p in COARSE),
                   key=lambda item: item['mean_relative_error'])['pair']
        refined = [tuple(round(v, 5) for v in p) for p in itertools.product(
            [best[0] - .04, best[0], best[0] + .04],
            [best[1] - .175, best[1], best[1] + .175])]
        record.update(stage='refined_grid_output_distortion', refined_grid=refined)
        save()
        distortion(model, inputs, refined, record, save)
        ranked = sorted(record['distortion'].values(), key=lambda item: item['mean_relative_error'])
        shortlist = [tuple(item['pair']) for item in ranked[:3]]
        if DEFAULT not in shortlist:
            shortlist.append(DEFAULT)
        record.update(stage='heldout_validation', shortlist=shortlist)
        save()
        for pair in shortlist:
            configure(model, pair)
            metric = record.setdefault('validation', {}).setdefault(pair_key(pair), {})
            token_eval(model, data['tokens'][4:12], metric, save)
        chosen = min(shortlist, key=lambda p: record['validation'][pair_key(p)]['nll'])
        record.update(stage='independent_confirmation', validation_winner=chosen)
        save()
        for pair in set([chosen, DEFAULT]):
            configure(model, pair)
            metric = record.setdefault('confirmation', {}).setdefault(pair_key(pair), {})
            token_eval(model, data['tokens'][12:20], metric, save)
        if record['confirmation'][pair_key(chosen)]['nll'] >= record['confirmation'][pair_key(DEFAULT)]['nll']:
            chosen = DEFAULT
        # Persist selection before reading test losses; no test-based retuning.
        record.update(locked_pair=chosen, parameters_locked_at=record.get('parameters_locked_at', now()),
                      stage='full_wikitext2_default')
        save()
        configure(model, DEFAULT)
        token_eval(model, test_tokens, record.setdefault('test_default', {}), save)
        delta = abs(record['test_default']['ppl'] - baseline['4']['ppl']['ppl'])
        record['a4_default_reproduction_difference'] = delta
        if delta > 1e-6:
            record['a4_default_reproduction_warning'] = dict(
                reference_ppl=baseline['4']['ppl']['ppl'],
                rebuilt_ppl=record['test_default']['ppl'], tolerance=1e-6,
                reason='historical quantization-profile artifact differs from current deterministic rebuild')
            log('A4_BASELINE_WARNING', name, 'historical=', baseline['4']['ppl']['ppl'],
                'rebuilt=', record['test_default']['ppl'], 'difference=', delta)
        record['stage'] = 'full_wikitext2_selected'
        save()
        configure(model, chosen)
        if chosen == DEFAULT:
            record['test_selected'] = dict(record['test_default'])
        else:
            token_eval(model, test_tokens, record.setdefault('test_selected', {}), save)
        record.update(status='complete', stage='complete', finished_at=now())
        save()
        report()
        publish(name, record)
        del model
        utils.clean()
        log('MODEL_COMPLETE', name, 'parameters', chosen, 'PPL', record['test_selected']['ppl'])
    except Contended as exc:
        record.update(status='waiting_for_gpu', interruption=str(exc))
        save()
        log('RELEASE_GPU', name, str(exc))
        raise SystemExit(75)
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
        save()
        raise


def supervise():
    lock = (OUT / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(supervisor_pid=os.getpid(), started_at=now(), models=list(MODELS), settings=SETTINGS)
    atomic_json(OUT / 'status.json', status)
    for name in MODELS:
        while True:
            idle = 0
            while idle < 2:
                pids = gpu_pids()
                memory, util = map(int, subprocess.check_output(['nvidia-smi', '--id=0',
                    '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                    text=True, timeout=15).strip().split(','))
                idle = idle + 1 if not pids and memory < 1024 and util < 5 else 0
                status.update(status='checking_gpu' if idle else 'waiting_for_gpu', current_model=name,
                    other_pids=pids, memory_mib=memory, utilization=util, updated_at=now())
                atomic_json(OUT / 'status.json', status)
                if idle < 2:
                    time.sleep(5)
            status.update(status='running', current_model=name, updated_at=now())
            atomic_json(OUT / 'status.json', status)
            log('START_MODEL', name)
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--model', name],
                cwd=ROOT, env={**os.environ, 'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false'})
            status['worker_pid'] = child.pid
            atomic_json(OUT / 'status.json', status)
            code = child.wait()
            report()
            if code == 0:
                break
            if code != 75:
                status.update(status='failed', exit_code=code, updated_at=now())
                atomic_json(OUT / 'status.json', status)
                raise SystemExit(code)
    status.update(status='complete', finished_at=now())
    atomic_json(OUT / 'status.json', status)
    report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(MODELS))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.model:
        worker(args.model)
    else:
        supervise()
