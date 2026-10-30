"""No-FT Llama RHT/RFFT ablation with packed E8P checkpoints and WT2 PPL.

Evaluation reconstructs FP16 weights from the packed checkpoints in memory.
This deliberately uses the same evaluation path for both transforms; it is
an accuracy experiment and does not benchmark compressed inference kernels.
"""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib import codebook, utils
from lib.algo import quip


REVISION = 'cafc59c036c6416ec2a9d5790752bec51297c197'
HESSIAN_REPO = 'relaxml/Hessians-Llama-2-7b-6144'


def fetch_hessian(root, name):
    path = root / name
    if path.exists():
        torch.load(path, map_location='cpu', weights_only=True)
        return path
    url = f'https://hf-mirror.com/{HESSIAN_REPO}/resolve/{REVISION}/{name}'
    for attempt in range(20):
        try:
            with requests.get(url, stream=True, timeout=(15, 90)) as r:
                if r.status_code == 429:
                    delay = min(max(int(r.headers.get('Retry-After', 60)), 5), 300)
                    logging.info('Hessian %s rate limited; retry in %ss', name, delay)
                    time.sleep(delay)
                    continue
                r.raise_for_status()
                tmp = path.with_suffix(f'.{os.getpid()}.partial')
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(2**20):
                        f.write(chunk)
                torch.load(tmp, map_location='cpu', weights_only=True)
                tmp.replace(path)
                logging.info('Downloaded %s (%s bytes)', name, path.stat().st_size)
                return path
        except (requests.RequestException, OSError, RuntimeError) as e:
            logging.warning('Download %s attempt %s: %s', name, attempt + 1, e)
            time.sleep(min(10 * (attempt + 1), 60))
    raise RuntimeError(f'Failed to download {name}')


def restore(path, cb, args, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    m = sum(s[0] for s in saved['shapes'])
    n = saved['shapes'][0][1]
    wr = cb.by_idxs(saved['Qidxs'].to(device)).view(m, n).float()
    w = quip.incoherence_process(wr, saved['SU'].float(),
                                 saved['SV'].float(), saved['scaleWH'], args)
    result = []
    offset = 0
    for shape, scale in zip(saved['shapes'], saved['scales']):
        result.append((w[offset:offset + shape[0]] * float(scale)).half().cpu())
        offset += shape[0]
    return result


@torch.inference_mode()
def evaluate(model, tokenizer, data_path, seqlen, max_blocks=None,
             evaluation='reconstructed FP16 weights -> HF SDPA'):
    data = pd.read_parquet(data_path)
    tokens = tokenizer('\n\n'.join(data['text'].tolist()),
                       return_tensors='pt').input_ids
    blocks = tokens.numel() // seqlen
    available_blocks = blocks
    if max_blocks is not None:
        if max_blocks <= 0:
            raise ValueError('max_blocks must be positive')
        blocks = min(blocks, max_blocks)
    if not blocks:
        raise ValueError('Test set is shorter than one context')
    model.cuda().eval()
    total = 0.0
    for i in range(blocks):
        x = tokens[:, i * seqlen:(i + 1) * seqlen].cuda()
        logits = model(x, use_cache=False).logits
        # Match the repository evaluator's cross-entropy convention.
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
            x[:, 1:].reshape(-1))
        total += loss.item()
        del logits
        if (i + 1) % 5 == 0 or i + 1 == blocks:
            logging.info('WT2 %s/%s blocks, running PPL %.6f',
                         i + 1, blocks, math.exp(total / (i + 1)))
    record = dict(ppl=math.exp(total / blocks), blocks=blocks,
                test_tokens=tokens.numel(), scored_tokens=blocks * (seqlen - 1),
                dropped_tail_tokens=tokens.numel() % seqlen,
                test_rows=len(data), context_length=seqlen,
                test_parquet_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                evaluation=evaluation)
    if max_blocks is not None:
        record.update(available_blocks=available_blocks,
                      unused_complete_blocks=available_blocks - blocks,
                      evaluation_scope='limited-block smoke test, not full WikiText2 PPL')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', default='/mnt/model/llama-2-7b-hf')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['baseline', 'had', 'fft'], required=True)
    parser.add_argument('--seqlen', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=8)
    cli = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    torch.set_num_threads(cli.threads)
    torch.manual_seed(cli.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = cli.work_dir / cli.mode
    out.mkdir(parents=True, exist_ok=True)
    # Prevent duplicate jobs from writing the same checkpoints on a resumed run.
    run_lock = (out / '.run.lock').open('w')
    fcntl.flock(run_lock, fcntl.LOCK_EX)
    metrics_path = out / 'metrics.json'
    if metrics_path.exists():
        prior = json.loads((out / 'config.json').read_text())
        if (prior['base_model'] != cli.base_model or prior['seed'] != cli.seed
                or json.loads(metrics_path.read_text())['context_length'] != cli.seqlen):
            raise ValueError('Existing results use different settings; use another work-dir')
        logging.info('Already complete: %s', metrics_path.read_text())
        return
    hroot = cli.work_dir / 'hessians'
    hroot.mkdir(exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        cli.base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation='sdpa').eval()
    tokenizer = AutoTokenizer.from_pretrained(cli.base_model, use_fast=False,
                                              local_files_only=True)
    args = SimpleNamespace(
        incoh_mode=cli.mode, use_fp64=False, rescale_WH=False,
        lora_rank=0, full_svd=False, sigma_reg=1e-2, sigma_reg2=1e-2,
        scale_override=0.9, resid_scale_override=-1,
        no_use_buffered=False, lowmem_ldlq=False, quip_tune_iters=10,
        save_pfx=str(out), seed=cli.seed, ft_epochs=0)
    config = dict(vars(args), base_model=cli.base_model, codebook='E8P12',
                  hessian_repo=HESSIAN_REPO, hessian_revision=REVISION,
                  hessian_sequences=6144, hessian_context=4096,
                  quantization_seed='seed + layer_index',
                  torch_version=torch.__version__)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    started = time.monotonic()
    if cli.mode != 'baseline':
        cb = codebook.get_codebook('E8P12')
        decoder = codebook.get_codebook('E8P12').cuda()
        groups = [('qkv', ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']),
                  ('o', ['self_attn.o_proj']),
                  ('up', ['mlp.up_proj', 'mlp.gate_proj']),
                  ('down', ['mlp.down_proj'])]
        # Prefetch a few files while quantization uses the GPU; keep downloads bounded.
        names = [f'{i}_{name}.pt' for i in range(len(model.model.layers))
                 for name, _ in groups]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending = {}
            for step, filename in enumerate(names):
                for lookahead in range(step, min(step + 4, len(names))):
                    name = names[lookahead]
                    if name not in pending:
                        pending[name] = pool.submit(fetch_hessian, hroot, name)
                hessian_path = pending.pop(filename).result()
                i, group_idx = divmod(step, len(groups))
                if group_idx == 0:
                    torch.manual_seed(cli.seed + i)
                name, attrs = groups[group_idx]
                layer = model.model.layers[i]
                modules = [layer.get_submodule(attr) for attr in attrs]
                save_path = out / filename
                logging.info('Quantizing %s %s (%s/%s)', cli.mode, filename,
                             step + 1, len(names))
                t = time.monotonic()
                existed = save_path.exists()
                quip.quantize_linear([m.weight for m in modules], str(save_path),
                                     str(hessian_path), cb, args, 'cuda:0')
                if existed:
                    # Recreate phase/sign RNG draws when resuming a partial run.
                    n = modules[0].weight.shape[1]
                    m = sum(module.weight.shape[0] for module in modules)
                    if cli.mode == 'had':
                        torch.randn(n, device='cuda:0')
                        torch.randn(m, device='cuda:0')
                    else:
                        torch.rand(n // 2, device='cuda:0')
                        torch.rand(m // 2, device='cuda:0')
                for module, weight in zip(modules, restore(save_path, decoder, args, 'cuda:0')):
                    module.weight.copy_(weight)
                utils.clean()
                logging.info('Completed %s in %.1fs; peak GPU %.2f GiB',
                             filename, time.monotonic() - t,
                             torch.cuda.max_memory_allocated() / 2**30)
        del decoder, cb
        utils.clean()
    logging.info('Starting full WikiText2 test evaluation, context %s', cli.seqlen)
    metrics = evaluate(model, tokenizer,
                       cli.work_dir / 'data/wikitext-2-raw-test.parquet', cli.seqlen)
    metrics.update(mode=cli.mode, wall_seconds=time.monotonic() - started,
                   peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                   paper_target=dict(had=8.22, fft=8.30).get(cli.mode))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    logging.info('RESULT %s', json.dumps(metrics))


if __name__ == '__main__':
    main()
