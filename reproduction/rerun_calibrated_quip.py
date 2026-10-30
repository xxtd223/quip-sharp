"""Queue independent paired W4A4 reruns after both activation calibrations finish.

The CPU-only scheduler waits for calibration and an idle GPU. Workers reuse
verified W4 checkpoints, lock the held-out-selected activation parameters, and
score defaults and selected parameters in a fresh model. Contention releases
CUDA memory and exits 75; completed test blocks survive restarts.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import calibrate_activation_scales as calibration

ROOT = calibration.ROOT
OUT = ROOT / 'reproduction/quip-calibrated-rerun'
MODELS = ('13b', '7b')


def read(path):
    return json.loads(path.read_text())


def sources():
    return {**calibration.source_fingerprint(),
            str(Path(__file__).resolve().relative_to(ROOT)):
            calibration.digest(Path(__file__))}


def ready():
    """Both models must finish the complete calibration, including test scoring."""
    for name in MODELS:
        path = calibration.OUT / name / 'results.json'
        if not path.exists() or read(path).get('status') != 'complete':
            return False
    return True


def idle():
    if calibration.gpu_pids():
        return False
    memory, utilization = map(int, subprocess.check_output([
        'nvidia-smi', '--id=0', '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(','))
    return memory < 1024 and utilization < 5


def publish():
    """Keep historical comparison rows; append independently rerun tuned rows."""
    main = ROOT / 'reproduction/RESULTS.md'
    original = main.read_text()
    heading = '### WUSH + GPTQ FP4 W4A4 对照'
    start = original.index(heading)
    stop = original.find('\n### ', start + len(heading))
    if stop == -1:
        stop = len(original)
    section = original[start:stop]
    marker_start = '<!-- calibrated-quip-rerun:start -->'
    marker_end = '<!-- calibrated-quip-rerun:end -->'
    rows = []
    details = []
    for name in ('7b', '13b'):
        path = OUT / name / 'results.json'
        if not path.exists():
            continue
        record = read(path)
        if record.get('status') != 'complete':
            continue
        rho, residual = record['locked_pair']
        default = record['default']['ppl']
        selected = record['selected']['ppl']
        historical = record['historical_ppl']
        rows.append((f'| Llama-2-{name.upper()} | QuIP#（激活参数校准后复测） | '
                     f'E8P12RVQ4B（8 维），RMS 尺度组=128，ρ={rho}，残差={residual} | '
                     f'全输入维 | {selected:.6f} | {selected-historical:+.6f} |'))
        details.append(f'| Llama-2-{name.upper()} | {historical:.6f} | {default:.6f} | '
                       f'{selected:.6f} | {selected-default:+.6f} |')
    if not rows:
        return
    lines = section.splitlines()
    lines = [line for line in lines if ' | QuIP#（激活参数校准后复测） | ' not in line]
    table_end = next(i for i, line in enumerate(lines)
                     if i > 0 and lines[i-1].startswith('| Llama-2-') and not line.startswith('|'))
    lines[table_end:table_end] = rows
    section = '\n'.join(lines) + '\n'
    note = '\n'.join([marker_start,
        '校准后 QuIP# 两组在独立新进程中复测，复用已核验的同一批 W4 权重；'
        '权重 ρ=1.03、残差=3.45 不变，仅使用 FineWeb-Edu 留出集锁定的激活参数。'
        '保留全输入维 Hadamard、激活 RMS=128、context=4096、batch=1、83 区块和 339885 计分 token。'
        '原表 QuIP# 默认参数结果保留；上表 ΔPPL 仍以原表 QuIP# 为基准。'
        '下表同批默认参数对照用于区分参数收益与历史基线漂移；负 ΔPPL 表示提升。'
        '详见[独立复测记录](quip-calibrated-rerun/RESULTS.md)。', '',
        '| 模型 | 原表默认 PPL | 本次默认 PPL | 校准参数复测 PPL | 相对本次默认 ΔPPL |',
        '| --- | ---: | ---: | ---: | ---: |', *details, marker_end])
    if marker_start in section:
        left = section.index(marker_start)
        right = section.index(marker_end, left) + len(marker_end)
        section = section[:left] + note + section[right:]
    else:
        section = section.rstrip() + '\n\n' + note + '\n\n'
    updated = original[:start] + section + original[stop:]
    if main.read_text() != original:
        raise RuntimeError('RESULTS.md changed during publication; retry')
    backup = OUT / 'RESULTS.before-rerun.md'
    if not backup.exists():
        backup.write_text(original)
    temp = main.with_suffix('.rerun.tmp')
    temp.write_text(updated)
    temp.replace(main)
    (OUT / 'RESULTS.md').write_text('\n'.join([
        '# QuIP# W4A4 校准参数独立复测', '', note, '',
        '原始记录：`13b/results.json`、`7b/results.json`。'
        '历史 A16/W4A4 重建漂移详见校准原始记录；复测不以 WikiText2 重新选参。\n']))


def worker(name):
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    from quantize_llama.profile_incoherence import build_compressed, StreamTimer

    directory = OUT / name
    directory.mkdir(exist_ok=True)
    path = directory / 'results.json'
    source = read(calibration.OUT / name / 'results.json')
    assert source['status'] == 'complete'
    assert source['settings'] == json.loads(json.dumps(calibration.SETTINGS))
    assert source['source_sha256'] == calibration.source_fingerprint()
    pair = tuple(source['locked_pair'])
    assert len(pair) == 2 and source['parameters_locked_at']
    record = read(path) if path.exists() else dict(model=name,
        started_at=calibration.now(), source_sha256=sources(), locked_pair=pair,
        calibration_results_sha256=calibration.digest(calibration.OUT / name / 'results.json'),
        parameters_locked_at=source['parameters_locked_at'], settings=source['settings'],
        backend_settings=source['backend_settings'], test_data=source['test_data'],
        weight_sha256={key: value['sha256'] for key, value in source['matrices'].items()})
    assert record['source_sha256'] == sources()
    assert tuple(record['locked_pair']) == pair
    assert record['calibration_results_sha256'] == calibration.digest(calibration.OUT / name / 'results.json')
    if record.get('status') == 'complete':
        return

    def save():
        record.update(updated_at=calibration.now(), worker_pid=os.getpid())
        calibration.atomic_json(path, record)

    record.update(status='running', stage='verifying_saved_weights')
    record.pop('error', None)
    save()
    try:
        calibration.guard()
        weights = calibration.OUT / name / 'weights'
        for key, expected in record['weight_sha256'].items():
            calibration.guard()
            if calibration.digest(weights / key) != expected:
                raise RuntimeError(f'Saved weight changed: {key}')
        torch.set_num_threads(8)
        torch.manual_seed(0)
        torch.set_grad_enabled(False)
        backend = record['backend_settings']
        torch.backends.cuda.matmul.allow_tf32 = backend['matmul_allow_tf32']
        torch.backends.cudnn.allow_tf32 = backend['cudnn_allow_tf32']
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = backend['matmul_allow_fp16_reduced_precision_reduction']
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = backend['matmul_allow_bf16_reduced_precision_reduction']
        cli = SimpleNamespace(base_model=calibration.MODELS[name],
                              work_dir=ROOT / 'reproduction', layer=16)
        index = read(cli.base_model / 'model.safetensors.index.json')['weight_map']
        record['stage'] = 'building_native_model'
        save()
        model, startup, errors = build_compressed(cli, 'had', StreamTimer(), index,
                                                  'E8P12RVQ4B', weights, native=True)
        record.update(startup=startup, operator_validation=errors)
        tokenizer = AutoTokenizer.from_pretrained(cli.base_model, local_files_only=True, use_fast=False)
        data = ROOT / 'reproduction/data/wikitext-2-raw-test.parquet'
        ids = tokenizer('\n\n'.join(pd.read_parquet(data)['text'].tolist()), return_tensors='pt').input_ids
        assert ids.numel() == 341469
        assert calibration.digest(data) == source['test_data']['parquet_sha256']
        assert hashlib.sha256(ids.numpy().tobytes()).hexdigest() == source['test_data']['token_ids_sha256']
        tokens = ids[0, :83*4096].reshape(83, 4096)
        for key, parameters, reference in [('default', calibration.DEFAULT, source['test_default']),
                                             ('selected', pair, source['test_selected'])]:
            record['stage'] = 'full_wikitext2_' + key
            save()
            calibration.configure(model, parameters)
            calibration.guard()
            model(torch.ones(1, 8, device='cuda', dtype=torch.long), use_cache=False)
            metric = calibration.token_eval(model, tokens, record.setdefault(key, {}), save)
            difference = abs(metric['ppl'] - reference['ppl'])
            record[key + '_calibration_reproduction_difference'] = difference
            save()
            if difference > 1e-6:
                raise RuntimeError(f'{key} rerun does not reproduce calibration test: {difference}')
        previous = read(ROOT / 'reproduction/quantization-profile.json')
        if name == '13b':
            previous = previous['model_comparisons']['llama2-13b']
        record['historical_ppl'] = previous['weight_activation_ppl']['4']['activation']['4']['ppl']['ppl']
        record.update(status='complete', stage='complete', finished_at=calibration.now(),
            delta_vs_same_weights_default=record['selected']['ppl']-record['default']['ppl'],
            delta_vs_historical=record['selected']['ppl']-record['historical_ppl'])
        save()
    except calibration.Contended as exc:
        record.update(status='waiting_for_gpu', interruption=str(exc))
        save()
        raise SystemExit(75)
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
        save()
        raise


def supervise():
    lock = (OUT / '.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(supervisor_pid=os.getpid(), started_at=calibration.now(), models=list(MODELS),
        source_sha256=sources(), dependency='both activation calibrations complete',
        status='waiting_for_calibration')

    def save(**fields):
        state.update(fields, updated_at=calibration.now())
        calibration.atomic_json(OUT / 'status.json', state)

    save()
    while not ready():
        time.sleep(30)
        save()
    for name in MODELS:
        while True:
            path = OUT / name / 'results.json'
            if path.exists() and read(path).get('status') == 'complete':
                publish()
                break
            save(status='waiting_for_gpu', current_model=name)
            checks = 0
            while checks < 3:
                checks = checks+1 if idle() else 0
                save(idle_checks=checks)
                if checks < 3:
                    time.sleep(10)
            # Calibration's supervisor must have released its final CUDA worker.
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                      '--model', name], cwd=ROOT)
            save(status='running', worker_pid=child.pid)
            code = child.wait()
            if code not in (0, 75):
                save(status='failed', exit_code=code)
                raise SystemExit(code)
    save(status='complete', finished_at=calibration.now())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--publish-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.publish_only:
        publish()
    elif args.model:
        worker(args.model)
    else:
        supervise()
