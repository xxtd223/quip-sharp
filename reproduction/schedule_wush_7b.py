"""Wait for a quiet night, finish the H128 suite, and publish measured rows."""

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproduction/wush-comparison-h128'
RESULT = OUT / 'results.json'
MAIN = ROOT / 'reproduction/RESULTS.md'
STATUS = OUT / 'night-7b-status.json'
PYTHON = Path('/home/jylu/miniconda3/envs/wush/bin/python')
ZONE = ZoneInfo('Asia/Shanghai')


def now():
    return datetime.now(ZONE)


def log(*args):
    print(now().isoformat(), *args, flush=True)


def atomic_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sync_results():
    """Upsert only measured H128 7B rows in the current user-edited table."""
    data = json.loads(RESULT.read_text())
    measured = []
    for fmt in ('mxfp4', 'nvfp4'):
        record = data['runs'].get(f'7b-{fmt}-wush-h128', {})
        if record.get('status') != 'complete':
            continue
        metric = record['ppl']
        assert (metric['context_length'], metric['blocks'], metric['scored_tokens']) == (4096, 83, 339885)
        reference = data['runs']['13b-mxfp4-wush-h128']['ppl']
        assert metric['test_parquet_sha256'] == reference['test_parquet_sha256']
        ppl = metric['ppl']
        group = 32 if fmt == 'mxfp4' else 16
        row = (f'| Llama-2-7B | WUSH + Hadamard + GPTQ | {fmt.upper()}，group={group} '
               f'| 128 | {ppl:.6f} | {ppl - 5.423203:+.6f} |')
        measured.append((fmt, row))
    if not measured:
        return []
    original = MAIN.read_text()
    start = original.index('### WUSH + GPTQ FP4 W4A4 对照')
    end = original.index('\n### ', start + 4)
    lines = original[start:end].splitlines()
    for fmt, row in measured:
        matching = []
        for i, line in enumerate(lines):
            cols = [c.strip() for c in line.split('|')[1:-1]]
            if (len(cols) == 6 and cols[0] == 'Llama-2-7B' and 'WUSH' in cols[1]
                    and fmt.upper() in cols[2] and cols[3] == '128'):
                matching.append(i)
        assert len(matching) <= 1
        if matching:
            lines[matching[0]] = row
        else:
            position = next(i for i, line in enumerate(lines) if line.startswith('| Llama-2-13B |'))
            lines.insert(position, row)
    updated = original[:start] + '\n'.join(lines) + '\n' + original[end:]
    if len(measured) == 2:
        updated = updated.replace('7B 的 H=128 两组尚未运行', '7B 的 H=128 两组已完成')
    if updated != original:
        backup = OUT / 'main-RESULTS.before-7b.md'
        if not backup.exists():
            backup.write_text(original)
        if MAIN.read_text() != original:
            raise RuntimeError('RESULTS.md changed during sync; retry to preserve user edits')
        temp = MAIN.with_suffix('.md.night7b.tmp')
        temp.write_text(updated)
        temp.replace(MAIN)
        log('RESULTS_UPDATED', [fmt for fmt, _ in measured])
    return [fmt for fmt, _ in measured]


def gpu_idle():
    processes = subprocess.check_output([
        'nvidia-smi', '--id=0', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    memory, utilization = map(int, subprocess.check_output([
        'nvidia-smi', '--id=0', '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(','))
    return not processes and memory < 1024 and utilization < 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--not-before', default='2026-10-28T00:00:00+08:00')
    parser.add_argument('--sync-only', action='store_true')
    args = parser.parse_args()
    if args.sync_only:
        print(sync_results())
        return
    lock = (OUT / '.night-7b.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    earliest = datetime.fromisoformat(args.not_before)
    state = dict(pid=os.getpid(), not_before=earliest.isoformat(),
                 status='scheduled', interval_seconds=30,
                 idle_checks_required=3, night_hours='00:00–07:00 Asia/Shanghai')

    def save(**fields):
        state.update(fields, updated_at=now().isoformat())
        atomic_json(STATUS, state)

    try:
        data = json.loads(RESULT.read_text())
        for path, expected in data['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
        download = json.loads((ROOT / 'reproduction/llama2-7b-download.json').read_text())
        assert download['status'] == 'complete'
        for item in download['verified_files']:
            p = Path(download['destination']) / item['file']
            assert p.exists() and p.stat().st_size == item['bytes'], str(p)
        previous_13b = {k: v for k, v in data['runs'].items() if k.startswith('13b-')}
        assert len(previous_13b) == 2 and all(v['status'] == 'complete' for v in previous_13b.values())
        save()
        log('SCHEDULED', earliest.isoformat())
        checks = 0
        while checks < 3:
            current = now()
            if current < earliest or current.hour >= 7:
                checks = 0
                save(status='waiting_for_night', idle_checks=checks)
            else:
                try:
                    checks = checks + 1 if gpu_idle() else 0
                    save(status='waiting_for_idle_gpu', idle_checks=checks)
                except (subprocess.SubprocessError, ValueError) as exc:
                    checks = 0
                    save(status='gpu_probe_retry', error=repr(exc))
            if checks < 3:
                time.sleep(30)
        backup = OUT / 'results.before-7b.json'
        if not backup.exists():
            shutil.copy2(RESULT, backup)
        command = [str(PYTHON), '-u', str(ROOT / 'reproduction/run_wush_h128.py'),
                   '--models', '7b', '13b', '--transforms', 'wush']
        # Keeping both models retains 13B metadata/reports; completed jobs are skipped.
        with (OUT / 'night-7b-run.log').open('a') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            save(status='running', runner_pid=child.pid, started_at=now().isoformat())
            log('STARTED', child.pid)
            while child.poll() is None:
                save(published_formats=sync_results())
                time.sleep(30)
        published = sync_results()
        after = json.loads(RESULT.read_text())
        assert {k: v for k, v in after['runs'].items() if k.startswith('13b-')} == previous_13b
        if child.returncode or len(published) != 2:
            raise RuntimeError(f'Runner exit={child.returncode}; published formats={published}')
        save(status='complete', published_formats=published, finished_at=now().isoformat())
        log('COMPLETE', '7B MXFP4 and NVFP4 published; 13B records preserved')
    except BaseException as exc:
        save(status='failed', error=repr(exc))
        log('FAILED', repr(exc))
        raise


if __name__ == '__main__':
    main()
