"""Restore Llama-2-7B using the mirror and checksum method used for 13B."""

import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent
DEST = Path('/mnt/model/llama-2-7b-hf')
REPO = 'NousResearch/Llama-2-7b-hf'
REVISION = '8efe6c9b93655b934e27bd9981e3ec13e55aee9d'
HOST = 'https://hf-mirror.com'
REPORT = ROOT / 'llama2-7b-download.json'
MANIFEST = ROOT / 'llama2-7b-download-manifest.json'


def emit(*args):
    print(datetime.datetime.now().astimezone().isoformat(), *args, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(entry):
    name = entry['rfilename']
    target = DEST / name
    partial = target.with_name(name + '.partial')
    size = entry['size']
    expected_sha = entry.get('lfs', {}).get('sha256')

    def valid(path):
        return (path.exists() and path.stat().st_size == size
                and (not expected_sha or sha256(path) == expected_sha))

    if not valid(target):
        for attempt in range(1, 9):
            if valid(partial):
                partial.replace(target)
                break
            if partial.exists() and partial.stat().st_size >= size:
                raise RuntimeError(f'Invalid completed partial file: {partial}')
            url = f'{HOST}/{REPO}/resolve/{REVISION}/{name}?download=true&attempt={attempt}'
            emit('DOWNLOAD', name, 'attempt', attempt)
            result = subprocess.run([
                'curl', '-fLsS', '--connect-timeout', '15', '--max-time', '1800',
                '--speed-time', '90', '--speed-limit', '1024',
                '-C', '-', '-o', str(partial), url,
            ])
            if valid(partial):
                partial.replace(target)
                break
            emit('RETRY', name, 'exit', result.returncode,
                 'bytes', partial.stat().st_size if partial.exists() else 0)
            time.sleep(min(attempt * 3, 20))
        else:
            raise RuntimeError(f'Download failed: {name}')
    actual_sha = sha256(target)
    emit('VERIFIED', name, size, actual_sha)
    return {'file': name, 'bytes': size, 'sha256': actual_sha,
            'expected_lfs_sha256': expected_sha}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    lock = (DEST / '.download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = json.loads(MANIFEST.read_text())
    assert meta['sha'] == REVISION
    names = {'config.json', 'generation_config.json', 'model.safetensors.index.json',
             'tokenizer.model', 'tokenizer.json', 'tokenizer_config.json',
             'special_tokens_map.json', 'LICENSE.txt', 'USE_POLICY.md'}
    selected = [e for e in meta['siblings']
                if e['rfilename'].endswith('.safetensors') or e['rfilename'] in names]
    assert sum(e['rfilename'].endswith('.safetensors') for e in selected) == 2
    state = dict(repo=REPO, revision=REVISION, endpoint=HOST, destination=str(DEST),
                 pid=os.getpid(), status='downloading', verified_files=[],
                 total_bytes=sum(e['size'] for e in selected))

    def save():
        state['updated_at'] = datetime.datetime.now().astimezone().isoformat()
        state['downloaded_bytes_on_disk'] = sum(
            p.stat().st_size for p in DEST.iterdir() if p.is_file())
        tmp = REPORT.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(REPORT)

    save()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            pending = {pool.submit(fetch, e) for e in sorted(selected, key=lambda e: e['size'])}
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    state['verified_files'].append(future.result())
                save()
                emit('PROGRESS', len(state['verified_files']), '/', len(selected),
                     'files;', state['downloaded_bytes_on_disk'], '/', state['total_bytes'], 'bytes')
        index = json.loads((DEST / 'model.safetensors.index.json').read_text())
        assert set(index['weight_map'].values()) == {
            e['rfilename'] for e in selected if e['rfilename'].endswith('.safetensors')}
        reference = ROOT / 'wush-comparison-h128/calibration.json'
        if reference.exists():
            emit('TOKENIZER_SHA256', sha256(DEST / 'tokenizer.model'))
        state['status'] = 'complete'
        save()
        emit('ALL_DOWNLOADS_VERIFIED')
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    main()
