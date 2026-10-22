"""Restore SHA256-verified learned models from this repository's Git LFS objects."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/srv/data/samsone_clm')
MODEL_NAMES = {'common': 'samsone_clm_common', 'music': 'samsone_clm_music'}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def matches(path, expected, size):
    return path.is_file() and path.stat().st_size == size and digest(path) == expected


def verified_copy(source, destination, expected, size):
    """Preserve an existing export; publish a new file only after verifying its bytes."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if not matches(destination, expected, size):
            raise ValueError('Refusing to overwrite a different model file: ' + str(destination))
        return 'already verified'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.download-', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            with source.open('rb') as original:
                shutil.copyfileobj(original, handle, length=4 * 1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        if not matches(temporary, expected, size):
            raise ValueError('Downloaded file checksum or size mismatch: ' + str(source))
        os.replace(temporary, destination)
        return 'restored'
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('common', 'music', 'all'), default='all')
    parser.add_argument('--data-root', type=Path, default=Path(os.environ.get('SAMSONE_CLM_DATA', DEFAULT_ROOT)))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    if not (PROJECT / 'checkpoints/catalog.json').is_file():
        parser.error('Checkpoint catalog is missing; restore checkpoints/catalog.json from this repository.')
    catalog = json.loads((PROJECT / 'checkpoints/catalog.json').read_text())
    names = list(MODEL_NAMES.values()) if args.model == 'all' else [MODEL_NAMES[args.model]]
    storage = root / 'cache/git-lfs-objects'
    if not args.verify_only:
        if shutil.which('git-lfs') is None:
            parser.error('Install Git LFS first: https://git-lfs.com/')
        include = ','.join('checkpoints/' + name + '/*.pt' for name in names)
        subprocess.run(['git', '-c', 'lfs.storage=' + str(storage), 'lfs', 'fetch', 'origin',
                        '--include=' + include, '--exclude='], cwd=PROJECT, check=True)
    count = 0
    for name in names:
        for item in catalog['models'][name]['files']:
            filename = item['name']
            if Path(filename).name != filename or item['path'] != 'checkpoints/' + name + '/' + filename:
                raise ValueError('Unsafe checkpoint catalog path')
            target = root / 'models' / name / filename
            if args.verify_only:
                if not matches(target, item['sha256'], item['bytes']):
                    raise ValueError('Model verification failed: ' + str(target))
                action = 'verified'
            else:
                sha = item['sha256']
                source = storage / 'objects' / sha[:2] / sha[2:4] / sha if item['storage'] == 'git-lfs' else PROJECT / item['path']
                action = verified_copy(source, target, sha, item['bytes'])
            count += 1
            print(action + ': ' + str(target), flush=True)
    print(json.dumps({'files_verified': count, 'models': names, 'data_root': str(root)}))


if __name__ == '__main__':
    main()
