"""Build a resumable rsync inventory; transfer the data volume separately from Git.

The training profile includes v3 checkpoints, four verified feature stores,
canonical snapshots and pretrained weights. The full profile adds all available
canonical audio and source/music metadata, but not duplicate source archives or
the host-specific Python environment. --verify reads every selected file.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from samsone_clm.common import PROJECT, ROOT, resolve_data_path, sha256, write_json


def collect_files(profile):
    config = json.loads((PROJECT / 'configs/migration.json').read_text())
    files = {}

    def add(path, expected=None, origin='measured'):
        path = resolve_data_path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(ROOT).as_posix()
        item = dict(path=relative, bytes=path.stat().st_size,
                    sha256=expected or sha256(path), checksum_source=origin)
        previous = files.get(relative)
        if previous and previous['sha256'] != item['sha256']:
            raise ValueError(f'Conflicting source checksums: {relative}')
        files[relative] = item

    original = json.loads((ROOT / 'models/provenance.json').read_text())
    expected = {resolve_data_path(item['path']): item['sha256'] for item in original}
    for name in ('samsone_checkpoint', 'clm_checkpoint'):
        path = ROOT / config[name]
        add(path, expected.get(path), 'pretrained_provenance')
    add(ROOT / 'models/provenance.json')
    for folder in ('models/samsone_clm_common', 'processed/samsone_clm_common'):
        checksums = json.loads((ROOT / folder / 'checksums.json').read_text())
        for path in sorted((ROOT / folder).iterdir()):
            add(path, checksums.get(path.name), 'immutable_export' if path.name in checksums else 'measured')
    for path in sorted((ROOT / 'processed/v3').iterdir()):
        if path.is_file():
            add(path)
    # Keep the v2 split report, which explains the immutable v3 split ancestry.
    for name in ('canonical.report.json', 'snapshots.json'):
        add(ROOT / 'processed/v2' / name)
    for folder in config['feature_directories']:
        directory = ROOT / folder
        manifest = json.loads((directory / 'manifest.json').read_text())
        for name, checksum in manifest['shard_sha256'].items():
            add(directory / name, checksum, 'feature_manifest')
        for path in sorted(directory.glob('*.json')):
            add(path)
    # Skip absolute demo symlinks. Explicit exported checkpoint paths work on
    # both hosts; bootstrap can recreate a relative demo symlink if desired.
    for path in sorted((ROOT / 'runs/v3').rglob('*')):
        if path.is_file() and not path.is_symlink() and path.suffix not in ('.tmp', '.part'):
            add(path)
    if profile == 'full':
        canonical = ROOT / config['canonical']
        audio = {}
        for line in canonical.open():
            row = json.loads(line)
            path = resolve_data_path(row['audio_path'])
            checksum = row['audio_sha256']
            if path in audio and audio[path] != checksum:
                raise ValueError(f'Conflicting audio provenance: {path}')
            audio[path] = checksum
        for path, checksum in sorted(audio.items()):
            add(path, checksum, 'canonical_audio_provenance')
        for path in sorted((ROOT / 'raw').glob('AudioMCQ_*.jsonl')):
            add(path)
        for path in sorted((ROOT / 'raw/music_expansion').glob('*')):
            if path.is_file():
                add(path)
        for path in sorted((ROOT / 'raw').rglob('*.json')):
            add(path)
        for path in sorted((ROOT / 'raw').rglob('*.csv')):
            add(path)
    return sorted(files.values(), key=lambda item: item['path'])


def verify_files(records):
    for number, item in enumerate(records, 1):
        path = ROOT / item['path']
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError(f'Transfer integrity failure: {item["path"]}')
        if number % 500 == 0:
            print(f'Verified {number}/{len(records)} files', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('training', 'full'), default='training')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/migration')
    parser.add_argument('--inventory', type=Path, help='Verify a source-generated inventory on the destination')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.inventory:
        if not args.verify:
            parser.error('--inventory requires --verify')
        records = json.loads(args.inventory.read_text())['files']
        verify_files(records)
        print(f'All {len(records)} transferred files verified')
        return
    records = collect_files(args.profile)
    if args.verify:
        verify_files(records)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(profile=args.profile, file_count=len(records), bytes=sum(item['bytes'] for item in records),
                   by_top_directory=dict(Counter(item['path'].split('/')[0] for item in records)),
                   source_data_root=str(ROOT), all_bytes_verified=args.verify,
                   exclusions=['Python environment/cache', 'duplicate source archives', 'legacy v1/v2 run directories'],
                   note='Inherited checksums are verified only with --verify; model/feature checksums remain unchanged after relocation.')
    write_json(args.output / f'{args.profile}.json', dict(summary=summary, files=records))
    write_json(args.output / f'{args.profile}.summary.json', summary)
    (args.output / f'{args.profile}.files').write_text(''.join(item['path'] + '\n' for item in records))
    # NUL-terminated names protect spaces and shell characters in source filenames.
    (args.output / f'{args.profile}.files0').write_bytes(b''.join(item['path'].encode() + b'\0' for item in records))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
