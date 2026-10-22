"""Download official music data under the configured data root, with resumable bytes."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import requests

from samsone_clm.common import ROOT, sha256, write_json
from samsone_clm.resources import require_resources

MUSICBENCH_REV = 'b141e962aacc19ffd51c15732738040377989203'
MUSICBENCH_FILES = {
    'MusicBench_train.json': (85670037, '579d35b724391cb6fd2b0ae181bed074e536f44158e5ebfc0d683d323d23ac26'),
    'MusicBench.tar.gz': (16757993240, 'b4c16fba91d21b7662f595c7559ed71fac8448a24daab840345c05d9da727b92'),
}


def fetch(url, target, expected_sha=None, expected_bytes=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if expected_bytes is not None and target.stat().st_size != expected_bytes:
            raise ValueError(f'Existing size mismatch: {target}')
        if expected_sha is None or sha256(target) == expected_sha:
            return
        raise ValueError(f'Existing checksum mismatch: {target}')
    partial = target.with_name(target.name+'.part')
    for attempt in range(5):
        require_resources()
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with requests.get(url, headers={'Range':f'bytes={offset}-'} if offset else {},
                              stream=True, timeout=(30,60)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                if offset and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('Server returned a different resume range')
                with partial.open('ab' if offset else 'wb') as handle:
                    for block in response.iter_content(1024*1024):
                        handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            if expected_bytes is not None and partial.stat().st_size != expected_bytes:
                raise ValueError(f'Incomplete download: {target}')
            if expected_sha is not None and sha256(partial) != expected_sha:
                raise ValueError(f'Download checksum mismatch: {target}')
            partial.replace(target)
            return
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(min(8, 2**attempt))


def jamendo(workers):
    root = ROOT/'raw/music/jamendo'
    provenance_file = root/'provenance.json'
    if provenance_file.exists():
        revision = json.loads(provenance_file.read_text())['revision']
    else:
        response = requests.get('https://api.github.com/repos/MTG/mtg-jamendo-dataset/commits/master',timeout=30)
        response.raise_for_status()
        revision = response.json()['sha']
        write_json(provenance_file,dict(repo='MTG/mtg-jamendo-dataset',revision=revision,
                   annotations='three-annotator consensus; original full-track labels',
                   audio='official raw_30s/audio-low individual CDN files',complete=False))
    base = f'https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/{revision}'
    for source, name in [
        ('derived/music-classification-annotations/music-classification-annotations-clean.tsv','annotations.tsv'),
        ('data/download/raw_30s_audio-low_sha256_tracks.txt','audio_sha256.txt'),
        ('derived/music-classification-annotations/README.md','annotations_README.md'),
        ('audio_licenses.txt','audio_licenses.txt'),
    ]:
        fetch(base+'/'+source, root/name)
    checksums = {}
    for line in (root/'audio_sha256.txt').read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        checksums[name.lstrip('*')] = checksum
    selected = []
    for line in (root/'annotations.tsv').read_text().splitlines()[1:]:
        values = line.split('\t')
        audio = values[3].removesuffix('.mp3')+'.low.mp3'
        if audio not in checksums:
            raise ValueError(f'No official audio checksum: {audio}')
        selected.append((audio,checksums[audio]))
    def one(item):
        name, checksum = item
        fetch('https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/'+name,root/'audio'/name,checksum)
    count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in executor.map(one, selected):
            count += 1
            if count % 100 == 0:
                print(json.dumps(dict(dataset='jamendo',downloaded=count,total=len(selected))),flush=True)
                write_json(root/'download_progress.json',dict(downloaded=count,total=len(selected)))
    report = json.loads(provenance_file.read_text())
    report.update(complete=True, tracks=len(selected),metadata_sha256=sha256(root/'annotations.tsv'),
                  published_audio_checksums_sha256=sha256(root/'audio_sha256.txt'))
    write_json(provenance_file, report)


def musicbench():
    root = ROOT/'raw/music/musicbench'
    for name, (size, checksum) in MUSICBENCH_FILES.items():
        print(f'Downloading {name} ({size} bytes)',flush=True)
        fetch(f'https://huggingface.co/datasets/amaai-lab/MusicBench/resolve/{MUSICBENCH_REV}/{name}',
              root/name, checksum, size)
    write_json(root/'provenance.json',dict(repo='amaai-lab/MusicBench',revision=MUSICBENCH_REV,
               files={name:dict(bytes=size,sha256=checksum) for name,(size,checksum) in MUSICBENCH_FILES.items()},
               labels='automatic BPM/beat/chord/key; confidence filtering required'))


def nsynth():
    root = ROOT/'raw/music/nsynth'
    files = []
    for split in ('train','valid','test'):
        name = f'nsynth-{split}.jsonwav.tar.gz'
        # The official download host is a public GCS bucket. Using its HTTPS
        # path avoids that host's TLS name mismatch, without disabling TLS.
        url = 'https://storage.googleapis.com/download.magenta.tensorflow.org/datasets/nsynth/'+name
        response = requests.head(url,timeout=30)
        response.raise_for_status()
        size = int(response.headers['Content-Length'])
        if require_resources()['free_disk_gib'] < size/2**30+25:
            raise RuntimeError('Insufficient disk budget for NSynth archive')
        print(f'Downloading {name} ({size} bytes)',flush=True)
        fetch(url,root/name,expected_bytes=size)
        files.append(dict(name=name,bytes=size,sha256=sha256(root/name),
                          sha256_origin='measured download, not a published reference',
                          google_hash=response.headers.get('x-goog-hash')))
        write_json(root/'provenance.json',dict(source='official NSynth JSON/WAV download',
                   source_page='https://magenta.withgoogle.com/datasets/nsynth',files=files,
                   split='official instrument-disjoint train/valid/test',complete=split=='test'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['jamendo','nsynth','musicbench'],required=True)
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    require_resources()
    if args.dataset == 'jamendo':
        jamendo(args.workers)
    elif args.dataset == 'nsynth':
        nsynth()
    else:
        musicbench()
