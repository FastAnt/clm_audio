"""Complete source-data commands; transport mirrors supply audio, AudioMCQ supplies all labels."""
import hashlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from .common import PROJECT, ROOT, sha256, write_json

ARCHIVES = [
    dict(name="compa_r_train_audios.tar.gz", url="https://huggingface.co/datasets/PinnHe/CompA-R-Backup/resolve/6cd6988a73abfd927c99ccbfbc84b3443a4cc3ab/compa_r_train_audios.tar.gz", bytes=31834749434,
         sha256="30df96adc17781ab8e8a59242811b186ca250d469bbc994c0cb78582e04fd1e6"),
    dict(name="tacos_audio.zip", url="https://zenodo.org/api/records/15379789/files/audio.zip/content", bytes=1629958981, md5="a1b11ed7a88d6b95109decb577c128fd"),
]


def download_archive(item):
    target = ROOT/"raw"/item["name"]
    if not target.exists() or target.stat().st_size != item["bytes"]:
        partial = target.with_suffix(target.suffix+".part")
        subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--continue-at", "-", "--output", str(partial), item["url"]], check=True)
        if partial.stat().st_size != item["bytes"]:
            raise ValueError(f"Incomplete source archive: {target}")
        partial.replace(target)
    kind = "sha256" if "sha256" in item else "md5"
    if kind == "sha256":
        checksum = sha256(target)
    else:
        checksum = hashlib.md5()
        with target.open("rb") as source:
            for chunk in iter(lambda:source.read(1024*1024), b""):
                checksum.update(chunk)
        checksum = checksum.hexdigest()
    if checksum != item[kind]:
        raise ValueError(f"Source archive checksum mismatch: {target}")
    write_json(target.with_suffix(target.suffix+".verified.json"), dict(**item, verified_checksum=checksum))


def download_all():
    subprocess.run([sys.executable, str(PROJECT/"scripts/download_metadata.py")], check=True)
    with ThreadPoolExecutor(max_workers=3) as workers:
        tasks = [workers.submit(download_archive, archive) for archive in ARCHIVES]
        tasks.append(workers.submit(subprocess.run, [sys.executable, str(PROJECT/"scripts/download_sources.py")], check=True))
        for task in tasks:
            task.result()


def prepare_all(arguments=None):
    subprocess.run([sys.executable, str(PROJECT/"scripts/prepare_full.py"), *(arguments or [])], check=True)
