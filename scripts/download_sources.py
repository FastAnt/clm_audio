"""Pinned transport mirrors of AudioMCQ source audio; all labels remain AudioMCQ."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from samsone_clm.common import ROOT, sha256, write_json
from samsone_clm.data import download

REPOS = {
    "clotho": ("gijs/clotho", "93be4f14a2d662389bb30c324781d2b489b583fd"),
    "audiocaps": ("OpenSound/AudioCaps", "b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded"),
    "libritts_r": ("mythicinfinity/libritts_r", "0d1718db351207d5a493135a62d5ed8cc4733830"),
    "magnatagatune": ("confit/magnatagatune", "042e24c2c26a71168dd35edc3bc0cb47b2f3c3cd"),
    "musiccaps": ("CLAPv2/MusicCaps", "02cc1186f497984e7f0c4e74dbc536c862904655"),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--only", choices=list(REPOS), nargs="+")
    args = parser.parse_args()
    if not args.only or "musiccaps" in args.only:
        csv_path = ROOT/"raw/sources/musiccaps/musiccaps-public.csv"
        google_revision = "0a51889b340037bb75a9a0858af2e4ece21f7f89"
        download(f"https://huggingface.co/datasets/google/MusicCaps/resolve/{google_revision}/musiccaps-public.csv", csv_path)
        write_json(csv_path.with_suffix(".provenance.json"), dict(repo="google/MusicCaps", revision=google_revision, sha256=sha256(csv_path)))
    plan = []
    for source, (repo, revision) in REPOS.items():
        if args.only and source not in args.only:
            continue
        response = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}", params={"recursive":"true", "limit":1000}, timeout=30)
        response.raise_for_status()
        for entry in response.json():
            path = entry["path"]
            wanted = path.startswith("data/train-") and path.endswith(".parquet") if source == "clotho" else path.endswith(".parquet") if source in ("audiocaps", "musiccaps") else path.startswith("data/train.") and path.endswith(".parquet") if source == "libritts_r" else path == "mp3.zip"
            if wanted:
                target = (ROOT / "raw" / ("clotho_train_" + Path(path).name.split("train-")[1].split("-of-")[0] + ".parquet")) if source == "clotho" else ROOT / "raw/sources" / source / Path(path).name
                plan.append(dict(source=source, repo=repo, revision=revision, path=path, target=str(target),
                                 bytes=entry["size"], expected_sha256=entry.get("lfs", {}).get("oid")))
    # Interleave speech/sound shards so the first usable corpus already covers both.
    plan.sort(key=lambda item: (int(Path(item["path"]).stem.split("-")[-3]) if "-of-" in item["path"] else -1,
                                item["source"], item["path"]))
    combined = plan
    if args.only and (ROOT/"raw/source_download_plan.json").exists():
        old = json.loads((ROOT/"raw/source_download_plan.json").read_text())["files"]
        combined = [i for i in old if i["source"] not in args.only] + plan
    write_json(ROOT / "raw/source_download_plan.json", dict(files=combined, bytes=sum(i["bytes"] for i in combined)))
    print(f"SOURCE PLAN: {len(plan)} files, {sum(i['bytes'] for i in plan)/1e9:.2f} GB", flush=True)
    def fetch(item):
        target = Path(item["target"])
        for attempt in range(5):
            try:
                download(f"https://huggingface.co/datasets/{item['repo']}/resolve/{item['revision']}/{item['path']}", target, item["bytes"])
                checksum = sha256(target)
                if item["expected_sha256"] and checksum != item["expected_sha256"]:
                    raise ValueError(f"Source checksum mismatch: {target}")
                write_json(target.with_suffix(target.suffix + ".verified.json"), dict(**item, sha256=checksum))
                print(f"SOURCE READY {item['source']} {target.name} {item['bytes']/1e6:.1f} MB", flush=True)
                return
            except Exception as error:
                print(f"SOURCE RETRY {attempt+1} {target.name}: {error}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(min(20, 2 ** attempt))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch, plan))
    print("ALL PLANNED SOURCES DOWNLOADED AND VERIFIED", flush=True)

if __name__ == "__main__":
    main()
