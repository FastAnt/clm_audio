from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
from pathlib import Path

import requests

from .common import ROOT, digest, identity_split, read_rows, sha256, write_json


def download(url: str, target: Path, expected_bytes: int | None = None) -> None:
    if target.exists() and (expected_bytes is None or target.stat().st_size == expected_bytes):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with partial.open("wb") as f:
            for block in response.iter_content(1024 * 1024):
                f.write(block)
    if expected_bytes and partial.stat().st_size != expected_bytes:
        raise ValueError(f"Wrong download size for {target}")
    partial.replace(target)


def download_spike(source_shards: int = 1) -> None:
    if not 1 <= source_shards <= 6:
        raise ValueError("Clotho development has six source shards")
    # All labels come from AudioMCQ. gijs/clotho is only a transport mirror of source waveforms.
    metadata_api = requests.get("https://huggingface.co/api/datasets/inclusionAI/AudioMCQ", timeout=30)
    metadata_api.raise_for_status()
    revision = metadata_api.json()["sha"]
    metadata_path = "datasets/AudioMCQ_Clotho.jsonl"
    download(f"https://huggingface.co/datasets/inclusionAI/AudioMCQ/resolve/{revision}/{metadata_path}",
             ROOT / "raw/AudioMCQ_Clotho.jsonl", 6684123)
    response = requests.get("https://huggingface.co/api/datasets/gijs/clotho", timeout=30)
    response.raise_for_status()
    audio_revision = response.json()["sha"]
    tree = requests.get(f"https://huggingface.co/api/datasets/gijs/clotho/tree/{audio_revision}/data", timeout=30)
    tree.raise_for_status()
    shards = sorted((r for r in tree.json() if r["path"].startswith("data/train-")), key=lambda r: r["path"])[:source_shards]
    def fetch(item):
        target = ROOT / "raw" / ("clotho_train_" + item["path"].split("train-")[1].split("-of-")[0] + ".parquet")
        print(f"Source download: {item['path']} ({item['size']} bytes)", flush=True)
        download(f"https://huggingface.co/datasets/gijs/clotho/resolve/{audio_revision}/{item['path']}", target, item["size"])
        result = dict(path=item["path"], local_path=str(target), bytes=item["size"], sha256=sha256(target))
        print(f"Source ready: {target.name}", flush=True)
        return result
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as pool:
        downloaded = list(pool.map(fetch, shards))
    write_json(ROOT / "raw/download_report.json", dict(scope=f"Clotho spike, {source_shards} source audio shards",
               labels=dict(repo="inclusionAI/AudioMCQ", revision=revision, path=metadata_path),
               source_audio=dict(dataset="Clotho v2 development", mirror="gijs/clotho",
                                 revision=audio_revision, shards=downloaded),
               metadata_sha256=sha256(ROOT / "raw/AudioMCQ_Clotho.jsonl"),
               source_shard_sha256=sha256(ROOT / "raw/clotho_train_00000.parquet")))


def prepare_spike(samples: int = 256, heldout: int = 32) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    metadata = read_rows(ROOT / "raw/AudioMCQ_Clotho.jsonl")
    grouped = collections.defaultdict(list)
    for row in metadata:
        if row.get("audio-contribution", row.get("audio_contribution")) == "strong":
            grouped[Path(row["audio_path"]).name].append(row)
    counts, records, rejects = collections.Counter(), [], []
    target_counts = {"train": samples, "validation": heldout, "test": heldout}
    audio_dir = ROOT / "processed/audio/clotho"
    audio_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = sorted((ROOT / "raw").glob("clotho_train_*.parquet"))
    seen_samples = set()
    source_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in shard_paths)
    batches = (batch for path in shard_paths for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["file_name", "audio"]))
    for batch in batches:
        for source in batch.to_pylist():
            name = source["file_name"]
            if name not in grouped:
                continue
            try:
                audio, sample_rate = sf.read(io.BytesIO(source["audio"]["bytes"]), dtype="float32", always_2d=True)
                audio = audio.mean(axis=1)
                if not len(audio) or not np.isfinite(audio).all() or len(audio) / sample_rate > 30.001:
                    raise ValueError("Empty/nonfinite/over-context audio")
                # Identity based on decoded content; same PCM in different files stays in the same split.
                identity = "pcm:" + digest([sample_rate, hashlib.sha256(audio.tobytes()).hexdigest()])
                split = identity_split(identity)
                if counts[split] >= target_counts[split]:
                    continue
                target = audio_dir / name
                if target.exists():
                    existing, existing_rate = sf.read(target, dtype="float32")
                    if existing_rate != sample_rate or not np.array_equal(existing, audio):
                        raise ValueError(f"Existing source audio content changed: {target}")
                else:
                    sf.write(target, audio, sample_rate, subtype="FLOAT")
                for row in grouped[name]:
                    if counts[split] >= target_counts[split]:
                        break
                    if len(set(row["choices"])) != len(row["choices"]):
                        raise ValueError("Duplicate choices")
                    matches = [i for i, text in enumerate(row["choices"]) if text == row["answer"]]
                    if len(matches) != 1:
                        raise ValueError("Answer does not uniquely match choices")
                    canonical = {k: row[k] for k in ("question", "choices", "question_type", "source_dataset")}
                    canonical.update(audio_path=str(target), answer_index=matches[0], audio_identity=identity,
                                     source_audio_id=name, duration_s=len(audio) / sample_rate, split=split,
                                     audio_contribution="strong", audio_sha256=sha256(target))
                    canonical["sample_id"] = digest([identity, row["question"], row["choices"]])
                    if canonical["sample_id"] in seen_samples:
                        continue
                    seen_samples.add(canonical["sample_id"])
                    records.append(canonical)
                    counts[split] += 1
            except Exception as exc:
                rejects.append(dict(audio=name, error=str(exc)))
    records.sort(key=lambda r: (r["split"], r["sample_id"]))
    if counts["train"] < samples:
        raise ValueError(f"Only {counts['train']} train rows available; requested {samples}. Download another source shard.")
    identities = {s: {r["audio_identity"] for r in records if r["split"] == s} for s in target_counts}
    assert not any(identities[a] & identities[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")])
    manifest = ROOT / "processed/spike.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    report = dict(scope="strong AudioMCQ / Clotho only", samples=dict(counts),
                  unique_audio={s: len(ids) for s, ids in identities.items()}, rejects=rejects,
                  manifest_sha256=sha256(manifest), split_recipe="decoded PCM identity hash v1, 80/10/10",
                  full_strong_metadata=sum(map(len, grouped.values())), source_shard_rows=source_rows,
                  source_shards=[str(p) for p in shard_paths])
    write_json(ROOT / "processed/prepare_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["download", "prepare"])
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--heldout", type=int, default=32)
    ap.add_argument("--source-shards", type=int, default=1)
    args = ap.parse_args()
    download_spike(args.source_shards) if args.command == "download" else prepare_spike(args.samples, args.heldout)


if __name__ == "__main__":
    main()
