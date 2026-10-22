"""Build strong-only AudioMCQ incrementally from verified source audio, with audio-disjoint splits."""
import argparse
import collections
import csv
import hashlib
import io
import json
import sqlite3
import tarfile
import zipfile
import fcntl
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from samsone_clm.common import ROOT, digest, identity_split, sha256, write_json
from samsone_clm.audio import decode_audio,write_audio_atomic

FAMILIES = {"Clotho":"clotho", "AudioCaps":"audiocaps", "CompA-R":"compa_r", "SpeechCraft":"libritts_r",
            "LP-MusicCaps-MTT":"magnatagatune", "MusicCaps":"musiccaps", "TACOS":"tacos"}


def recording_identity(family, name, pcm):
    stem = Path(name).stem
    if family == "audiocaps":
        return "youtube:" + stem.rsplit("_", 1)[0]
    if family == "compa_r" and len(stem) == 12 and stem.startswith("Y"):
        return "youtube:" + stem[1:]
    if family == "musiccaps":
        return "youtube:" + stem
    if family == "tacos":
        return "freesound:" + stem
    if family == "libritts_r":
        return "libritts_r:" + stem
    if family == "magnatagatune":
        parts = stem.rsplit("-", 2)
        if len(parts)==3 and parts[1].isdigit() and parts[2].isdigit():
            return "magnatagatune:" + parts[0]
    return pcm


def archive_label_name(family, name):
    name = Path(name).name
    # Official CompA-R backup stores raw YouTube IDs; AudioMCQ labels prepend AudioSet's Y marker.
    if family == "compa_r" and len(Path(name).stem)==11:
        return "Y" + name
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "processed/full.jsonl")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--heldout-limit", type=int)
    parser.add_argument("--snapshot-only", action="store_true", help="Write a snapshot of already committed source audio without processing new files")
    args = parser.parse_args()
    lock=(ROOT/'processed/.prepare_full.lock').open('a')
    if not args.snapshot_only:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another source-preparation writer is active; refusing concurrent WAV writes')
    offset_path = ROOT/"raw/sources/musiccaps/musiccaps-public.csv"
    music_offsets = {r["ytid"]:(float(r["start_s"]),float(r["end_s"])) for r in csv.DictReader(offset_path.open())} if offset_path.exists() else {}
    groups = collections.defaultdict(list)
    totals = collections.Counter()
    for path in sorted((ROOT / "raw").glob("AudioMCQ_*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            if row.get("audio-contribution", row.get("audio_contribution")) == "strong":
                family = FAMILIES[row["source_dataset"]]
                groups[family, Path(row["audio_path"]).name].append(row)
                totals[family] += 1
    db_path = ROOT / "processed/source_index.sqlite"
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60) if args.snapshot_only else sqlite3.connect(db_path, timeout=60)
    if not args.snapshot_only:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS samples (id TEXT PRIMARY KEY, row TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS completed_sources (path TEXT PRIMARY KEY)")
        db.execute("CREATE TABLE IF NOT EXISTS rejects (key TEXT PRIMARY KEY, error TEXT)")
    done = {x[0] for x in db.execute("SELECT path FROM completed_sources")}

    def accept(family, name, payload):
        labels = groups.get((family, Path(name).name), [])
        if not labels:
            return
        try:
            native_path = ROOT / "processed/audio" / family / (Path(name).name if family == "clotho" else digest([family, name])[:24] + ".wav")
            native_path.parent.mkdir(parents=True, exist_ok=True)
            if native_path.exists():
                audio, rate = sf.read(native_path, dtype="float32", always_2d=True)
                audio = audio.mean(axis=1)
            else:
                try:
                    audio, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
                    audio = audio.mean(axis=1)
                except Exception:
                    temporary = ROOT / "tmp" / (digest([family, name])[:24] + Path(name).suffix)
                    temporary.write_bytes(payload)
                    audio, rate = decode_audio(temporary), 16000
                    temporary.unlink()
                if not len(audio) or not np.isfinite(audio).all() or len(audio) / rate > 30.001:
                    raise ValueError("empty/nonfinite/audio exceeds Samsone 30-second context; no label-changing crop")
                write_audio_atomic(native_path,audio,rate)
            if not len(audio) or not np.isfinite(audio).all() or len(audio) / rate > 30.001:
                raise ValueError("audio exceeds supported context or is invalid")
            pcm = "pcm:" + digest([rate, hashlib.sha256(audio.tobytes()).hexdigest()])
            recording = recording_identity(family, name, pcm)
            checksum = sha256(native_path)
            for label in labels:
                if len(set(label["choices"])) != len(label["choices"]):
                    raise ValueError("duplicate choices")
                matches = [i for i, c in enumerate(label["choices"]) if c == label["answer"]]
                if len(matches) != 1:
                    raise ValueError("answer does not uniquely match choices")
                row = {k: label[k] for k in ("question", "choices", "question_type", "source_dataset")}
                row.update(audio_path=str(native_path), answer_index=matches[0], pcm_identity=pcm,
                           source_recording_id=recording, source_audio_id=name, duration_s=len(audio)/rate,
                           audio_contribution="strong", audio_sha256=checksum)
                row["sample_id"] = digest([pcm, row["question"], row["choices"]])
                db.execute("INSERT OR IGNORE INTO samples VALUES (?,?)", (row["sample_id"], json.dumps(row, ensure_ascii=False)))
            if db.total_changes % 200 == 0:
                db.commit()
        except Exception as error:
            db.execute("INSERT OR REPLACE INTO rejects VALUES (?,?)", (f"{family}/{name}", str(error)))

    # Prioritize the compact temporal/music sources before the much larger speech transport.
    archives = [("tacos", ROOT/"raw/tacos_audio.zip"), ("magnatagatune", ROOT/"raw/sources/magnatagatune/mp3.zip"),
                ("compa_r", ROOT/"raw/compa_r_train_audios.tar.gz")]
    for family, path in ([] if args.snapshot_only else archives):
        if not path.exists() or str(path) in done:
            continue
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    label_name = archive_label_name(family, name)
                    if (family, label_name) in groups:
                        accept(family, label_name, archive.read(name))
        else:
            with tarfile.open(path, "r|gz") as archive:
                for member in archive:
                    label_name = archive_label_name(family, member.name)
                    if member.isfile() and (family, label_name) in groups:
                        accept(family, label_name, archive.extractfile(member).read())
        db.execute("INSERT OR IGNORE INTO completed_sources VALUES (?)", (str(path),))
        db.commit()
        done.add(str(path))
        print(f"Prepared archive {family} {path.name}", flush=True)
    sources = [("clotho", p) for p in sorted((ROOT/"raw").glob("clotho_train_*.parquet"))]
    sources += [(p.parent.name, p) for p in sorted((ROOT/"raw/sources").glob("*/*.parquet")) if p.with_suffix(p.suffix+".verified.json").exists()]
    for family, path in ([] if args.snapshot_only else sources):
        if str(path) in done:
            continue
        parquet = pq.ParquetFile(path)
        columns = {"clotho":["file_name", "audio"], "audiocaps":["youtube_id", "start_time", "audio"],
                   "libritts_r":["id", "audio"], "musiccaps":["index", "start_s", "end_s", "audio"]}[family]
        for batch in parquet.iter_batches(batch_size=16, columns=columns):
            for source in batch.to_pylist():
                name = source["file_name"] if family == "clotho" else f"{source['youtube_id']}_{source['start_time']}.wav" if family == "audiocaps" else source["index"] if family == "musiccaps" else source["id"] + ".wav"
                if family == "musiccaps" and abs(float(source["end_s"])-float(source["start_s"])-10.)>.01:
                    raise ValueError("MusicCaps source mirror window is not the original 10-second clip")
                if family == "musiccaps" and (family, name) in groups:
                    expected = music_offsets.get(Path(name).stem)
                    if expected != (float(source["start_s"]),float(source["end_s"])):
                        raise ValueError(f"MusicCaps window does not match the official Google source offsets: {name}")
                accept(family, name, source["audio"]["bytes"])
        db.execute("INSERT OR IGNORE INTO completed_sources VALUES (?)", (str(path),))
        db.commit()
        print(f"Prepared source {family} {path.name}", flush=True)
    records = [json.loads(row[0]) for row in db.execute("SELECT row FROM samples ORDER BY id")]
    # Union exact PCM duplicates and shared source recording IDs, even across source datasets.
    parent = {}
    def root(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def priority(key):
        return (0 if key.startswith("youtube:") else 1 if key.startswith("freesound:") else 2 if key.startswith("libritts_r:") else 3, key)
    for row in records:
        a, b = root(row["pcm_identity"]), root(row["source_recording_id"])
        if a != b:
            first, second = sorted((a, b), key=priority)
            parent[second] = first
    for row in records:
        row["audio_identity"] = root(row["pcm_identity"])
        row["split"] = identity_split(row["audio_identity"])
    available = collections.Counter(row["split"] for row in records)
    records.sort(key=lambda r: (r["split"], digest([42, r["sample_id"]])))
    selected, counts = [], collections.Counter()
    for row in records:
        limit = args.train_limit if row["split"] == "train" else args.heldout_limit
        if limit is None or counts[row["split"]] < limit:
            selected.append(row)
            counts[row["split"]] += 1
    identities = {s:{r["audio_identity"] for r in selected if r["split"]==s} for s in ("train", "validation", "test")}
    assert not any(identities[a] & identities[b] for a,b in (("train","validation"),("train","test"),("validation","test")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in selected))
    temporary.replace(args.output)
    available_by_source = collections.Counter(r["source_dataset"] for r in records)
    report = dict(metadata_strong=dict(totals), available_samples=dict(available), selected_samples=dict(counts),
                  available_by_source=dict(available_by_source), question_types=dict(collections.Counter(r["question_type"] for r in selected)),
                  unique_audio={s:len(ids) for s,ids in identities.items()}, manifest=str(args.output), sha256=sha256(args.output),
                  source_files_completed=db.execute("SELECT COUNT(*) FROM completed_sources").fetchone()[0],
                  rejected_audio=db.execute("SELECT COUNT(*) FROM rejects").fetchone()[0],
                  missing_source_audio_is_excluded=True, split_recipe="recording IDs plus exact decoded PCM union; audio-split-v1 80/10/10")
    write_json(args.output.with_suffix(".report.json"), report)
    print(json.dumps(report, indent=2), flush=True)
    db.close()

if __name__ == "__main__":
    main()
