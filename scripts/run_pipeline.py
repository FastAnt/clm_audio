"""Durable supervised AudioMCQ pipeline: 10K -> 50K -> 150K -> all available strong samples."""
import argparse
import concurrent.futures
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from samsone_clm.common import PROJECT, ROOT, digest, read_rows, sha256, write_json

STATUS = ROOT / "runs/pipeline_status.json"
COMP_SIZE = 31834749434
COMP_SHA = "30df96adc17781ab8e8a59242811b186ca250d469bbc994c0cb78582e04fd1e6"


def processes(script, needle=""):
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry/"cmdline").read_bytes().decode().replace("\0", " ")
            if str(PROJECT/".venv/bin/python") in args and script in args and needle in args:
                found.append(int(entry.name))
        except (OSError, UnicodeError):
            pass
    return found


def counts():
    try:
        db = sqlite3.connect(f"file:{ROOT/'processed/source_index.sqlite'}?mode=ro", uri=True, timeout=1)
        n = db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        sources = db.execute("SELECT COUNT(*) FROM completed_sources").fetchone()[0]
        db.close()
        return dict(indexed_samples=n, prepared_source_files=sources)
    except sqlite3.Error:
        return {}


def source_download_complete():
    plan = json.loads((ROOT/"raw/source_download_plan.json").read_text())
    return all(Path(r["target"]).with_suffix(Path(r["target"]).suffix+".verified.json").exists() for r in plan["files"])


def finalize_comp():
    partial = ROOT/"raw/compa_r_train_audios.tar.gz.part"
    target = partial.with_suffix("")
    if target.exists():
        return True
    if partial.exists() and partial.stat().st_size == COMP_SIZE:
        if sha256(partial) != COMP_SHA:
            raise ValueError("CompA-R archive checksum mismatch")
        partial.replace(target)
        write_json(target.with_suffix(".verified.json"), dict(repo="PinnHe/CompA-R-Backup", revision="6cd6988a73abfd927c99ccbfbc84b3443a4cc3ab", sha256=COMP_SHA, bytes=COMP_SIZE))
        return True
    return False


def feature_manifest(dataset):
    semantic = digest([{k:v for k,v in row.items() if k != "audio_sha256"} for row in read_rows(dataset)])
    for p in sorted((ROOT/"features").glob("spike-*/manifest.json")):
        m = json.loads(p.read_text())
        if m["provenance"]["dataset_semantic_sha256"] == semantic and m["provenance"].get("ablation_splits") == ["test", "validation"]:
            return str(p)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32, help="Frozen encoder precompute batch")
    args = parser.parse_args()
    state = dict(status="running", architecture="frozen Samsone -> SAME adapter -> original state/action heads", completed_stages=[])
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    archive_job = None
    prep_job = None
    prep_log = None
    preparation_enabled = True
    print(json.dumps(dict(event="pipeline_started", pid=os.getpid(), epochs=args.epochs,
                          time=datetime.now(timezone.utc).isoformat())), flush=True)

    def heartbeat(phase, **extra):
        changed = state.get("phase") != phase
        state.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **counts(), **extra)
        write_json(STATUS, state)
        if changed:
            print(json.dumps(state), flush=True)

    def preparation_tick():
        nonlocal archive_job, prep_job, prep_log
        if not preparation_enabled:
            return
        if archive_job is None:
            p = ROOT/"raw/compa_r_train_audios.tar.gz.part"
            if p.exists() and p.stat().st_size == COMP_SIZE:
                archive_job = pool.submit(finalize_comp)
        elif archive_job.done():
            archive_job.result()
        if prep_job is not None and prep_job.poll() is not None:
            code = prep_job.returncode
            prep_job = None
            prep_log.close()
            prep_log = None
            if code not in (0, -15, 143):
                raise RuntimeError(f"Source preparation failed with exit {code}; see prepare_full.log")
        if prep_job is None and not processes("scripts/prepare_full.py", "--output"):
            prep_log = (ROOT/"runs/prepare_full.log").open("a")
            prep_job = subprocess.Popen([sys.executable, str(PROJECT/"scripts/prepare_full.py"), "--output", str(ROOT/"processed/full.jsonl")], cwd=PROJECT, stdout=prep_log, stderr=subprocess.STDOUT)

    def run(command, log_name, phase):
        with (ROOT/"runs"/log_name).open("a") as log:
            child = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                heartbeat(phase, active_pid=child.pid, log=str(ROOT/"runs"/log_name))
                preparation_tick()
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(f"{phase} failed with exit {child.returncode}; see {log_name}")

    def snapshot(dataset, train_limit, heldout):
        cmd = [sys.executable, str(PROJECT/"scripts/prepare_full.py"), "--snapshot-only", "--output", str(dataset)]
        if train_limit:
            cmd += ["--train-limit", str(train_limit)]
        if heldout:
            cmd += ["--heldout-limit", str(heldout)]
        run(cmd, f"snapshot_{dataset.stem}.log", "creating_dataset_snapshot")

    def precompute(dataset, phase):
        while active := processes("scripts/precompute.py", str(dataset)):
            log = ROOT/"runs"/f"precompute_{dataset.stem}.log"
            if dataset.stem == "stage1_10k" and not log.exists():
                log = ROOT/"runs/precompute_stage1.log"
            heartbeat(phase, active_pid=active[0], log=str(log))
            preparation_tick()
            time.sleep(10)
        existing = feature_manifest(dataset)
        if existing:
            return existing
        run([sys.executable, str(PROJECT/"scripts/precompute.py"), "--dataset", str(dataset), "--batch-size", str(args.batch_size),
             "--shard-size", "512", "--ablation-splits", "validation", "test", "--ablation-limit-per-split", "2048"], f"precompute_{dataset.stem}.log", phase)
        path = feature_manifest(dataset)
        if not path:
            raise RuntimeError("Precompute exited without a matching completed feature manifest")
        return path

    def stage(number, manifest):
        report_path = ROOT/f"runs/stage{number}/metrics.json"
        if report_path.exists() and json.loads(report_path.read_text()).get("status") == "completed":
            state["completed_stages"].append(number)
            return
        cmd = [sys.executable, "-m", "samsone_clm.train_all", "--resume", "--start-stage", str(number), "--end-stage", str(number),
               "--epochs", str(args.epochs), "--feature-manifest", manifest]
        run(cmd, f"stage{number}.log", f"training_stage_{number}")
        state["completed_stages"].append(number)
        heartbeat(f"stage_{number}_completed", checkpoint=str(ROOT/f"runs/stage{number}/best.pt"))

    try:
        first = ROOT/"processed/stage1_10k.jsonl"
        if not first.exists():
            snapshot(first, 10000, 1000)
        if json.loads(first.with_suffix(".report.json").read_text())["selected_samples"].get("train",0) < 10000:
            raise RuntimeError("Initial corpus does not yet contain 10K train samples")
        first_manifest = precompute(first, "precompute_stage_1_10k")
        stage(1, first_manifest)
        second = ROOT/"processed/stage2_50k.jsonl"
        while True:
            if second.exists() and second.with_suffix(".report.json").exists():
                report = json.loads(second.with_suffix(".report.json").read_text())
                if report["selected_samples"].get("train",0)==50000:
                    break
            snapshot(second, 50000, 2000)
            report = json.loads(second.with_suffix(".report.json").read_text())
            if report["selected_samples"].get("train",0) == 50000:
                break
            heartbeat("waiting_for_50k_source_audio", available_train=report["available_samples"].get("train",0))
            time.sleep(30)
        second_manifest = precompute(second, "precompute_stage_2_50k")
        stage(2, second_manifest)
        while not source_download_complete() or not (ROOT/"raw/compa_r_train_audios.tar.gz").exists():
            heartbeat("waiting_for_full_source_download")
            preparation_tick()
            time.sleep(20)
        # Finish the active preparer, then ensure every completed source/archive was indexed.
        while prep_job is not None and prep_job.poll() is None or processes("scripts/prepare_full.py", "--output"):
            heartbeat("finishing_full_source_preparation")
            time.sleep(10)
        if prep_log:
            prep_log.close()
        prep_job, prep_log = None, None
        run([sys.executable, str(PROJECT/"scripts/prepare_full.py"), "--output", str(ROOT/"processed/full.jsonl")], "prepare_final.log", "final_full_dataset_preparation")
        preparation_enabled = False
        full = ROOT/"processed/full.jsonl"
        report = json.loads(full.with_suffix(".report.json").read_text())
        if report["selected_samples"].get("train",0) < 150000:
            raise RuntimeError(f"Available source audio yields only {report['selected_samples'].get('train',0)} train samples; cannot falsely claim a 150K Stage 3")
        full_manifest = precompute(full, "precompute_full_strong_corpus")
        stage(3, full_manifest)
        stage(4, full_manifest)
        run([sys.executable, "-m", "samsone_clm.evaluate", "--checkpoint", str(ROOT/"runs/stage4/best.pt"), "--split", "test", "--feature-manifest", full_manifest], "evaluate_test.log", "final_test_evaluation")
        run([sys.executable, str(PROJECT/"scripts/verify_inference.py"), "--checkpoint", str(ROOT/"runs/stage4/best.pt"), "--feature-manifest", full_manifest], "inference_final.log", "real_audio_inference_verification")
        heartbeat("completed", status="completed", active_pid=None, checkpoint=str(ROOT/"runs/stage4/best.pt"))
    except BaseException as error:
        heartbeat("failed", status="failed", error=str(error))
        raise
    finally:
        pool.shutdown(wait=False)

if __name__ == "__main__":
    main()
