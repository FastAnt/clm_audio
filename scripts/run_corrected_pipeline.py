"""Supervise corrected overfit, adapter comparison and staged training with audio-use gates."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from samsone_clm.common import PROJECT, ROOT, digest, read_rows, write_json
from samsone_clm.encoder import FrozenSamsone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16, help="Frozen encoder batch size")
    args = parser.parse_args()
    run_root = ROOT / "runs/v2"
    state = dict(status="running", run_root=str(run_root), completed_stages=[], supervisor_pid=os.getpid(),
                 architecture="frozen Samsone356M -> SAME adapter -> original CLM state/action heads",
                 encoder_recipe=FrozenSamsone.recipe)
    accepted_pair_mode = "same-options"

    def heartbeat(phase, **extra):
        state.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(ROOT / "runs/pipeline_status.json", state)

    def run(command, name, phase):
        log_path = run_root / name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log:
            child = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                heartbeat(phase, active_pid=child.pid, log=str(log_path))
                time.sleep(5)
            if child.returncode:
                raise RuntimeError(f"{phase} exited {child.returncode}: {log_path}")

    def feature_manifest(dataset):
        semantic = digest([{k:v for k,v in row.items() if k != "audio_sha256"} for row in read_rows(dataset)])
        for path in (ROOT / "features").glob("spike-*/manifest.json"):
            m = json.loads(path.read_text())
            if m["provenance"]["recipe"] == FrozenSamsone.recipe and m["provenance"]["dataset_semantic_sha256"] == semantic:
                return str(path)

    def precompute(name, pilot=False):
        dataset = ROOT / f"processed/v2/{name}.jsonl"
        if accepted_pair_mode == "reciprocal-answers" and name in ("stage2_50k", "stage3_150k", "stage4_all"):
            dataset = ROOT / f"processed/v2/{name}_audio_contrast_reciprocal.jsonl"
            if not dataset.exists():
                run([sys.executable, str(PROJECT / "scripts/prepare_audio_contrast.py"), "--pair-mode",
                     accepted_pair_mode, "--base-snapshot", name], f"prepare_contrast_{name}.log", f"preparing_contrast_{name}")
        # Join the initial pilot worker if it was already launched interactively.
        if pilot:
            while True:
                workers = []
                for p in Path("/proc").iterdir():
                    if not p.name.isdigit():
                        continue
                    try:
                        command = (p/"cmdline").read_bytes().decode()
                        if str(PROJECT/".venv/bin/python") in command and "scripts/precompute.py" in command and str(dataset) in command:
                            workers.append(int(p.name))
                    except (OSError, UnicodeError):
                        pass
                if not workers:
                    break
                heartbeat("precompute_pilot",active_pid=workers[0],log=str(run_root/"precompute_pilot.log"))
                time.sleep(5)
        manifest = feature_manifest(dataset)
        if manifest:
            return manifest
        command = [sys.executable,str(PROJECT/"scripts/precompute.py"),"--dataset",str(dataset),
                   "--batch-size",str(args.batch_size),"--shard-size","128" if pilot else "512"]
        if not pilot:
            command += ["--ablation-splits","validation","test"]
        run(command, f"precompute_{name}.log", f"precompute_{name}")
        manifest = feature_manifest(dataset)
        if not manifest:
            raise RuntimeError("Precompute produced no matching completed v2 manifest")
        return manifest

    def staged(number, manifest, kind, *, diagnostic=False, init=None, name=None, swap=0.):
        directory = run_root/(name or f"stage{number}")
        report = directory/"metrics.json"
        if report.exists() and json.loads(report.read_text()).get("status") == "completed":
            return json.loads(report.read_text())
        command = [sys.executable,"-m","samsone_clm.train","--stage",str(number),"--epochs",str(args.epochs),
                   "--feature-manifest",manifest,"--run-root",str(run_root),"--out",str(directory),
                   "--adapter-kind",kind,"--batch-size","64","--lr","0.0003","--head-lr","0.0001",
                   "--audio-swap-weight",str(swap)]
        if diagnostic:
            command += ["--diagnostic","--samples","10000"]
        if init:
            command += ["--init-checkpoint",str(init)]
        if swap > 0:
            command += ["--audio-pair-batch-size", "32", "--audio-pair-ce-weight", "0.5",
                        "--audio-pair-mode", accepted_pair_mode]
        if (directory/"last.pt").exists():
            command += ["--resume"]
        run(command, f"{directory.name}.log", f"training_{directory.name}")
        return json.loads(report.read_text())

    try:
        if not (ROOT/"processed/v2/snapshots.json").exists():
            run([sys.executable,str(PROJECT/"scripts/prepare_v2.py")],"prepare.log","freezing_dataset")
        pilot = precompute("pilot_1000",pilot=True)
        comparisons = {}
        for kind in ("mlp","linear"):
            directory = run_root/f"overfit-1000-{kind}"
            if not (directory/"metrics.json").exists() or not json.loads((directory/"metrics.json").read_text()).get("status") == "completed":
                run([sys.executable,"-m","samsone_clm.train","--overfit","--samples","1000","--steps","2000",
                     "--evaluate-every","50","--feature-manifest",pilot,"--adapter-kind",kind,"--out",str(directory)],
                    f"overfit-{kind}.log",f"overfit_{kind}")
            comparisons[kind] = json.loads((directory/"metrics.json").read_text())
        passed = [kind for kind,r in comparisons.items() if r["overfit_gate_passed"]]
        if not passed:
            raise RuntimeError("Corrected adapters did not pass the 1000-sample overfit gate; large training remains disabled")
        def validation(kind):
            return comparisons[kind]["evaluations"]["validation"]["correct"]["overall"]["top1"]
        kind = max(passed,key=validation)
        if "linear" in passed and validation("linear") >= validation(kind)-.02:
            kind = "linear"
        summary = {k:dict(overfit_gate_passed=r["overfit_gate_passed"],train=r["final"],
                         validation=r["evaluations"]["validation"],model=r["model"],best_step=r["best_step"]) for k,r in comparisons.items()}
        write_json(run_root/"adapter_comparison.json",dict(selected=kind,runs=summary,
                   selection="best validation Top-1; prefer fewer-parameter linear within 2 percentage points; no test selection"))
        state["adapter_kind"] = kind
        manifest1 = precompute("stage1_10k")
        staged(1,manifest1,kind)
        state["completed_stages"].append(1)
        # Train both heads in a bounded diagnostic before expanding beyond 10K.
        dense_reports = [json.loads(p.read_text()) for name in ("diagnostic-stage2-10k-dense-swap",
                                                               "diagnostic-stage2-10k-reciprocal-swap")
                         if (p := run_root / name / "metrics.json").exists()]
        passing_dense = [r for r in dense_reports if r.get("status") == "completed" and r.get("audio_dependence_gate_passed")]
        dense = max(passing_dense, key=lambda r:r["evaluations"]["validation"]["correct"]["overall"]["top1"], default={})
        dense_accepted = (dense.get("status") == "completed" and dense.get("audio_dependence_gate_passed")
                          and all(dense.get("trained_heads_changed", {}).get(h) for h in ("state", "action"))
                          and dense.get("pair_updates") == dense.get("last_step")
                          and dense.get("model", {}).get("contract_version") == 2
                          and dense.get("model", {}).get("adapter_kind") == kind)
        if dense_accepted:
            dense_manifest = json.loads(Path(dense["configuration"]["feature_manifest"]).read_text())
            original_manifest = json.loads(Path(manifest1).read_text())
            for split in ("validation", "test"):
                if [r["sample_id"] for r in dense_manifest["rows"] if r["split"] == split] != [
                        r["sample_id"] for r in original_manifest["rows"] if r["split"] == split]:
                    raise ValueError("Dense pilot changed the frozen evaluation split")
            pilot2 = dense
            accepted_pair_mode = dense["configuration"].get("audio_pair_mode", "same-options")
            state["audio_pair_mode"] = accepted_pair_mode
        else:
            pilot2 = staged(2,manifest1,kind,diagnostic=True,init=run_root/"stage1/best.pt",name="diagnostic-stage2-10k")
        if not pilot2["audio_dependence_gate_passed"]:
            # Only verified same-question/options, different-answer audio negatives.
            pilot2 = staged(2,manifest1,kind,diagnostic=True,init=run_root/"stage1/best.pt",
                            name="diagnostic-stage2-10k-swap",swap=1.)
        if not pilot2["audio_dependence_gate_passed"]:
            heartbeat("audio_use_diagnostic_failed",status="needs_diagnosis",active_pid=None,
                      error="Both heads trained on corrected 10K pilot; validation audio-use gate still failed. No larger training launched.")
            return
        for number,name in ((2,"stage2_50k"),(3,"stage3_150k"),(4,"stage4_all")):
            manifest = precompute(name)
            result = staged(number,manifest,kind,swap=1.)
            state["completed_stages"].append(number)
            if not result["audio_dependence_gate_passed"]:
                heartbeat("audio_use_diagnostic_failed",status="needs_diagnosis",active_pid=None,
                          error=f"Stage {number} failed the audio-use gate; larger stages not launched")
                return
        heartbeat("completed",status="completed",active_pid=None)
    except Exception as error:
        heartbeat("failed",status="failed",active_pid=None,error=str(error))
        raise


if __name__ == "__main__":
    main()
