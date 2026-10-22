"""Resume the expanded verified-pair diagnostic; expand stages only if the audio gate passes."""
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
    run_root = ROOT / "runs/v2"
    name = "diagnostic-stage2-10k-reciprocal-swap"
    directory = run_root / name
    dataset = ROOT / "processed/v2/stage2_audio_contrast_reciprocal.jsonl"
    state = json.loads((ROOT / "runs/pipeline_status.json").read_text())
    state.pop("error", None)
    state.update(supervisor_pid=os.getpid(), status="running", completed_points=[1, 3, 4])

    def heartbeat(phase, **extra):
        state.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(ROOT / "runs/pipeline_status.json", state)

    def run(command, phase, filename):
        path = run_root / filename
        with path.open("a") as log:
            child = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                heartbeat(phase, active_pid=child.pid, log=str(path))
                time.sleep(5)
            if child.returncode:
                raise RuntimeError(f"{phase} exited {child.returncode}; {path}")

    def feature_manifest():
        semantic = digest([{k:v for k,v in r.items() if k != "audio_sha256"} for r in read_rows(dataset)])
        for path in (ROOT / "features").glob("spike-*/manifest.json"):
            m = json.loads(path.read_text())
            if (m["provenance"]["recipe"] == FrozenSamsone.recipe and
                m["provenance"]["dataset_semantic_sha256"] == semantic and
                (path.parent / "extraction_audit.json").exists()):
                return str(path)

    try:
        manifest = feature_manifest()
        if not manifest:
            run([sys.executable, str(PROJECT / "scripts/precompute.py"), "--dataset", str(dataset),
                 "--batch-size", "16", "--shard-size", "512", "--ablation-splits", "validation", "test"],
                "precompute_audio_contrast_reciprocal", "precompute_audio_contrast_reciprocal.log")
            manifest = feature_manifest()
        if not manifest:
            raise RuntimeError("No verified completed feature manifest")
        state["completed_points"] = [1, 2, 3, 4]
        report_path = directory / "metrics.json"
        if not report_path.exists() or json.loads(report_path.read_text()).get("status") != "completed":
            command = [sys.executable, "-m", "samsone_clm.train", "--stage", "2", "--diagnostic", "--samples", "10000",
                "--epochs", "12", "--feature-manifest", manifest, "--run-root", str(run_root), "--out", str(directory),
                "--init-checkpoint", str(run_root / "stage1/best.pt"), "--adapter-kind", "linear", "--batch-size", "64",
                "--lr", "0.0003", "--head-lr", "0.0001", "--audio-swap-weight", "1", "--audio-swap-margin", "0.05",
                "--audio-pair-batch-size", "32", "--audio-pair-ce-weight", "0.5", "--audio-pair-mode", "reciprocal-answers"]
            if (directory / "last.pt").exists():
                command += ["--resume"]
            run(command, "training_" + name, name + ".log")
        state["completed_points"] = [1, 2, 3, 4, 5]
        run([sys.executable, str(PROJECT / "scripts/report_audio_contrast.py"), "--name", name],
            "evaluating_audio_contrast_reciprocal", "audio_contrast_report_reciprocal.log")
        state["completed_points"] = [1, 2, 3, 4, 5, 6]
        run([sys.executable, str(PROJECT / "scripts/audit_corrected_audio.py"), "--checkpoints",
             str(run_root / "stage1/best.pt"), str(run_root / "diagnostic-stage2-10k-dense-swap/best.pt"),
             str(directory / "best.pt"), "--feature-manifest", manifest, "--pair-mode", "reciprocal-answers",
             "--output", str(run_root / "audio_contrast_signal_audit_reciprocal.json")],
            "diagnosing_audio_contrast_reciprocal", "audio_contrast_signal_audit_reciprocal.log")
        report = json.loads(report_path.read_text())
        if not report["audio_dependence_gate_passed"]:
            heartbeat("audio_contrast_plan_completed_gate_failed", status="needs_diagnosis", active_pid=None,
                      completed_points=[1, 2, 3, 4, 5, 6, 7], diagnostics_completed=True,
                      contrast_result=str(run_root / "audio_contrast_result_reciprocal.json"),
                      signal_audit=str(run_root / "audio_contrast_signal_audit_reciprocal.json"),
                      error="Expanded verified-pair pilot completed and audited; held-out audio gate failed. Larger stages were not launched.")
            return
        # The main pipeline validates and accepts this bounded pilot, then starts 50K from best Stage 1.
        with (run_root / "pipeline_continued.log").open("a") as log:
            child = subprocess.Popen([sys.executable, str(PROJECT / "scripts/run_corrected_pipeline.py")],
                                     cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            if child.wait():
                raise RuntimeError("Large-stage continuation failed; pipeline_continued.log")
    except Exception as error:
        heartbeat("audio_contrast_failed", status="failed", active_pid=None, error=str(error))
        raise


if __name__ == "__main__":
    main()
