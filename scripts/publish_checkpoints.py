"""Publish the latest trained CLM checkpoint for the local demo while the full pipeline continues."""
import json
import time
from pathlib import Path
from samsone_clm.common import ROOT, write_json


def main():
    previous = None
    while True:
        candidate = ROOT/"runs/diagnostic-stage2-10k/best.pt"
        status_path = ROOT/"runs/pipeline_status.json"
        pipeline = json.loads(status_path.read_text()) if status_path.exists() else {}
        run_root = Path(pipeline.get("run_root", ROOT/"runs"))
        corrected = run_root != ROOT/"runs"
        if corrected:
            candidate = ROOT/"runs/latest.pt"
        for stage in (2, 3, 4):
            run = run_root/f"stage{stage}"
            info = run/"best.pt.info.json"
            metrics_path = run/"metrics.json"
            passed = not corrected or (metrics_path.exists() and json.loads(metrics_path.read_text()).get("audio_dependence_gate_passed"))
            if passed and info.exists() and json.loads(info.read_text())["step"] > 0 and (run/"best.pt").exists():
                candidate = run/"best.pt"
        if candidate == ROOT/"runs/latest.pt":
            candidate = candidate.resolve()
        if candidate.exists():
            current = (str(candidate), candidate.stat().st_mtime_ns)
            if current != previous:
                temporary = ROOT/"runs/.latest.pt.tmp"
                temporary.unlink(missing_ok=True)
                temporary.symlink_to(candidate)
                temporary.replace(ROOT/"runs/latest.pt")
                previous = current
                write_json(ROOT/"runs/served_checkpoint.json", dict(checkpoint=str(candidate), stamp=current[1]))
                print(f"Published trained CLM checkpoint: {candidate}", flush=True)
        if status_path.exists() and json.loads(status_path.read_text()).get("status")=="completed":
            break
        time.sleep(10)

if __name__ == "__main__":
    main()
