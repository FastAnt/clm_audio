"""Remote-only FMA genre continuation: guarded jobs, exact overfit and validation selection."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from samsone_clm.common import ROOT, PROJECT, resolve_data_path, sha256, write_json
from samsone_clm.monitor import read_json, render_png
from samsone_clm.resources import require_resources, resource_status

BASE = ROOT / "runs/music-genre"
COMMON_SHA = "4f4137e03975054ffe725ec635589afa31c8ddfd11b1d71db20e863a43710e78"
START_SHA = "c9c00f307c12828eee9e66893bd3e7f4c6eba3a2a477d8e4ff4bdaafeed7f694"


def alive(pid: int) -> bool:
    try:
        proc = Path("/proc") / str(pid)
        return (pid > 0 and proc.joinpath("stat").read_text().split(")", 1)[1].split()[0] != "Z"
                and proc.joinpath("cwd").resolve() == PROJECT.resolve()
                and os.getpgid(pid) == pid)
    except (OSError, IndexError):
        return False


def pid_for(name):
    p = BASE / (name + ".pid")
    return int(p.read_text().strip()) if p.exists() else -1


class Pipeline:
    def __init__(self):
        BASE.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.version = "v1"
        self.phase = "starting"
        self.log = BASE / "pipeline.log"
        self.parent = ROOT / "models/samsone_clm_music_window_lowlr/best.pt"

    def status(self, state="running", **extra):
        record = dict(phase=self.phase, status=state,
            updated_at=datetime.now(timezone.utc).isoformat(),
            elapsed_s=time.time() - self.started, music_only=True, local_training=False,
            remote_data_root=str(ROOT), run_root=str(BASE), resources=resource_status(),
            parent_checkpoint=str(self.parent), log=str(self.log),
            stage_directories={"4": str(ROOT / "runs/remote-stage4/stage4"),
                               "5": str(ROOT / "runs/music/stage5"),
                               "6": str(BASE / ("iteration-" + self.version) / ("stage6-genre-" + self.version))},
            experiment="FMA primary-genre 10s windows, question/choice variants, balanced CE and music replay; frozen Samsone356M",
            source="FMA medium then FMA large", **extra)
        write_json(BASE / "status.json", record)
        # The already-running demo imported the preceding monitor version.
        write_json(ROOT / "runs/music/refinement_status.json", record)

    def spawn(self, name, arguments):
        require_resources()
        if alive(pid_for(name)):
            return
        previous = read_json(BASE / (name + ".exit.json"))
        if previous:
            if previous["exit_code"] == 0:
                return
            raise RuntimeError(f"{name} previously failed: inspect its log before retrying")
        cmd = [sys.executable, "-u", *map(str, arguments)]
        worker = (shlex.join(cmd) + "\ncode=$?\n" +
            "printf '{\"exit_code\":%s}\\n' \"$code\" > " +
            shlex.quote(str(BASE / (name + ".exit.json.tmp"))) + "\n" +
            "mv " + shlex.quote(str(BASE / (name + ".exit.json.tmp"))) + " " +
            shlex.quote(str(BASE / (name + ".exit.json"))) + "\nexit \"$code\"\n")
        with (BASE / (name + ".log")).open("ab", buffering=0) as log:
            child = subprocess.Popen(["bash", "-c", worker], cwd=PROJECT,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (BASE / (name + ".pid")).write_text(str(child.pid) + "\n")
        write_json(BASE / (name + ".command.json"), dict(arguments=cmd, pid=child.pid,
                   started_at=datetime.now(timezone.utc).isoformat()))
        print("LAUNCHED", name, child.pid, shlex.join(cmd), flush=True)

    def background_preparation(self):
        if not (ROOT / "raw/music/fma/fma_large.zip.verified.json").exists():
            return
        report = ROOT / "processed/music_genre/fma_large.report.json"
        if report.exists() or alive(pid_for("prepare-large")):
            return
        if (BASE / "prepare-large.exit.json").exists():
            result = read_json(BASE / "prepare-large.exit.json")
            if result.get("exit_code"):
                write_json(BASE / "large_preparation_failure.json", result)
            return
        resources = resource_status()
        # Preparation and frozen feature extraction may overlap; there is one GPU job.
        if resources["available_ram_gib"] >= 30 and resources["free_disk_gib"] >= 90:
            self.spawn("prepare-large", ["scripts/prepare_genre_data.py", "--subset", "large", "--workers", "4"])

    def tick(self):
        self.background_preparation()
        self.status()

    def wait_external(self, name, output):
        self.phase, self.log = name, BASE / (name + ".log")
        while not output.exists():
            if not alive(pid_for(name)):
                if output.exists():
                    break
                raise RuntimeError(f"{name} stopped without its completed output; inspect {self.log}")
            self.tick()
            time.sleep(15)
        self.status()

    def job(self, name, arguments):
        self.phase, self.log = name, BASE / (name + ".log")
        self.spawn(name, arguments)
        while True:
            result = read_json(BASE / (name + ".exit.json"))
            if result:
                if result["exit_code"]:
                    raise RuntimeError(f"{name} failed with exit code {result['exit_code']}; inspect {self.log}")
                self.status()
                return
            if not alive(pid_for(name)):
                time.sleep(1)
                result = read_json(BASE / (name + ".exit.json"))
                if not result or result["exit_code"]:
                    raise RuntimeError(f"{name} exited without successful status; inspect {self.log}")
                return
            self.tick()
            time.sleep(15)

    @staticmethod
    def pointer(name):
        item = read_json(ROOT / "features" / (name + ".json"))
        path = resolve_data_path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError("Feature manifest pointer checksum mismatch")
        return path

    def select_publish(self, run, feature):
        report = read_json(run / "metrics.json")
        accepted = bool(report.get("genre_gate_passed"))
        decision = dict(accepted=accepted, test_used_for_selection=False,
            parent_sha256=sha256(self.parent), candidate_sha256=sha256(run / "best.pt"),
            initial_validation=report["initial_validation"],
            selected_validation=report["evaluations"]["validation"],
            selection="Genre macro gain >=2pp; worst question gain >=1pp; each question audio gaps >=5pp; replay family loss <=1pp")
        write_json(BASE / ("genre-selection-" + self.version + ".json"), decision)
        if not accepted:
            render_png(6, run)
            write_json(run / "checksums.json", {p.name: sha256(p) for p in run.iterdir()
                       if p.is_file() and p.name != "checksums.json"})
            print("Validation gate failed; existing published model preserved", self.version, flush=True)
            return
        export = ROOT / "models" / ("samsone_clm_music_genre_" + self.version)
        self.job("finalize-" + self.version, ["scripts/finalize_music_stage.py", "--stage", "6",
                 "--feature-manifest", feature, "--run-dir", run, "--export-dir", export])
        self.job("publish-" + self.version, ["scripts/publish_music_model.py", "--export-dir", export,
                 "--archive-name", "samsone_clm_music_before_genre_" + self.version])
        demo = ROOT / "runs/music/demo.pt"
        temporary = demo.with_suffix(".pt.tmp")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(run / "calibrated.pt")
        temporary.replace(demo)
        write_json(ROOT / "runs/served_checkpoint.json", dict(stage=6, checkpoint=str(run / "calibrated.pt"),
                   checkpoint_sha256=sha256(run / "calibrated.pt"),
                   selection="genre validation gate, then validation-only calibration"))
        self.parent = export / "best.pt"

    def iteration(self, subset, version):
        self.version = version
        feature = self.pointer("music_genre_fma_" + subset + "_manifest")
        pointer = ROOT / "features" / ("music_genre_" + version + "_manifest.json")
        self.job("compose-" + version, ["scripts/compose_music_features.py", "--stage", "6",
                 "--new-manifests", feature,
                 "--common-manifest", ROOT / "features/spike-a86e057eae695e43/manifest.json",
                 "--stage5-manifest", ROOT / "features/music-stage6-6e48fecc8cce6e96/manifest.json",
                 "--replay-train-ratio", ".35", "--pointer", pointer])
        combined = self.pointer("music_genre_" + version + "_manifest")
        run_root = BASE / ("iteration-" + version)
        common = ["scripts/train_music.py", "--stage", "6", "--feature-manifest", combined,
                  "--run-root", run_root, "--parent-checkpoint", self.parent, "--resume"]
        self.job("overfit-" + version, [*common, "--diagnostic"])
        gate = read_json(run_root / "overfit_stage6_gate.json")
        if not gate.get("passed"):
            raise RuntimeError("Exact-feature overfit gate failed; larger training remains stopped")
        run_name = "stage6-genre-" + version
        self.job("train-" + version, [*common, "--run-name", run_name, "--epochs", "32",
                 "--patience", "0", "--batch-size", "256", "--lr", ".00003",
                 "--head-lr", ".000015", "--scale-lr", ".0003",
                 "--selection-metric", "genre-grounded", "--balance-genre-ce"])
        run = run_root / run_name
        # Point the existing UI at the actual optimizer run, rather than the job wrapper.
        self.status(actual_run=str(run))
        self.select_publish(run, combined)

    def run(self):
        if sha256(ROOT / "models/samsone_clm_common/best.pt") != COMMON_SHA:
            raise ValueError("Immutable common checkpoint changed")
        if sha256(self.parent) != START_SHA:
            raise ValueError("Unexpected initial music parent")
        medium = ROOT / "features/music_genre_fma_medium_manifest.json"
        self.wait_external("precompute-medium", medium)
        # The initial worker started before grouped rare-genre ablations landed.
        # Reuse its SHA-verified correct features and compute only missing controls.
        self.job("precompute-medium-balanced", ["-m", "samsone_clm.precompute", "--dataset",
                 ROOT / "processed/music_genre/fma_medium.jsonl", "--batch-size", "64",
                 "--shard-size", "512", "--ablation-splits", "validation", "test",
                 "--ablation-limit-per-split", "4096", "--pointer", medium])
        self.iteration("medium", "v1")
        self.phase, self.log = "prepare-large", BASE / "prepare-large.log"
        report = ROOT / "processed/music_genre/fma_large.report.json"
        while not report.exists():
            failure = read_json(BASE / "prepare-large.exit.json")
            if failure.get("exit_code"):
                raise RuntimeError("FMA large preparation failed; v1 artifacts preserved")
            if not (ROOT / "raw/music/fma/fma_large.zip.verified.json").exists() and not alive(pid_for("download-large")):
                raise RuntimeError("FMA large download stopped without SHA verification")
            self.tick()
            time.sleep(15)
        self.version = "v2"
        self.job("precompute-large", ["-m", "samsone_clm.precompute", "--dataset",
                 ROOT / "processed/music_genre/fma_large.jsonl", "--batch-size", "64",
                 "--shard-size", "512", "--ablation-splits", "validation", "test",
                 "--ablation-limit-per-split", "4096", "--pointer",
                 ROOT / "features/music_genre_fma_large_manifest.json"])
        self.iteration("large", "v2")
        if sha256(ROOT / "models/samsone_clm_common/best.pt") != COMMON_SHA:
            raise ValueError("Immutable common checkpoint changed")
        write_json(BASE / "completed.json", dict(completed=True, versions=["v1", "v2"],
                   artifacts_remote_only=True, common_sha256=COMMON_SHA,
                   selected_parent=str(self.parent), timestamp=time.time()))
        self.phase = "completed"
        self.status("completed")


if __name__ == "__main__":
    pipeline = Pipeline()
    try:
        pipeline.run()
    except BaseException as error:
        pipeline.phase = "failed"
        pipeline.status("failed", error=str(error))
        raise
