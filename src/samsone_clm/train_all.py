import argparse
import json
import subprocess
import sys
from pathlib import Path

from .common import ROOT, PROJECT,digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe',choices=('v3','manual','music'),default='v3')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--feature-manifest")
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--end-stage", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.0001)
    parser.add_argument("--head-lr", type=float, default=.0001)
    parser.add_argument("--run-root", type=Path, default=ROOT/"runs/v2")
    parser.add_argument("--adapter-kind", choices=("mlp", "linear"), default="mlp")
    parser.add_argument("--overfit-report", type=Path)
    args = parser.parse_args()
    if args.recipe == 'music':
        from .common import sha256
        root = ROOT/'runs/music'
        start = 5 if args.start_stage == 1 else args.start_stage
        end = 6 if args.end_stage == 4 else args.end_stage
        if not 5 <= start <= end <= 6:
            raise ValueError('Music recipe supports only Stage 5 and 6')
        for stage in range(start,end+1):
            if stage == 6:
                returned = json.loads((root/'returned_stage5.json').read_text())
                if not returned['verified_locally'] or returned['best_sha256'] != sha256(root/'stage5/best.pt'):
                    raise ValueError('Stage 5 artifacts must be returned and verified locally before Stage 6')
            for diagnostic in (True,False):
                command = [sys.executable,str(PROJECT/'scripts/train_music.py'),'--stage',str(stage)]
                if diagnostic:
                    command += ['--diagnostic']
                if args.resume:
                    command += ['--resume']
                subprocess.run(command,check=True)
        return
    if args.recipe=='v3':
        # The supervisor uses immutable per-stage data, the mechanical gate and resource checks.
        # It resumes committed shards/checkpoints automatically; avoid an ambient feature pointer.
        subprocess.run([sys.executable,str(PROJECT/'scripts/run_v3_pipeline.py')],cwd=PROJECT,check=True)
        return
    args.overfit_report = args.overfit_report or args.run_root/f"overfit-1000-{args.adapter_kind}/metrics.json"
    gate = json.loads(args.overfit_report.read_text())
    from .encoder import FrozenSamsone
    if (not gate.get("overfit_gate_passed") or gate.get("feature_provenance", {}).get("recipe") != FrozenSamsone.recipe
        or not gate.get("model", {}).get("head_input_l2") or gate.get("model", {}).get("score_scale") != 100.):
        raise ValueError("Run the mechanical overfit test successfully before staged training")
    # Audio ablation is an outcome metric; it does not prohibit training the pretrained CLM heads.
    feature_path = args.feature_manifest or json.loads((ROOT/"features/spike_manifest.json").read_text())["path"]
    manifest = json.loads(Path(feature_path).read_text())
    manifest_hash = digest(manifest)
    for stage in range(args.start_stage, args.end_stage+1):
        run = args.run_root/f"stage{stage}"
        report = json.loads((run/"metrics.json").read_text()) if (run/"metrics.json").exists() else {}
        if args.resume and report.get("status")=="completed":
            # Later stages may use a larger corpus; completed stages keep their own immutable manifests.
            print(f"Stage {stage} already completed: {run}", flush=True)
            continue
        cmd = [sys.executable, "-m", "samsone_clm.train", "--stage", str(stage), "--epochs", str(args.epochs),
               "--batch-size", str(args.batch_size), "--lr", str(args.lr), "--head-lr", str(args.head_lr),
               "--feature-manifest", feature_path, "--out", str(run),
               "--run-root", str(args.run_root), "--adapter-kind", args.adapter_kind]
        if args.resume and (run/"last.pt").exists():
            import torch
            ck = torch.load(run/"last.pt", map_location="cpu", weights_only=True)
            if ck["feature_manifest"] != manifest_hash:
                raise ValueError(f"Cannot resume Stage {stage} on a different feature manifest")
            del ck
            cmd += ["--resume"]
        print(f"Launching Stage {stage}", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
