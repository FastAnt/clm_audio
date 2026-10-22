import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from samsone_clm.common import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "runs/overfit-256-seed42-lr0.001")
    run = parser.parse_args().run
    report = json.loads((run / "metrics.json").read_text())
    history = report["history"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    steps = [r["step"] for r in history]
    axes[0].plot(steps, [r["loss"] for r in history], marker="o", color="#2463a5")
    axes[0].axhline(report["pure_cosine_four_choice_ce_lower_bound"], linestyle="--", color="gray", label="cosine CE lower bound")
    count = report["final"]["samples"]
    axes[0].set(xlabel="Training steps", ylabel="Cross entropy", title=f"Adapter-only overfit, {count} samples")
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, [100*r["top1"] for r in history], marker="o", color="#178052", label="train")
    validation = report["evaluations"]["validation"]
    for mode, marker in [("correct", "o"), ("random", "x"), ("silence", "+")]:
        axes[1].scatter([report["best_step"]], [100*validation[mode]["overall"]["top1"]], marker=marker, s=90, label=f"validation {mode}")
    axes[1].set(xlabel="Training steps", ylabel="Top-1 accuracy (%)", ylim=(0, 105), title="Train and audio ablations")
    axes[1].legend(fontsize=8)
    fig.savefig(run / "overfit.png", dpi=180)
    fig.savefig(run / "overfit.pdf")


if __name__ == "__main__":
    main()
