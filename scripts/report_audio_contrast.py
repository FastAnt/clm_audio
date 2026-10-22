"""Summarize the controlled dense-pair pilot and export its training curves."""
import json
import os
import argparse
from datetime import datetime, timezone

from samsone_clm.common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="diagnostic-stage2-10k-dense-swap")
    args = parser.parse_args()
    run_root = ROOT / "runs/v2"
    name = args.name
    run = run_root / name
    current = json.loads((run / "metrics.json").read_text())
    if current["status"] != "completed":
        raise ValueError("Wait for the complete pilot")
    comparisons = {}
    for label in dict.fromkeys(("stage1", "diagnostic-stage2-10k", "diagnostic-stage2-10k-swap",
                               "diagnostic-stage2-10k-dense-swap", name)):
        r = json.loads((run_root / label / "metrics.json").read_text())
        val = r["evaluations"]["validation"]
        comparisons[label] = dict(top1={m:val[m]["overall"]["top1"] for m in ("correct", "random", "silence")},
            audio_top1_gaps=val["audio_top1_gaps"], correct_metrics=val["correct"]["overall"],
            category_gaps={c:{m:val["correct"][c]["top1"]-val[m][c]["top1"] for m in ("random", "silence")}
                           for c in ("music", "sound", "speech", "temporal")})
    assert current["pair_updates"] == current["last_step"]
    assert all(current["trained_heads_changed"].values())
    suffix = "_reciprocal" if current["configuration"].get("audio_pair_mode") == "reciprocal-answers" else ""
    report = dict(point1=json.loads((ROOT / f"processed/v2/audio_contrast{suffix}.report.json").read_text()),
                  point2=json.load(open(current["configuration"]["feature_manifest"]))["extraction_counts"],
                  point3=dict(pair_updates=current["pair_updates"], optimizer_steps=current["last_step"],
                              pair_samples_seen=current["pair_samples_seen"], pairs_every_update=True),
                  point4=dict(configuration=current["configuration"], heads_changed=current["trained_heads_changed"],
                              encoder_calls=current["encoder_calls_during_training"]),
                  point5=dict(status="completed", epochs=current["configuration"]["epochs"],
                              best_step=current["best_step"], seconds=current["seconds"]),
                  point6=comparisons,
                  point7=dict(audio_gate_passed=current["audio_dependence_gate_passed"],
                              next_action="large_stages" if current["audio_dependence_gate_passed"] else "layer_signal_diagnosis"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = current["history"]
    steps = [x["step"] for x in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Frozen Samsone356M + shared adapter + both CLM heads trained\n"
        f"10K main + {current['auxiliary_train_samples']} auxiliary rows; main batch 64 + 32 verified pairs/update\n"
        "12 epochs; adapter LR 3e-4, heads LR 1e-4; pair CE weight 0.5, swap weight 1, cosine margin 0.05")
    for key, label in (("main_ce", "Main train CE"), ("pair_ce", "Verified-pair CE")):
        axes[0, 0].plot(steps, [x["objective_means"][key] for x in history], label=label)
    axes[0, 0].plot(steps, [x["validation"]["loss"] for x in history], label="Validation CE")
    axes[0, 0].set_ylabel("Cross entropy")
    for mode, label in (("correct", "Correct audio"), ("random", "Random audio"), ("silence", "Silence")):
        axes[0, 1].plot(steps, [100*(x["validation"]["top1"]-(0 if mode == "correct" else x["audio_top1_gaps"][mode]))
                               for x in history], label=label)
    axes[0, 1].set_ylabel("Validation Top-1 (%)")
    for mode in ("random", "silence"):
        axes[1, 0].plot(steps, [100*x["audio_top1_gaps"][mode] for x in history], label=f"Correct minus {mode}")
    axes[1, 0].axhline(5, color="black", linestyle="--", linewidth=1, label="Audio gate: 5 pp")
    axes[1, 0].set_ylabel("Accuracy gap (percentage points)")
    axes[1, 1].plot(steps, [x["objective_means"]["swap_loss"] for x in history], label="Verified-pair swap loss")
    axes[1, 1].set_ylabel("Unweighted cosine margin loss")
    for ax in axes.flat:
        ax.set_xlabel("Optimizer step")
        ax.grid(alpha=.2)
        ax.legend()
    plot = run_root / f"audio_contrast_curves{suffix}.png"
    fig.savefig(plot, dpi=160)
    plt.close(fig)
    report["training_curves"] = str(plot)
    write_json(run_root / f"audio_contrast_result{suffix}.json", report)
    state = json.loads((ROOT / "runs/pipeline_status.json").read_text())
    state.update(phase="audio_contrast_gate_passed" if current["audio_dependence_gate_passed"] else "audio_contrast_signal_diagnosis",
                 status="ready_for_large_stages" if current["audio_dependence_gate_passed"] else "diagnosing", active_pid=None,
                 updated_at=datetime.now(timezone.utc).isoformat(), contrast_result=str(run_root / f"audio_contrast_result{suffix}.json"))
    write_json(ROOT / "runs/pipeline_status.json", state)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
