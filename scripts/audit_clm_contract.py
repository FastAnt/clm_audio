"""Read-only CLM head-input and scale audit on the existing Stage 2 checkpoint.

Post-hoc normalization/scaling is a diagnostic, not retraining or a corrected model.
"""
import json
import math

import torch
from torch.nn import functional as F

from samsone_clm.common import ROOT, write_json
from samsone_clm.metrics import metrics
from samsone_clm.model import AudioCLM
from samsone_clm.precompute import load_features


def input_stats(head, value):
    pre = head.inp(value)
    derivative = 0.5 * (1 + torch.erf(pre / math.sqrt(2))) + pre * torch.exp(-pre.square() / 2) / math.sqrt(2 * math.pi)
    return dict(mean_input_norm=float(value.norm(dim=-1).mean()),
                mean_abs_preactivation=float(pre.abs().mean()),
                fraction_below_minus_3=float((pre < -3).float().mean()),
                fraction_above_plus_3=float((pre > 3).float().mean()),
                fraction_abs_gelu_derivative_below_001=float((derivative.abs() < 0.01).float().mean()))


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True  # Match the existing training/evaluation recipe.
    saved = torch.load(ROOT / "runs/stage2/best.pt", map_location="cpu", weights_only=True)
    original = torch.load(ROOT / "models/clm/CLM_v0.1-8B.pt", map_location="cpu", weights_only=True)
    scale = float(torch.as_tensor(original["logit_scale"]).exp().clamp(max=100))
    data, manifest = load_features(saved["args"]["feature_manifest"], dtype=torch.bfloat16)
    ids = torch.tensor([i for i, row in enumerate(manifest["rows"])
                        if row["split"] == "validation" and data["ablation_available"][i]])
    model = AudioCLM(saved["spec"]["samsone_width"], bottleneck=saved["spec"]["bottleneck"])
    initial_model = AudioCLM(saved["spec"]["samsone_width"], bottleneck=saved["spec"]["bottleneck"])
    model.load_state_dict(saved["model_state"], strict=True)
    model.cuda().eval().requires_grad_(False)
    initial_model.cuda().eval().requires_grad_(False)
    report = dict(checkpoint=str(ROOT / "runs/stage2/best.pt"), validation_samples=len(ids),
                  original_logit_scale=float(torch.as_tensor(original["logit_scale"])), original_scale=scale,
                  legacy_scale=1., allow_tf32=True, statistics={}, posthoc_evaluations={},
                  warning="Post-hoc changes on legacy-trained weights do not establish results of corrected training")
    first = ids[:512]
    for branch, raw in (("state", data["correct"][first]),
                        ("action", data["actions"][data["indices"][first]].reshape(-1, model.spec["samsone_width"]))):
        adapted = model.adapter(raw.cuda().float())
        for weights, source in (("trained", model), ("original", initial_model)):
            head = getattr(source, f"{branch}_head")
            report["statistics"][f"{weights}_{branch}"] = {
                "legacy": input_stats(head, adapted),
                "unit_l2": input_stats(head, F.normalize(adapted, dim=-1)),
            }
    labels = data["labels"][ids]
    for normalized in (False, True):
        all_scores = {mode: [] for mode in ("correct", "random", "silence")}
        for chunk in ids.split(64):
            actions = model.adapter(data["actions"][data["indices"][chunk]].cuda().float())
            if normalized:
                actions = F.normalize(actions, dim=-1)
            za = F.normalize(model.action_head(actions), dim=-1)
            for mode in all_scores:
                states = model.adapter(data[mode][chunk].cuda().float())
                if normalized:
                    states = F.normalize(states, dim=-1)
                zs = F.normalize(model.state_head(states), dim=-1)
                all_scores[mode].append(torch.einsum("bd,bkd->bk", zs, za).cpu())
        all_scores = {mode: torch.cat(values) for mode, values in all_scores.items()}
        for current_scale in (1., 20., scale):
            key = f"l2_{normalized}_scale_{current_scale:g}"
            report["posthoc_evaluations"][key] = {mode: metrics(scores * current_scale, labels)
                                                     for mode, scores in all_scores.items()}
    write_json(ROOT / "runs/review_clm_contract.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
