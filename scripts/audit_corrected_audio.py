"""Locate audio-signal attenuation, using each checkpoint's actual adapter/head contract."""
import argparse
import json

import torch
from torch.nn import functional as F

from samsone_clm.common import write_json
from samsone_clm.contrast import supervised_audio_partners, contrast_key
from samsone_clm.model import AudioCLM
from samsone_clm.precompute import load_features


def distance(a, b):
    return dict(mean_l2=float((a-b).norm(dim=-1).mean()),
                mean_relative_l2=float(((a-b).norm(dim=-1)/a.norm(dim=-1).clamp_min(1e-9)).mean()),
                mean_cosine=float(F.cosine_similarity(a, b, dim=-1).mean()))


@torch.no_grad()
def audit(paths, manifest_path, output, pair_mode="same-options"):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    data, manifest = load_features(manifest_path, dtype=torch.bfloat16)
    data = {k:v.cuda() for k,v in data.items()}
    rows = manifest["rows"]
    val = torch.tensor([i for i,r in enumerate(rows) if r["split"] == "validation"], device="cuda")
    train = [i for i,r in enumerate(rows) if r["split"] == "train"]
    partners = supervised_audio_partners(rows, train, pair_mode).cuda()
    paired = (partners >= 0).nonzero().flatten()
    pair_keys = {contrast_key(rows[i]) for i in paired.tolist()}
    covered = torch.tensor([contrast_key(rows[i]) in pair_keys for i in val.tolist()])
    report = dict(validation_samples=len(val), verified_train_pair_rows=len(paired),
                  train_contrast_groups=len(pair_keys),
                  validation_rows_with_trained_contrast_template=int(covered.sum()), checkpoints={})
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        model = AudioCLM.from_spec(ck["spec"]).cuda().eval().requires_grad_(False)
        model.load_state_dict(ck["model_state"], strict=True)
        levels = {mode:{key:[] for key in ("hidden", "adapter_raw", "head_input", "state_z", "scores")}
                  for mode in ("correct", "random", "silence")}
        for chunk in val.split(64):
            actions = data["actions"][data["indices"][chunk]]
            for mode in levels:
                raw = data[mode][chunk].float()
                adapted = model.adapter(raw)
                normalized = model._adapt(raw)
                state = model.state(raw)
                scores = model(raw, actions)
                for key,value in (("hidden",raw), ("adapter_raw",adapted), ("head_input",normalized),
                                  ("state_z",state), ("scores",scores)):
                    levels[mode][key].append(value.cpu())
        levels = {m:{k:torch.cat(v) for k,v in d.items()} for m,d in levels.items()}
        reference = levels["correct"]
        correct = reference["scores"].argmax(-1).eq(data["labels"][val].cpu())
        result = dict(spec=ck["spec"], validation={})
        result["template_coverage"] = {}
        for label,mask in (("covered",covered), ("not_covered",~covered)):
            if not mask.any():
                result["template_coverage"][label] = dict(samples=0)
                continue
            labels = data["labels"][val].cpu()[mask]
            accuracy = {mode:float(levels[mode]["scores"][mask].argmax(-1).eq(labels).float().mean())
                        for mode in levels}
            result["template_coverage"][label] = dict(samples=int(mask.sum()), top1=accuracy,
                    gaps={m:accuracy["correct"]-accuracy[m] for m in ("random","silence")})
        for mode in ("random", "silence"):
            changed = levels[mode]
            alternative = changed["scores"].argmax(-1).eq(data["labels"][val].cpu())
            result["validation"][mode] = dict(
                levels={key:distance(reference[key],changed[key])
                        for key in ("hidden", "adapter_raw", "head_input", "state_z")},
                correct_accuracy=float(correct.float().mean()), altered_accuracy=float(alternative.float().mean()),
                predictions_changed=int(reference["scores"].argmax(-1).ne(changed["scores"].argmax(-1)).sum()),
                correct_answers_lost=int((correct & ~alternative).sum()),
                correct_answers_gained=int((~correct & alternative).sum()),
                mean_absolute_probability_change=float((reference["scores"].softmax(-1)-changed["scores"].softmax(-1)).abs().mean()))
        gaps, correct_pair, swapped_pair = [], [], []
        for chunk in paired.split(64):
            actions = data["actions"][data["indices"][chunk]]
            own = model(data["correct"][chunk], actions)
            swap = model(data["correct"][partners[chunk]], actions)
            labels = data["labels"][chunk, None]
            gaps.append(((own.gather(1,labels)-swap.gather(1,labels))/model.scale).cpu().flatten())
            correct_pair.append(own.argmax(-1).eq(labels.flatten()).cpu())
            swapped_pair.append(swap.argmax(-1).eq(labels.flatten()).cpu())
        gap = torch.cat(gaps)
        result["verified_training_pairs"] = dict(samples=len(gap), correct_accuracy=float(torch.cat(correct_pair).float().mean()),
                swapped_original_label_accuracy=float(torch.cat(swapped_pair).float().mean()),
                mean_correct_choice_cosine_gap=float(gap.mean()), fraction_margin_passed=float((gap >= .05).float().mean()))
        report["checkpoints"][str(path)] = result
        del model
    write_json(output, report)
    print(json.dumps({p:dict(pairs=r["verified_training_pairs"],
                            validation={m:dict(correct=v["correct_accuracy"], changed=v["altered_accuracy"],
                                hidden_relative_l2=v["levels"]["hidden"]["mean_relative_l2"],
                                adapter_relative_l2=v["levels"]["adapter_raw"]["mean_relative_l2"],
                                state_z_relative_l2=v["levels"]["state_z"]["mean_relative_l2"])
                                        for m,v in r["validation"].items()})
                      for p,r in report["checkpoints"].items()}, indent=2))


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-mode", choices=("same-options", "reciprocal-answers"), default="same-options")
    args = parser.parse_args()
    audit(args.checkpoints, args.feature_manifest, args.output, args.pair_mode)
