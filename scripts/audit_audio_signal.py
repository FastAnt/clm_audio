"""Measure paired audio sensitivity through the existing frozen-feature/adapter/head path."""
import json

import torch
from torch.nn import functional as F

from samsone_clm.common import ROOT,write_json
from samsone_clm.model import AudioCLM
from samsone_clm.precompute import load_features


def comparison(a,b):
    delta=(a-b).norm(dim=-1)
    relative=delta/a.norm(dim=-1).clamp_min(1e-9)
    cosine=F.cosine_similarity(a,b,dim=-1)
    return dict(mean_l2=float(delta.mean()),mean_relative_l2=float(relative.mean()),
                median_relative_l2=float(relative.median()),mean_cosine=float(cosine.mean()),
                min_cosine=float(cosine.min()))


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    saved=torch.load(ROOT/"runs/stage2/best.pt",map_location="cpu",weights_only=True)
    data,manifest=load_features(saved["args"]["feature_manifest"],dtype=torch.bfloat16)
    ids=torch.tensor([i for i,r in enumerate(manifest["rows"]) if r["split"]=="validation" and data["ablation_available"][i]])
    model=AudioCLM(saved["spec"]["samsone_width"],bottleneck=saved["spec"]["bottleneck"])
    model.load_state_dict(saved["model_state"],strict=True)
    model.cuda().eval().requires_grad_(False)
    levels={mode:{key:[] for key in ("raw_hidden","adapter_output","state_z","scores")} for mode in ("correct","random","silence")}
    for chunk in ids.split(64):
        action=model.action(data["actions"][data["indices"][chunk]].cuda())
        for mode in levels:
            raw=data[mode][chunk].cuda().float()
            adapted=model.adapter(raw)
            state=F.normalize(model.state_head(adapted),dim=-1)
            scores=torch.einsum("bd,bkd->bk",state,action)
            for key,value in (("raw_hidden",raw),("adapter_output",adapted),("state_z",state),("scores",scores)):
                levels[mode][key].append(value.cpu())
    levels={mode:{key:torch.cat(value) for key,value in values.items()} for mode,values in levels.items()}
    labels=data["labels"][ids]
    reference=levels["correct"]
    correct_predictions=reference["scores"].argmax(-1)
    correct_probs=reference["scores"].softmax(-1)
    report=dict(samples=len(ids),checkpoint=str(ROOT/"runs/stage2/best.pt"),modes={})
    for mode in ("random","silence"):
        alternative=levels[mode]
        predictions=alternative["scores"].argmax(-1)
        probs=alternative["scores"].softmax(-1)
        report["modes"][mode]=dict(
            **{key:comparison(reference[key],alternative[key]) for key in ("raw_hidden","adapter_output","state_z")},
            correct_audio_accuracy=float(correct_predictions.eq(labels).float().mean()),
            altered_audio_accuracy=float(predictions.eq(labels).float().mean()),
            prediction_disagreements=int(correct_predictions.ne(predictions).sum()),
            correct_answers_lost=int((correct_predictions.eq(labels)&predictions.ne(labels)).sum()),
            new_correct_answers=int((correct_predictions.ne(labels)&predictions.eq(labels)).sum()),
            mean_absolute_probability_change=float((correct_probs-probs).abs().mean()),
            max_absolute_probability_change=float((correct_probs-probs).abs().max()))
    write_json(ROOT/"runs/samsone_native_audit_cuda/stage2_audio_signal.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
