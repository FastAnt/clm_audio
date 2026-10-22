from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .common import ROOT, digest, write_json,commit_file
from .metrics import evaluate_model, metrics
from .model import AudioCLM
from .precompute import load_features
from .objectives import bidirectional_choice_loss,canonical_action_ids
from .resources import require_resources


@torch.no_grad()
def train_metrics(model, data, indices, batch_size):
    scores = torch.cat([model(data["correct"][part], data["actions"][data["indices"][part]], data['candidate_mask'][part])
                        for part in indices.split(batch_size)])
    return metrics(scores, data["labels"][indices])


def train(args):
    require_resources()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    data, manifest = load_features(args.feature_manifest)
    device = torch.device(args.device)
    data = {key: tensor.to(device) for key, tensor in data.items()}
    if args.loss=='clm-infonce':
        alias=canonical_action_ids(manifest['choices'],device)
        data['contrast_indices']=alias[data['indices']]
    rows = manifest["rows"]
    splits = {split: torch.tensor([i for i, r in enumerate(rows) if r["split"] == split], device=device)
              for split in ("train", "validation", "test")}
    if not args.overfit:
        requested = {1: 10000, 2: 50000, 3: 150000, 4: len(splits["train"])}[args.stage]
        if len(splits["train"]) < requested:
            raise ValueError(f"Spike contains only {len(splits['train'])} train samples; Stage {args.stage} requires {requested}.")
        raise ValueError("This implementation spike exposes overfit only; full staged training is not enabled.")
    if args.stage != 1:
        raise ValueError("Overfit gate uses Stage 1: adapter train, both original heads frozen")
    indices = splits["train"][:args.samples]
    if not 256 <= len(indices) <= 1000 or len(indices) != args.samples:
        raise ValueError("Overfit requires 256–1000 existing TRAIN samples")
    model = AudioCLM(data["correct"].shape[-1], bottleneck=args.bottleneck,
                     adapter_kind=args.adapter_kind, learnable_scale=args.learnable_scale).to(device)
    model.set_stage(args.stage)
    initial_heads = {name: {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
                     for name, head in (("state", model.state_head), ("action", model.action_head))}
    run = Path(args.out or ROOT / f"runs/overfit-{args.samples}-seed{args.seed}-lr{args.lr:g}")
    run.mkdir(parents=True, exist_ok=True)
    initial = train_metrics(model, data, indices, args.batch_size)
    opt = torch.optim.AdamW(model.adapter.parameters(), lr=args.lr, weight_decay=0)
    history = [{"step": 0, **initial}]
    print(json.dumps({"initial": initial, "model": model.spec, "run": str(run)}), flush=True)
    best_loss = initial["loss"]
    first_grad_norm = None
    started = time.monotonic()

    def save(name, step, current):
        checkpoint = dict(model_state=model.state_dict(), spec=model.spec, step=step, metrics=current,
                          feature_provenance=manifest["provenance"], feature_manifest=digest(manifest),
                          optimizer=opt.state_dict(), rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all(), stage=args.stage, overfit=True, args=vars(args))
        temporary = run / (name + ".tmp")
        torch.save(checkpoint, temporary)
        commit_file(temporary,run/name)

    save("best.pt", 0, initial)
    for step in range(1, args.steps + 1):
        if step % 100 == 0:
            require_resources()
        positions = torch.randperm(len(indices), device=device)[:args.batch_size]
        selected = indices[positions]
        scores = model(data["correct"][selected], data["actions"][data["indices"][selected]], data['candidate_mask'][selected])
        loss = F.cross_entropy(scores, data["labels"][selected])
        if args.loss in ('infonce','clm-infonce'):
            contrast, _ = bidirectional_choice_loss(model,data['correct'][selected],data['actions'],
                                                    data.get('contrast_indices',data['indices'])[selected],data['labels'][selected],
                                                    mask_unknown=args.loss!='clm-infonce')
            loss = contrast if args.loss=='clm-infonce' else loss + args.infonce_weight * contrast
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.0)
        if first_grad_norm is None:
            first_grad_norm = float(gradient_norm)
            assert first_grad_norm > 0 and all(p.grad is None for h in (model.state_head, model.action_head) for p in h.parameters())
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise ValueError("Nonfinite loss/gradients")
        opt.step()
        if step % args.evaluate_every == 0 or step == args.steps:
            current = train_metrics(model, data, indices, args.batch_size)
            current.update(step=step, elapsed_s=time.monotonic()-started)
            history.append(current)
            print(json.dumps(current), flush=True)
            if current["loss"] < best_loss:
                best_loss = current["loss"]
                save("best.pt", step, current)
            save("last.pt", step, current)
            write_json(run / "metrics.json", {"history": history, "initial": initial, "best_loss": best_loss})
            write_json(run / "progress.json", dict(status="running",stage=1,step=step,target_steps=args.steps,
                       train_samples=len(indices),train_curve=[dict(step=x["step"],loss=x["loss"]) for x in history],
                       configuration=vars(args),model=model.spec,encoder_calls_during_training=0))
            if current["top1"] >= 0.99 and current["loss"] < min(0.2, initial["loss"] * 0.75):
                break
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    frozen_unchanged = {name: all(torch.equal(value.detach().cpu(), initial_heads[name][key])
                                for key, value in head.state_dict().items())
                        for name, head in (("state", model.state_head), ("action", model.action_head))}
    evaluations = {split: evaluate_model(model, data, rows, ids, args.batch_size)
                   for split, ids in {"overfit_train": indices, "validation": splits["validation"]}.items() if len(ids)}
    final = evaluations["overfit_train"]["correct"]["overall"]
    report = dict(initial=initial, final=final, history=history, evaluations=evaluations,
                  best_step=checkpoint["step"], seconds=time.monotonic()-started, model=model.spec,
                  configuration=vars(args), feature_provenance=manifest["provenance"],
                  feature_manifest=digest(manifest), feature_directory=manifest["directory"],
                  split_samples={split: len(ids) for split, ids in splits.items()},
                  overfit_gate_passed=final["top1"] >= 0.95 and final["loss"] < min(0.2, initial["loss"] * 0.75),
                  frozen_heads_unchanged=frozen_unchanged, first_adapter_gradient_norm=first_grad_norm,
                  encoder_calls_during_training=0, scope="Diagnostic overfit on AudioMCQ; no generalization claim",
                  status="completed")
    validation_gaps = evaluations.get("validation", {}).get("audio_top1_gaps", {})
    report["audio_dependence_gate_passed"] = bool(validation_gaps) and all(gap >= 0.05 for gap in validation_gaps.values())
    report["ready_for_large_training"] = report["overfit_gate_passed"] and report["audio_dependence_gate_passed"]
    assert all(frozen_unchanged.values())
    write_json(run / "metrics.json", report)
    write_json(run / "progress.json", dict(status="completed",stage=1,step=checkpoint["step"],
               target_steps=args.steps,train_samples=len(indices),configuration=vars(args),model=model.spec,
               train_curve=[dict(step=x["step"],loss=x["loss"]) for x in history],encoder_calls_during_training=0))
    print(json.dumps({"final": final, "gate": report["overfit_gate_passed"], "audio_gaps": evaluations["overfit_train"]["audio_top1_gaps"],
                      "frozen_heads_unchanged": frozen_unchanged, "run": str(run)}), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--evaluate-every", type=int, default=0)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--bottleneck", type=int)
    ap.add_argument("--adapter-kind", choices=("mlp", "linear"), default="mlp")
    ap.add_argument("--run-root", default=str(ROOT/"runs/v2"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    ap.add_argument("--feature-manifest", help="Explicit immutable feature manifest; defaults to current dataset pointer")
    ap.add_argument("--diagnostic", action="store_true", help="Explicit smaller diagnostic run; does not claim the prescribed stage sample count")
    ap.add_argument("--init-checkpoint")
    ap.add_argument("--continue-checkpoint", help="Additional same-stage raw weights; new optimizer/schedule and output directory; music feature transfer requires its explicit flag")
    ap.add_argument('--allow-feature-transfer',action='store_true',help='Explicit music-only same-stage continuation on new features; encoder and held-out checks remain strict')
    ap.add_argument("--resume", nargs="?", const="auto")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--loss", choices=("ce", "infonce", "clm-infonce"), default="ce",
                    help="ce; CE+masked bidirectional contrast; or the official CLM choice InfoNCE objective")
    ap.add_argument("--infonce-weight", type=float, default=1.)
    ap.add_argument("--learnable-scale", action="store_true")
    ap.add_argument("--scale-lr", type=float, default=.001)
    ap.add_argument("--weight-decay", type=float, default=0.)
    ap.add_argument("--scheduler", choices=("constant", "onecycle"), default="constant")
    ap.add_argument("--patience", type=int, default=0, help="Early stop after this many evaluations without validation improvement")
    ap.add_argument('--selection-metric',choices=('accuracy','audio-grounded','music-grounded','music-window-grounded','genre-grounded'),default='accuracy')
    ap.add_argument('--balance-genre-ce',action='store_true',help='Inverse-square-root semantic class weighting for new genre data')
    ap.add_argument("--head-lr", type=float, default=0.0001)
    ap.add_argument("--audio-swap-weight", type=float, default=0.)
    ap.add_argument("--audio-swap-margin", type=float, default=.05)
    ap.add_argument("--audio-pair-batch-size", type=int, default=0,
                    help="Verified audio pairs per update; both endpoints receive CE and swap loss")
    ap.add_argument("--audio-pair-ce-weight", type=float, default=.5)
    ap.add_argument("--audio-pair-mode", choices=("same-options", "reciprocal-answers", "reciprocal-diverse"), default="same-options")
    args = ap.parse_args()
    if args.overfit:
        args.lr = args.lr or .001
        args.steps = args.steps or 2000
        args.evaluate_every = args.evaluate_every or 50
        if not args.out:
            args.out = str(Path(args.run_root)/f"overfit-{args.samples}-{args.adapter_kind}")
        train(args)
    else:
        args.lr = args.lr or .0001
        from .staged_train import train_staged
        train_staged(args)


if __name__ == "__main__":
    main()
