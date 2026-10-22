from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .common import ROOT, digest, write_json,commit_file
from .metrics import evaluate_model
from .model import AudioCLM
from .precompute import load_features
from .contrast import supervised_audio_partners, supervised_swap_loss, sample_audio_pair_batch
from .objectives import bidirectional_choice_loss,canonical_action_ids
from .resources import require_resources

STAGE_SAMPLES = {1: 10000, 2: 50000, 3: 150000}
ENCODER_FIELDS = ("recipe", "samsone_sha256", "hidden_width", "dtype", "sample_rate", "max_audio_samples", "compute_dtype", "position_recipe")


def head_signature(state):
    checksum = hashlib.sha256()
    for key, value in sorted(state.items()):
        checksum.update(key.encode())
        checksum.update(str((value.dtype, tuple(value.shape))).encode())
        checksum.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def parameter_groups(model, adapter_lr, head_lr, scale_lr=.001):
    groups = [{"params": list(model.adapter.parameters()), "lr": adapter_lr}]
    for head in (model.state_head, model.action_head):
        parameters = [p for p in head.parameters() if p.requires_grad]
        if parameters:
            groups.append({"params": parameters, "lr": head_lr})
    if model.logit_scale is not None and model.logit_scale.requires_grad:
        groups.append({"params": [model.logit_scale], "lr": scale_lr, "weight_decay": 0.})
    return groups


def validate_initialization(checkpoint, manifest, stage, resume=False, continuation=False, transfer_features=False):
    if resume and continuation:
        raise ValueError("Resume and continuation are mutually exclusive")
    if transfer_features and (not continuation or stage<5 or not manifest.get('music_only')):
        raise ValueError('Feature transfer requires explicit same-stage musical continuation')
    for field in ENCODER_FIELDS:
        if checkpoint["feature_provenance"].get(field) != manifest["provenance"].get(field):
            raise ValueError(f"Frozen encoder mismatch: {field}")
    if resume or continuation:
        if checkpoint["stage"] != stage or (checkpoint["feature_manifest"] != digest(manifest) and not transfer_features):
            raise ValueError("Resume/continuation requires the same stage and exact feature manifest")
    elif checkpoint["stage"] != stage - 1:
        raise ValueError("Each stage must initialize from the previous stage checkpoint")


def train_staged(args):
    continuation = getattr(args, 'continue_checkpoint', None)
    if sum(bool(value) for value in (args.resume, args.init_checkpoint, continuation)) > 1:
        raise ValueError("Choose exactly one of resume, initialization, or continuation")
    if continuation and not ((args.stage==4 and not args.diagnostic) or
                             (args.stage in (5,6) and getattr(args,'allow_feature_transfer',False))):
        raise ValueError("Same-stage continuation requires full Stage4 or explicit musical feature transfer")
    require_resources()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Keep precomputed representations in their stored precision; adapter casts each selected batch to FP32.
    data, manifest = load_features(args.feature_manifest, dtype=torch.bfloat16)
    rows = manifest["rows"]
    manifest_hash = digest(manifest)
    device = torch.device(args.device)
    data = {key: value.to(device) for key, value in data.items()}
    if args.loss=='clm-infonce':
        alias=canonical_action_ids(manifest['choices'],device)
        data['contrast_indices']=alias[data['indices']]
    splits = {s: torch.tensor([i for i, r in enumerate(rows) if r["split"] == s], device=device)
              for s in ("train", "validation", "test")}
    main_pool = [i for i in splits["train"].tolist() if rows[i].get("training_role") != "contrast_only"]
    auxiliary = [i for i in splits["train"].tolist() if rows[i].get("training_role") == "contrast_only"]
    requested = args.samples if args.diagnostic else STAGE_SAMPLES.get(args.stage, len(main_pool))
    if requested > len(main_pool):
        raise ValueError(f"Stage {args.stage} needs {requested} main train samples, available {len(main_pool)}")
    if not len(splits["validation"]):
        raise ValueError("Validation is required for checkpoint selection")
    args.samples = requested
    # Stable hash order keeps stages nested; dataset metadata contains audio-disjoint splits.
    selected = sorted(main_pool, key=lambda i: digest([args.seed, rows[i]["sample_id"]]))[:requested]
    indices = torch.tensor(selected, device=device)
    trained_indices = selected + auxiliary
    partners = supervised_audio_partners(rows, trained_indices, args.audio_pair_mode).to(device)
    data["partners"] = partners
    paired_rows = int((partners >= 0).sum())
    paired_indices = (partners >= 0).nonzero().flatten()
    if args.audio_pair_batch_size < 0 or args.audio_pair_ce_weight < 0:
        raise ValueError("Audio pair batch size and CE weight must be nonnegative")
    if auxiliary and not args.audio_pair_batch_size:
        raise ValueError("Auxiliary-only training rows require dedicated audio pair batches")
    if args.audio_pair_batch_size and args.audio_swap_weight <= 0:
        raise ValueError("Dedicated audio pair training requires positive swap weight")
    if args.audio_swap_weight > 0 and not paired_rows:
        raise ValueError("Audio-swap objective requires verified same-question/options, different-answer training pairs")
    genre_ce_weights=torch.ones(len(rows),device=device)
    if getattr(args,'balance_genre_ce',False):
        from collections import Counter
        genre_ids=[i for i in indices.tolist() if rows[i].get('genre_corpus')]
        counts=Counter(rows[i].get('semantic_label',rows[i]['choices'][rows[i]['answer_index']]) for i in genre_ids)
        if not counts:raise ValueError('Genre balancing requires explicit genre_corpus training rows')
        for i in genre_ids:
            label=rows[i].get('semantic_label',rows[i]['choices'][rows[i]['answer_index']])
            genre_ce_weights[i]=counts[label]**-.5
        if genre_ids:
            genre_ce_weights[genre_ids]/=genre_ce_weights[genre_ids].mean()
    run_root = Path(args.run_root)
    run = Path(args.out or run_root / f"{'diagnostic-' if args.diagnostic else ''}stage{args.stage}")
    if continuation and any((run / name).exists() for name in ('best.pt', 'last.pt')):
        raise ValueError("Continuation requires a new output directory; use --resume for an existing run")
    run.mkdir(parents=True, exist_ok=True)
    resume_path = run / "last.pt" if args.resume == "auto" else Path(args.resume) if args.resume else None
    previous = run_root / f"stage{args.stage-1}/best.pt"
    init_path = Path(continuation or args.init_checkpoint) if continuation or args.init_checkpoint else previous if args.stage > 1 else None
    saved = torch.load(resume_path or init_path, map_location="cpu", weights_only=True) if resume_path or init_path else None
    if saved:
        validate_initialization(saved, manifest, args.stage, resume=resume_path is not None,
                                continuation=bool(continuation),transfer_features=getattr(args,'allow_feature_transfer',False))
        if resume_path:
            for field, default in (("audio_pair_batch_size", 0), ("audio_pair_ce_weight", .5),
                                   ("audio_pair_mode", "same-options"),
                                   ("audio_swap_weight", 0.), ("audio_swap_margin", .05),
                                   ("loss", "ce"), ("infonce_weight", 1.), ("learnable_scale", False),
                                   ("scale_lr", .001), ("weight_decay", 0.), ("scheduler", "constant"),
                                   ("epochs", 8), ("batch_size", 64), ("patience", 0), ('selection_metric','accuracy'),('balance_genre_ce',False)):
                if saved["args"].get(field, default) != getattr(args, field):
                    raise ValueError(f"Resume requires the same objective: {field}")
        heldout = {r["audio_identity"] for r in rows if r["split"] != "train"}
        heldout_pcm = {r.get("pcm_identity", r["audio_identity"]) for r in rows if r["split"] != "train"}
        if heldout.intersection(saved.get("train_audio_identities", [])) or heldout_pcm.intersection(saved.get("train_pcm_identities", [])):
            raise ValueError("New held-out corpus overlaps the previous stage training audio")
    inherited_audio = set(saved.get('train_audio_identities', [])) if saved else set()
    inherited_pcm = set(saved.get('train_pcm_identities', [])) if saved else set()
    if saved and (saved["spec"].get("contract_version") != 2 or not saved["spec"].get("head_input_l2")):
        raise ValueError("Corrected training must not initialize/resume from a legacy head contract")
    model = (AudioCLM.from_spec(saved["spec"]) if saved else
             AudioCLM(manifest["provenance"]["hidden_width"], bottleneck=args.bottleneck,
                      adapter_kind=args.adapter_kind)).to(device)
    baseline = saved.get("head_initial_signatures") if resume_path else None
    if resume_path and baseline is None:
        baseline = {name:head_signature(head.state_dict()) for name, head in (("state",model.state_head),("action",model.action_head))}
    if saved:
        model.load_state_dict(saved["model_state"], strict=True)
    if args.learnable_scale:
        model.enable_learnable_scale()
    model.set_stage(args.stage)
    model.constrain_scale()
    heads_before = {name: {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                    for name, head in (("state", model.state_head), ("action", model.action_head))}
    baseline = baseline or {name:head_signature(state) for name,state in heads_before.items()}
    del heads_before
    optimizer = torch.optim.AdamW(parameter_groups(model, args.lr, args.head_lr, args.scale_lr), weight_decay=args.weight_decay)
    total_steps = min(args.steps or 10**15,args.epochs*((len(indices)+args.batch_size-1)//args.batch_size))
    scheduler = (torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=[g['lr'] for g in optimizer.param_groups],
                    total_steps=total_steps,pct_start=.1,anneal_strategy='cos') if args.scheduler=='onecycle' else None)
    trainable = [p for p in model.parameters() if p.requires_grad]
    step, epoch, offset, order = 0, 0, 0, None
    pair_updates, pair_samples_seen = 0, 0
    history, gradients = [], {}
    stale_evaluations = 0
    started = time.monotonic()
    initial = evaluate_model(model, data, rows, splits["validation"], args.batch_size)
    best_key = (float("-inf"), float("-inf"))  # Select among trained checkpoints; baseline is reported separately.
    if resume_path:
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler:
            scheduler.load_state_dict(saved['scheduler'])
        stale_evaluations = saved.get('stale_evaluations',0)
        step, epoch, offset = saved["step"], saved["epoch"], saved["offset"]
        order = saved["epoch_order"].to(device) if saved["epoch_order"] is not None else None
        history, best_key = saved["history"], tuple(saved["best_key"])
        gradients = saved.get("first_gradients", {})
        initial = saved.get("initial_validation", initial)
        pair_updates = saved.get("pair_updates", 0)
        pair_samples_seen = saved.get("pair_samples_seen", 0)
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    del saved

    def save(name, evaluation):
        payload = dict(model_state=model.state_dict(), spec=model.spec, stage=args.stage, overfit=False,
                       diagnostic=args.diagnostic, step=step, epoch=epoch, offset=offset,
                       epoch_order=order.cpu() if order is not None else None,
                       optimizer=optimizer.state_dict(), rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                       scheduler=scheduler.state_dict() if scheduler else None, stale_evaluations=stale_evaluations,
                       feature_provenance=manifest["provenance"], feature_manifest=manifest_hash,
                       metrics=evaluation["correct"]["overall"], history=history, best_key=list(best_key), args=vars(args),
                       train_sample_ids=[rows[i]["sample_id"] for i in trained_indices],
                       main_train_sample_ids=[rows[i]["sample_id"] for i in selected],
                       train_audio_identities=sorted(inherited_audio | {rows[i]["audio_identity"] for i in trained_indices}),
                       train_pcm_identities=sorted(inherited_pcm | {rows[i].get("pcm_identity", rows[i]["audio_identity"]) for i in trained_indices}),
                       pair_updates=pair_updates, pair_samples_seen=pair_samples_seen)
        payload.update(head_initial_signatures=baseline, first_gradients=gradients, initial_validation=initial)
        temporary = run / (name + ".tmp")
        torch.save(payload, temporary)
        commit_file(temporary,run/name)
        write_json(run/(name+".info.json"), dict(stage=args.stage, step=step, epoch=epoch, samples=len(indices), checkpoint=str(run/name)))

    if not resume_path:
        save("best.pt", initial)
    print(json.dumps(dict(event="training_started", stage=args.stage, diagnostic=args.diagnostic,
                          samples=len(indices), initialized_from=str(init_path), resumed_from=str(resume_path),
                          continuation=bool(continuation),
                          trainable_parameters=sum(p.numel() for p in trainable),
                          trainable_modules=["adapter"] + (["state_head", "action_head"] if args.stage >= 2 else []),
                          encoder_calls_during_training=0, initial_validation=initial["correct"]["overall"], run=str(run))), flush=True)
    interval = args.evaluate_every or max(100, min(1000, (len(indices) + args.batch_size - 1) // args.batch_size))
    recent_loss, recent_count = 0., 0
    objective_totals = dict(main_ce=0., infonce=0., pair_ce=0., swap_loss=0.)
    latest = initial
    progress_path = run/"progress.json"
    train_curve = []
    if resume_path and progress_path.exists():
        train_curve = [point for point in json.loads(progress_path.read_text()).get("train_curve", []) if point["step"] <= step]
    progress_at = 0.
    resumed_samples = epoch*len(indices)+offset

    def progress(status="running", batch_loss=None):
        elapsed = time.monotonic()-started
        write_json(progress_path, dict(status=status, stage=args.stage, step=step, epoch=epoch,
                   epochs=args.epochs, offset=offset, target_steps=min(args.steps or 10**15, args.epochs*((len(indices)+args.batch_size-1)//args.batch_size)),
                   train_samples=len(indices), samples_seen=epoch*len(indices)+offset, elapsed_s=elapsed,
                   samples_per_second=max(0,epoch*len(indices)+offset-resumed_samples)/max(elapsed,1e-9),
                   batch_loss=batch_loss, train_curve=train_curve, configuration=vars(args),
                   first_gradients=gradients, trainable_parameters=sum(p.numel() for p in trainable),
                   evaluation_every_steps=interval, optimizer=dict(name="AdamW", betas=list(optimizer.defaults["betas"]),
                       eps=optimizer.defaults["eps"], weight_decay=optimizer.defaults["weight_decay"], max_grad_norm=1.),
                   trainable_modules=["adapter"]+(["state_head","action_head"] if args.stage>=2 else []),
                   model=model.spec, encoder_calls_during_training=0,
                   learning_rates=[g['lr'] for g in optimizer.param_groups], score_scale=float(model.scale),
                   auxiliary_train_samples=len(auxiliary), supervised_audio_pair_rows=paired_rows,
                   pair_updates=pair_updates, pair_samples_seen=pair_samples_seen))

    progress()
    write_json(run/"metrics.json", dict(status="running", stage=args.stage, train_samples=len(indices),
               history=history, initial_validation=initial, configuration=vars(args), first_gradients=gradients, model=model.spec))
    while epoch < args.epochs and (not args.steps or step < args.steps):
        if step % 100 == 0:
            require_resources()
        if order is None:
            order = indices[torch.randperm(len(indices), device=device)]
            offset = 0
        batch = order[offset:offset + args.batch_size]
        model.train()
        model.set_stage(args.stage)
        scores = model(data["correct"][batch], data["actions"][data["indices"][batch]], data['candidate_mask'][batch])
        main_ce = (F.cross_entropy(scores, data["labels"][batch],reduction="none") * genre_ce_weights[batch]).mean()
        loss = main_ce
        infonce = scores.new_zeros(())
        if args.loss in ('infonce','clm-infonce'):
            infonce, _ = bidirectional_choice_loss(model,data['correct'][batch],data['actions'],
                                                  data.get('contrast_indices',data['indices'])[batch],data['labels'][batch],
                                                  mask_unknown=args.loss!='clm-infonce')
            loss = infonce if args.loss=='clm-infonce' else loss + args.infonce_weight * infonce
        pair_ce = scores.new_zeros(())
        swap_loss = scores.new_zeros(())
        if args.audio_swap_weight > 0:
            if args.audio_pair_batch_size:
                paired = sample_audio_pair_batch(partners, paired_indices, args.audio_pair_batch_size)
                own_scores = model(data["correct"][paired], data["actions"][data["indices"][paired]], data['candidate_mask'][paired])
                pair_ce = F.cross_entropy(own_scores, data["labels"][paired])
                loss = loss + args.audio_pair_ce_weight * pair_ce
            else:
                paired = batch[partners[batch] >= 0]
                own_scores = scores[partners[batch] >= 0]
            if len(paired):
                swap_loss = supervised_swap_loss(
                    model, data, paired, own_scores, args.audio_swap_margin)
                loss = loss + args.audio_swap_weight * swap_loss
                pair_updates += 1
                pair_samples_seen += len(paired)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise ValueError("Nonfinite training loss or gradients")
        if not gradients:
            for name, module in (("adapter", model.adapter), ("state_head", model.state_head), ("action_head", model.action_head)):
                gradients[name] = sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None)
            if gradients["adapter"] <= 0 or (args.stage >= 2 and min(gradients["state_head"], gradients["action_head"]) <= 0):
                raise ValueError("Expected trainable module has no gradient")
            print(json.dumps(dict(event="first_gradients", norms=gradients)), flush=True)
        optimizer.step()
        model.constrain_scale()
        if scheduler:
            scheduler.step()
        step += 1
        offset += len(batch)
        recent_loss += float(loss.detach()) * len(batch)
        for key, value in (("main_ce", main_ce), ("infonce", infonce), ("pair_ce", pair_ce), ("swap_loss", swap_loss)):
            objective_totals[key] += float(value.detach()) * len(batch)
        recent_count += len(batch)
        finished_epoch = offset >= len(order)
        if finished_epoch:
            epoch += 1
            offset, order = 0, None
        now = time.monotonic()
        if now-progress_at >= 5 or finished_epoch:
            train_curve.append(dict(step=step, epoch=epoch, loss=recent_loss/recent_count, elapsed_s=now-started))
            progress(batch_loss=float(loss.detach()))
            progress_at = now
        if step % interval == 0 or (finished_epoch and (not args.diagnostic or epoch >= args.epochs)) or (args.steps and step >= args.steps):
            model.eval()
            latest = evaluate_model(model, data, rows, splits["validation"], args.batch_size)
            current = latest["correct"]["overall"]
            entry = dict(step=step, epoch=epoch, train_loss=recent_loss / recent_count,
                         validation=current, audio_top1_gaps=latest["audio_top1_gaps"], elapsed_s=time.monotonic()-started,
                         objective_means={k:v/recent_count for k,v in objective_totals.items()},
                         score_scale=float(model.scale), learning_rates=[g['lr'] for g in optimizer.param_groups],
                         pair_updates=pair_updates, pair_samples_seen=pair_samples_seen)
            if 'genre_evaluation' in latest:
                entry['genre_macro']=latest['genre_evaluation']['music_macro']
                entry['genre_question_macros']={k:v['music_macro'] for k,v in latest['genre_question_variants'].items()}
            if 'music_macro' in latest:
                entry['music_macro'] = latest['music_macro']
            if 'music_views' in latest:
                entry['music_view_macros']={view:report['music_macro'] for view,report in latest['music_views'].items()}
            history.append(entry)
            print(json.dumps(entry), flush=True)
            recent_loss, recent_count = 0., 0
            objective_totals = dict(main_ce=0., infonce=0., pair_ce=0., swap_loss=0.)
            key = (current["top1"], -current["loss"])
            if args.selection_metric=='audio-grounded':
                gap=min(latest['audio_top1_gaps'].values(),default=float('-inf'))
                key=(int(gap>=.05),current['top1'],gap,-current['loss'])
            elif args.selection_metric in ('music-grounded','music-window-grounded'):
                from .metrics import music_selection_key
                key=music_selection_key(latest,initial if args.selection_metric=='music-window-grounded' else None)
            elif args.selection_metric=='genre-grounded':
                from .metrics import genre_selection_key
                key=genre_selection_key(latest,initial)
            if key > best_key:
                best_key = key
                stale_evaluations = 0
                save("best.pt", latest)
            else:
                stale_evaluations += 1
            save("last.pt", latest)
            write_json(run / "metrics.json", dict(status="running", stage=args.stage, train_samples=len(indices),
                                                  history=history, first_gradients=gradients, best_validation_key=list(best_key),
                                                  initial_validation=initial, configuration=vars(args), model=model.spec))
            if args.patience and stale_evaluations >= args.patience:
                print(json.dumps(dict(event='early_stopping',step=step,patience=args.patience)),flush=True)
                break
    progress(status="evaluating")
    changes = {name: head_signature(head.state_dict()) != baseline[name]
               for name, head in (("state", model.state_head), ("action", model.action_head))}
    if args.stage >= 2 and not all(changes.values()):
        raise ValueError("Training completed without updating both CLM heads")
    if args.stage == 1 and any(changes.values()):
        raise ValueError("Stage 1 changed a frozen CLM head")
    best = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"], strict=True)
    model.eval()
    evaluation = {s: evaluate_model(model, data, rows, ids, args.batch_size)
                  for s, ids in (("train", indices), ("validation", splits["validation"]))}
    report = dict(status="completed", stage=args.stage, diagnostic=args.diagnostic, train_samples=len(indices),
                  initial_validation=initial, evaluations=evaluation, history=history, best_step=best["step"],
                  last_step=step, seconds=time.monotonic()-started, first_gradients=gradients,
                  trained_heads_changed=changes, encoder_calls_during_training=0, model=model.spec,
                  feature_provenance=manifest["provenance"], feature_manifest=manifest_hash, configuration=vars(args),
                  supervised_audio_pair_rows=paired_rows,
                  auxiliary_train_samples=len(auxiliary), total_train_samples=len(trained_indices),
                  pair_updates=pair_updates, pair_samples_seen=pair_samples_seen,
                  score_scale=float(model.scale), stopped_early=epoch<args.epochs and not (args.steps and step>=args.steps),
                  audio_dependence_gate_passed=bool(evaluation["validation"]["audio_top1_gaps"]) and all(gap >= .05 for gap in evaluation["validation"]["audio_top1_gaps"].values()))
    write_json(run / "metrics.json", report)
    if args.selection_metric in ('music-grounded','music-window-grounded'):
        report['audio_dependence_gate_passed'] = bool(evaluation['validation']['music_macro']['audio_top1_gaps']) and all(
            gap >= .05 for values in evaluation['validation']['music_macro']['family_audio_top1_gaps'].values() for gap in values.values())
        write_json(run/'metrics.json',report)
    if args.selection_metric=='music-window-grounded':
        from .metrics import music_selection_key
        report['serving_window_gate_passed']=bool(music_selection_key(evaluation['validation'],initial)[0])
        write_json(run/'metrics.json',report)
    if args.selection_metric=='genre-grounded':
        from .metrics import genre_selection_key
        report['genre_gate_passed']=bool(genre_selection_key(evaluation['validation'],initial)[0])
        write_json(run/'metrics.json',report)
    progress(status="completed")
    if not args.diagnostic and run.resolve() == (run_root/f"stage{args.stage}").resolve():
        from .monitor import render_png
        render_png(args.stage, run)
    print(json.dumps(dict(event="training_completed", stage=args.stage, best_step=best["step"],
                          validation=evaluation["validation"]["correct"]["overall"], heads_changed=changes, run=str(run))), flush=True)
    return report
