"""Read-only training dashboard data and publication-independent PNG snapshots."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading

from .common import ROOT

_plot_lock = threading.Lock()


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def log_tail(path, size=32768):
    try:
        with Path(path).open("rb") as source:
            source.seek(0, 2)
            source.seek(max(0, source.tell()-size))
            return source.read().decode(errors="replace")
    except OSError:
        return ""


def run_snapshot(run, stage):
    run = Path(run)
    metrics = read_json(run/"metrics.json")
    progress = read_json(run/"progress.json")
    info = read_json(run/"best.pt.info.json")
    updated = max((p.stat().st_mtime for p in (run/"metrics.json",run/"progress.json") if p.exists()),default=0)
    return dict(stage=stage,run_directory=str(run),status=metrics.get("status",progress.get("status","running" if "step" in progress else "not_started")),
        configuration=progress.get("configuration") or metrics.get("configuration",{}),
        progress=progress,history=metrics.get("history",[]),initial_validation=metrics.get("initial_validation",{}),
        evaluations=metrics.get("evaluations",{}),best_step=metrics.get("best_step",info.get("step")),
        best_checkpoint=str(run/"best.pt") if (run/"best.pt").exists() else None,
        last_checkpoint=str(run/"last.pt") if (run/"last.pt").exists() else None,
        first_gradients=progress.get("first_gradients") or metrics.get("first_gradients",{}),
        heads_changed=metrics.get("trained_heads_changed"),
        audio_dependence_gate_passed=metrics.get("audio_dependence_gate_passed"),
        genre_gate_passed=metrics.get("genre_gate_passed"),
        model=progress.get("model") or metrics.get("model",{}),
        train_samples=metrics.get("train_samples",progress.get("train_samples")),updated_timestamp=updated)


def experiment_directories():
    runs=ROOT/"runs"
    paths=set()
    for parent in (runs/"music",runs/"music-window"):
        paths.update(p for p in parent.glob("stage*") if p.is_dir())
    for parent in (runs/"music-genre").glob("iteration-*"):
        paths.update(p for p in parent.glob("stage6-genre-*") if p.is_dir())
    return sorted(p for p in paths if (p/"metrics.json").exists() or (p/"progress.json").exists())


def resolve_experiment(run_id):
    # Only known training directories are exposed; arbitrary paths are rejected.
    for run in experiment_directories():
        if run.relative_to(ROOT/"runs").as_posix()==run_id:
            return run
    raise ValueError("Unknown training experiment")


def experiment_label(run):
    name=run.name
    if name=="stage6-genre-v1":return "Stage 6 · FMA Medium · genre v1"
    if name=="stage6-genre-v2":return "Stage 6 · FMA Large · genre v2"
    if name=="stage6-window-lowlr":return "Stage 6 · 10s windows · selected music"
    if name=="stage6-window":return "Stage 6 · 10s windows · first run"
    if name=="stage6-full32":return "Stage 6 · full clips · 32 epochs"
    return "Stage "+("5" if name=="stage5" else "6")+" · "+name


def training_snapshot():
    pipeline=read_json(ROOT/"runs/music-genre/status.json") or read_json(ROOT/"runs/music/refinement_status.json") or read_json(ROOT/"runs/music/status.json") or read_json(ROOT/"runs/v3/status.json") or read_json(ROOT/"runs/pipeline_status.json")
    updated=pipeline.get("updated_at")
    if updated:pipeline["heartbeat_age_s"]=max(0,(datetime.now(timezone.utc)-datetime.fromisoformat(updated)).total_seconds())
    matches=list(re.finditer(r"(action|correct|random|silence) hidden (\d+)/(\d+)",log_tail(pipeline.get("log",""))))
    precompute={m[1]:dict(done=int(m[2]),total=int(m[3])) for m in matches}
    last_mode=matches[-1][1] if matches else None
    corpus=read_json(ROOT/"processed/full.report.json")
    run_root=Path(pipeline.get("run_root",ROOT/"runs"))
    experiments=[]
    for run in experiment_directories():
        stage=5 if run.name=="stage5" else 6
        entry=run_snapshot(run,stage)
        entry.update(id=run.relative_to(ROOT/"runs").as_posix(),label=experiment_label(run))
        experiments.append(entry)
    experiments.sort(key=lambda entry:entry["updated_timestamp"])
    stages=[]
    for stage in range(1,7):
        requested=Path(pipeline.get("stage_directories",{}).get(str(stage),run_root/("stage"+str(stage))))
        run=requested
        if not (run/"metrics.json").exists() and not (run/"progress.json").exists():
            if stage<=3:
                for fallback in (ROOT/"runs/history"/("stage"+str(stage)),ROOT/"runs/v3"/("stage"+str(stage)),ROOT/"runs"/("stage"+str(stage))):
                    if (fallback/"metrics.json").exists() or (fallback/"progress.json").exists():
                        run=fallback;break
            elif stage==6:
                previous=[e for e in experiments if e["stage"]==6 and (e["history"] or "step" in e["progress"])]
                if previous:run=Path(previous[-1]["run_directory"])
        item=run_snapshot(run,stage)
        item["target_samples"]={1:10000,2:50000,3:150000,4:corpus.get("selected_samples",{}).get("train"),5:None,6:None}[stage]
        if run!=requested:item["next_run_directory"]=str(requested)
        stages.append(item)
    diagnostics={}
    diagnostic_parents={run_root}
    diagnostic_parents.update((ROOT/"runs/music-genre").glob("iteration-*"))
    for parent in sorted(diagnostic_parents):
        for p in sorted(parent.glob("*/progress.json")):
            if not p.parent.name.startswith(("overfit-","diagnostic-")):continue
            key=p.parent.name if parent==run_root else parent.name+"/"+p.parent.name
            record=run_snapshot(p.parent,read_json(p).get("stage",6))
            diagnostics[key]={**record["progress"],**{k:v for k,v in record.items() if k!="progress"}}
    music_corpora={name:read_json(ROOT/("processed/music/"+name+".report.json")) for name in ("jamendo","nsynth","musicbench")}
    genre_corpora={}
    for name in ("fma_medium","fma_large"):
        report=read_json(ROOT/("processed/music_genre/"+name+".report.json"))
        if report:genre_corpora[name]={k:report.get(k) for k in ("samples","unique_recordings","unique_train_recordings","split_counts","split_leaks")}
    return dict(updated_at=datetime.now(timezone.utc).isoformat(),pipeline=pipeline,stages=stages,experiments=experiments,
        diagnostics=diagnostics,precompute=precompute,precompute_mode=last_mode,corpus=corpus,
        music_corpora=music_corpora,genre_corpora=genre_corpora,
        architecture=dict(state="audio + question → frozen Samsone → shared adapter → state_head",
            action="choice text → same frozen Samsone text path → SAME adapter → action_head",
            score="learned/checkpoint scale × cosine",loss="candidate CE + configured pair CE/swap",optimizer="AdamW",samsone_frozen=True),
        served_checkpoint=read_json(ROOT/"runs/served_checkpoint.json"))


def render_png(stage=None, run=None):
    with _plot_lock:
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"cache/matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if run is None:
            snapshot = training_snapshot()
            stage = stage or max((s["stage"] for s in snapshot["stages"] if s["history"]), default=1)
            data = snapshot["stages"][stage-1]
        else:
            run = Path(run)
            report = read_json(run/'metrics.json')
            progress = read_json(run/'progress.json')
            stage = stage or report.get('stage') or progress.get('stage')
            snapshot = {'updated_at':datetime.now(timezone.utc).isoformat()}
            data = dict(history=report.get('history',[]),progress=progress,
                        configuration=progress.get('configuration',report.get('configuration',{})),
                        status=report.get('status',progress.get('status','unknown')),
                        train_samples=report.get('train_samples',progress.get('train_samples')),
                        target_samples=None,run_directory=str(run))
        history = data["history"]
        config = data["configuration"]
        target=Path(data['run_directory'])/'training_curves.png'
        if data['status']=='completed' and target.exists() and 'training_curves.png' in read_json(target.parent/'checksums.json'):
            return target  # A dashboard download must not mutate a finalized bundle.
        fig, axes = plt.subplots(2,2, figsize=(12,8), constrained_layout=True)
        fig.suptitle(f"Samsone–CLM · Stage {stage} · {data['status']}\n"
                     f"train samples={data['train_samples'] or data['target_samples']} · batch={config.get('batch_size','—')} · "
                     f"adapter LR={config.get('lr','—')} · heads LR={config.get('head_lr','—')} · {snapshot['updated_at'][:19]} UTC")
        curve = data["progress"].get("train_curve", [])
        for ax in axes.flat:
            ax.grid(alpha=.2)
            ax.set_xlabel("Optimizer step")
        if history:
            steps = [p["step"] for p in history]
            axes[0,0].plot(steps,[p["train_loss"] for p in history],label="Train objective",color="#2c8f7c")
            axes[0,0].plot(steps,[p["validation"]["loss"] for p in history],label="Validation CE",color="#8361bc")
            for key,label,color in (("top1","Top-1","#2c8f7c"),("top2","Top-2","#8361bc")):
                axes[0,1].plot(steps,[100*p["validation"][key] for p in history],label=label,color=color)
            musical = [point for point in history if 'music_macro' in point]
            if musical:
                axes[0,1].plot([p['step'] for p in musical],[100*p['music_macro']['top1'] for p in musical],
                               label='Music balanced, equal task families',linestyle='--',color='#cc8e37')
            for view,color in [('full','#347bbc'),('serving-window','#c56583')]:
                points=[p for p in history if view in p.get('music_view_macros',{})]
                if points:
                    axes[0,1].plot([p['step'] for p in points],
                        [100*p['music_view_macros'][view]['family_top1']['perception'] for p in points],
                        label=view+' perception balanced',linestyle=':',color=color)
            for key,label,color in (("random","Correct − random","#cc8e37"),("silence","Correct − silence","#8361bc")):
                points=[p for p in history if key in p.get("audio_top1_gaps",{})]
                axes[1,0].plot([p["step"] for p in points],[100*p["audio_top1_gaps"][key] for p in points],label=label,color=color)
                if musical:
                    values=[p for p in musical if key in p['music_macro'].get('audio_top1_gaps',{})]
                    axes[1,0].plot([p['step'] for p in values],[100*p['music_macro']['audio_top1_gaps'][key] for p in values],
                                   label='Music balanced − '+key,color=color,linestyle='--')
            axes[1,1].plot(steps,[p["validation"]["ece"] for p in history],label="ECE",color="#cc8e37")
            axes[1,1].plot(steps,[p["validation"]["mean_correct_probability"] for p in history],label="Mean P(correct)",color="#2c8f7c")
        if curve:
            axes[0,0].plot([p["step"] for p in curve],[p["loss"] for p in curve],label="Train CE (live window)",alpha=.45,color="#2c8f7c")
        axes[0,0].set_title("Cross Entropy")
        axes[0,1].set_title("Validation accuracy (%)")
        axes[0,1].set_ylim(0,100)
        axes[1,0].set_title("Audio ablation gaps (percentage points)")
        axes[1,0].axhline(0,color="grey",linewidth=1)
        axes[1,0].axhline(5,color="grey",linestyle="--",label="Audio gate: 5 pp")
        axes[1,1].set_title("Validation calibration")
        for ax in axes.flat:
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            if not history and not curve:
                ax.text(.5,.5,"No optimizer metrics yet",ha="center",va="center",transform=ax.transAxes)
        target=Path(data["run_directory"])/"training_curves.png"
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(".png.tmp")
        fig.savefig(temporary,format="png",dpi=160)
        plt.close(fig)
        temporary.replace(target)
        return target
