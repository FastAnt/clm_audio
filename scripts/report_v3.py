"""Read-only final report and PNG for all completed v3 training runs."""
import json
import os
from pathlib import Path

from samsone_clm.common import ROOT,write_json


def main():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=ROOT/'runs/v3'
    reports={p.parent.name:json.loads(p.read_text()) for p in sorted(directory.glob('*/metrics.json'))}
    compact={}
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for name,report in reports.items():
        if report.get('status')!='completed' or 'initial_validation' not in report: continue
        ev=report['evaluations']['validation']
        compact[name]=dict(train_samples=report['train_samples'],auxiliary_samples=report.get('auxiliary_train_samples',0),
                          validation=ev,score_scale=report.get('score_scale'),best_step=report['best_step'],
                          last_step=report['last_step'],heads_changed=report['trained_heads_changed'],
                          encoder_calls=report['encoder_calls_during_training'],configuration=report['configuration'])
        history=report['history']
        # Selection artifacts refer to an existing run; don't draw its curves twice.
        if report.get('artifact_kind'):continue
        steps=[r['step'] for r in history]
        axes[0,0].plot(steps,[r['validation']['top1']*100 for r in history],label=name)
        axes[0,1].plot(steps,[r['validation']['loss'] for r in history],label=name)
        axes[0,2].plot(steps,[r['train_loss'] for r in history],label=name)
        axes[1,0].plot(steps,[r['audio_top1_gaps'].get('random',float('nan'))*100 for r in history],label=name)
        axes[1,1].plot(steps,[r['audio_top1_gaps'].get('silence',float('nan'))*100 for r in history],label=name)
        axes[1,2].plot(steps,[r.get('score_scale',100.) for r in history],label=name)
    titles=['Validation Top-1 (%)','Validation CE','Training objective','Correct - random (pp)','Correct - silence (pp)','Logit scale']
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title)
        ax.set_xlabel('Optimizer step')
        ax.grid(alpha=.25)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
    for ax in (axes[1,0],axes[1,1]):ax.axhline(5.,linestyle='--',color='gray',linewidth=1)
    fig.suptitle('Frozen Samsone356M + shared adapter + official pretrained CLM heads | v3')
    fig.savefig(directory/'training_comparison.png',dpi=160)
    plt.close(fig)
    native=json.loads((directory/'native_baseline/results.json').read_text())
    report=dict(scope='AudioMCQ fixed audio-disjoint validation; no successful System1 claim unless final audio gate passes',
                native_baseline=native['summary'], native_hidden_proof=native['extraction_proof'],runs=compact)
    write_json(directory/'result.json',report)
    lines=['# Samsone-CLM v3 experiment','',
           'Architecture: frozen Samsone356M -> shared linear adapter -> original CLM state/action heads; cosine ranking with checkpoint/learned scale.',
           '',f"Native 100-example baseline: correct={native['summary']['correct']['overall']['accuracy']:.1%}, random={native['summary']['random']['overall']['accuracy']:.1%}, silence={native['summary']['silence']['overall']['accuracy']:.1%}.",
           'Native generation is a diagnostic only; training and ranking use no generation. Native baseline and 2048-row ranking validation have different sample counts and prompts.',
           '', '| Run | Main train | Correct | Random | Silence | CE | Audio gaps (pp) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in compact.items():
        ev=r['validation']; gap=ev['audio_top1_gaps']
        lines.append(f"| {name} | {r['train_samples']} | {ev['correct']['overall']['top1']:.2%} | {ev['random']['overall']['top1']:.2%} | {ev['silence']['overall']['top1']:.2%} | {ev['correct']['overall']['loss']:.4f} | {100*gap['random']:.2f} / {100*gap['silence']:.2f} |")
    for filename in ('recipe_comparison.json','scale_recipe_comparison.json','scale_decision.json','paired_audio_evaluation.json',
                     'raw_test.json','calibrated_test.json','window_transfer.json','source_integrity.json','partial_feature_recovery.json'):
        path=directory/filename
        if path.exists():report[filename]=json.loads(path.read_text())
    for filename in ('provisional_50k_live.json','provisional_50k_validation_ci.json','provisional_50k_window.json'):
        path=directory/filename
        if path.exists():report[filename]=json.loads(path.read_text())
    selected=directory/'stage2-audio-grounded/calibrated.json'
    if selected.exists():
        report['provisional_50k_calibration']=json.loads(selected.read_text())
        lines+=['','50K retained-checkpoint selection is a separate artifact, not another training run.',
                'Provisional 50K demo uses validation-only scale calibration. Validation CI and calibration are fitted/selected on validation; reserved test remains separate.',
                'For the selected 50K CE checkpoint, music/speech show clear audio gains; sound is weaker and temporal has no measurable benefit. Passing the overall gate does not establish all-category success.']
    if 'provisional_50k_window.json' in report:
        lines+=['',f"Provisional 50K: full-clip vs 10s center-window Top1 changed in {report['provisional_50k_window.json']['prediction_change_rate']:.1%} of {report['provisional_50k_window.json']['samples']} validation cases. This is stability, not crop accuracy."]
    write_json(directory/'result.json',report)
    repair=ROOT/'processed/v3/repair_report.json'
    if repair.exists():
        repairs=json.loads(repair.read_text())
        report['source_repair']=repairs
        write_json(directory/'result.json',report)
        lines+=['',f"Source audit checked {report.get('source_integrity.json',{}).get('unique_paths','all')} unique WAVs. One TACOS WAV failed checksum and contained a missing tail.",
                'It was restored from the original ZIP with its published MD5 and exact PCM verification. Three QA rows received corrected PCM IDs and duration. Stage1 and all held-out rows are unchanged.',
                'A validated 18432-state prefix and all 128538 actions were recovered without rerunning Samsone. Original damaged artifacts were retained.']
    lines+=['','Detailed metrics, configuration, checksums, scale and head-change checks are in each run metrics.json.',
            'Paired confidence intervals and reserved-test results are separate artifacts; they do not select the training recipe.',
            '', 'Masked bidirectional InfoNCE uses exact text IDs, multiple annotated positives and only explicit distractor relationships. CE remains part of the objective.',
            'Stage1 freezes both heads and temperature; Stage2+ trains both heads, adapter and temperature. Encoder is never called during training.',
            '', 'Resource guard: >=20 GiB RAM/disk before jobs, supervised termination below 12 GiB during jobs; per-training checks every 100 updates.',
            'Three old NTFS JSON entries could not be read. They were preserved; new experiment files use runs/v3. No mounted-volume filesystem repair was attempted.']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:{'correct':v['validation']['correct']['overall']['top1'], 'gaps':v['validation']['audio_top1_gaps']} for k,v in compact.items()}),flush=True)


if __name__=='__main__':main()
