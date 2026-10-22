"""Remote-only continuation for music serving windows; never overwrites common."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from samsone_clm.common import ROOT,PROJECT,sha256,write_json
from samsone_clm.monitor import read_json
from samsone_clm.resources import require_resources


def phase(name,log=None):
    write_json(ROOT/'runs/music/window_pipeline_status.json',dict(phase=name,job_name='window-pipeline',
        updated_at=datetime.now(timezone.utc).isoformat(),
        log=str(log or ROOT/'runs/music-preparation/window-pipeline.log'),
        experiment='Stage6 mixed full/10s music views; original shared adapter and both heads; frozen Samsone'))
    print(name,flush=True)


def pointer(name):return read_json(ROOT/f'features/{name}.json')['path']


def run(*arguments):
    require_resources()
    subprocess.run([sys.executable,*map(str,arguments)],cwd=PROJECT,check=True)


def main():
    phase('precompute-music-windows',ROOT/'runs/music-preparation/window-precompute.log')
    while True:
        status=read_json(ROOT/'runs/music-preparation/window-precompute.exit.json')
        if status:
            if status['exit_code']:raise RuntimeError('Window precompute failed')
            break
        pid=int((ROOT/'runs/music-preparation/window-precompute.pid').read_text())
        if not Path('/proc',str(pid),'cmdline').exists() or not Path('/proc',str(pid),'cmdline').read_bytes():
            time.sleep(2)
            status=read_json(ROOT/'runs/music-preparation/window-precompute.exit.json')
            if not status or status['exit_code']:raise RuntimeError('Window precompute stopped; inspect resources/log')
        time.sleep(15)
    phase('compose-stage6-windows')
    run('scripts/compose_music_features.py','--stage','6','--new-manifests',pointer('musicbench_manifest'),
        '--common-manifest',ROOT/'features/spike-a86e057eae695e43/manifest.json',
        '--stage5-manifest',pointer('music_stage5_manifest'),'--replay-window-manifest',pointer('music_windows_manifest'),
        '--pointer',ROOT/'features/music_stage6_window_manifest.json')
    manifest=pointer('music_stage6_window_manifest');run_root=ROOT/'runs/music-window'
    parent=ROOT/'runs/music/stage6-full32/best.pt'
    common=['scripts/train_music.py','--stage','6','--feature-manifest',manifest,
            '--run-root',run_root,'--parent-checkpoint',parent,'--resume']
    phase('overfit-stage6-windows');run(*common,'--diagnostic')
    phase('train-stage6-windows');run(*common,'--run-name','stage6-window','--epochs','32','--patience','0',
                                    '--batch-size','256','--selection-metric','music-window-grounded')
    directory=run_root/'stage6-window';report=read_json(directory/'metrics.json')
    accepted=report.get('serving_window_gate_passed',False)
    decision=dict(accepted=accepted,test_used_for_selection=False,
        parent_sha256=sha256(parent),candidate_sha256=sha256(directory/'best.pt'),
        selection='All view/family audio gaps>=5pp; each full family accuracy drop<=1pp; window gain>=1pp; mixed balanced validation',
        initial_validation=report['initial_validation'],selected_validation=report['evaluations']['validation'])
    write_json(ROOT/'runs/music/window_selection.json',decision)
    if accepted:
        phase('finalize-stage6-windows')
        run('scripts/finalize_music_stage.py','--stage','6','--feature-manifest',manifest,
            '--run-dir',directory,'--export-dir',ROOT/'models/samsone_clm_music_window')
    else:
        from samsone_clm.monitor import render_png
        render_png(6,directory)
        write_json(directory/'checksums.json',{p.name:sha256(p) for p in directory.iterdir() if p.is_file() and p.name!='checksums.json'})
    phase('return-stage6-windows')
    print(json.dumps({'training_completed':True,'window_candidate_accepted':accepted}),flush=True)


if __name__=='__main__':main()
