"""Validation calibration, one fixed test evaluation, PNG, live audio check and export."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from samsone_clm.common import ROOT, PROJECT, resolve_data_path, sha256, write_json
from samsone_clm.infer import Ranker
from samsone_clm.precompute import load_features
from samsone_clm.monitor import render_png
from samsone_clm.resources import require_resources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=int,choices=[5,6],required=True)
    parser.add_argument('--feature-manifest',required=True)
    parser.add_argument('--run-dir',type=Path,help='Explicit separate music experiment directory')
    parser.add_argument('--export-dir',type=Path,help='Separate export directory for an additional experiment')
    args=parser.parse_args()
    require_resources()
    run=args.run_dir or ROOT/f'runs/music/stage{args.stage}'
    report=json.loads((run/'metrics.json').read_text())
    if report['status']!='completed' or not all(report['trained_heads_changed'].values()) or report['encoder_calls_during_training']:
        raise ValueError('Music stage did not complete with the required architecture')
    subprocess.run([sys.executable,str(PROJECT/'scripts/calibrate_checkpoint.py'),
                    '--checkpoint',str(run/'best.pt'),'--feature-manifest',args.feature_manifest],check=True)
    # A completed test file is immutable to restarts. Never select epochs using it.
    if not (run/'evaluation_test.json').exists():
        subprocess.run([sys.executable,'-m','samsone_clm.evaluate','--checkpoint',str(run/'calibrated.pt'),
                        '--feature-manifest',args.feature_manifest,'--split','test',
                        '--output',str(run/'evaluation_test.json')],check=True)
    render_png(args.stage,run)
    if not (run/'live_audio.json').exists():
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        data,manifest=load_features(args.feature_manifest,dtype=torch.bfloat16)
        ranker=Ranker(run/'calibrated.pt')
        model=ranker.model
        selected={}
        for i,row in enumerate(manifest['rows']):
            if row['split']=='test' and row.get('music_task') and row['music_task'] not in selected:
                selected[row['music_task']]=i
        results=[]
        with torch.no_grad():
            for task,i in selected.items():
                row=manifest['rows'][i]
                ranking=ranker.rank(row['audio_path'],row['question'],row['choices'])
                probabilities={item['choice']:item['probability'] for item in ranking}
                live=torch.tensor([probabilities[text] for text in row['choices']])
                cached=model(data['correct'][i:i+1].cuda(),data['actions'][data['indices'][i:i+1]].cuda(),
                             data['candidate_mask'][i:i+1].cuda()).softmax(-1).cpu()[0,:len(row['choices'])]
                delta=float((live-cached).abs().max())
                results.append(dict(task=task,audio=row['audio_path'],question=row['question'],choices=row['choices'],
                                    answer_index=row['answer_index'],prediction=int(live.argmax()),probabilities=live.tolist(),
                                    ranking=ranking,max_live_cached_probability_difference=delta))
                if not torch.isfinite(live).all() or delta > .02:
                    raise ValueError(f'Live/cache inconsistency in {task}: {delta}')
        write_json(run/'live_audio.json',dict(samples=len(results),frozen_encoder=True,results=results,passed=True))
    manifest=json.loads(resolve_data_path(args.feature_manifest).read_text())
    if not manifest.get('music_only'):
        raise ValueError('A musical export requires music-only training data')
    test=json.loads((run/'evaluation_test.json').read_text())
    parent=report['configuration'].get('init_checkpoint') or report['configuration'].get('continue_checkpoint')
    card=dict(name='samsone_clm_music_stage'+str(args.stage),stage=args.stage,
              parent=parent,parent_checkpoint_sha256=sha256(resolve_data_path(parent)) if parent else None,
              stage_train_qa=report['train_samples'],music_only=True,
              encoder_frozen=True,encoder_calls_during_training=0,both_original_heads_train=True,shared_adapter=True,
              validation=report['evaluations']['validation'],test=test,
              test_used_for_selection=False,feature_manifest=manifest['directory']+'/manifest.json',
              data_quality='NSynth full-note labels, optionally silence-padded; Jamendo full-track consensus projected to excerpts (weak labels); MusicBench automatic confidence-filtered structural labels',
              live_audio_verified=True)
    if 'serving_window_gate_passed' in report:
        card['serving_window_gate_passed']=report['serving_window_gate_passed']
        card['validation_views_are_paired']=True
    if 'genre_gate_passed' in report:
        card['genre_gate_passed']=report['genre_gate_passed']
        card['data_quality']='FMA official primary track genre projected to exact 10s windows (weak local labels); replay retains original NSynth/Jamendo/MusicBench quality notes'
        card['genre_evaluation_recordings_are_paired_across_questions']=True
        card['dataset_components']=manifest.get('components',[])
    write_json(run/'model_card.json',card)
    demo=ROOT/'runs/music/demo.pt'
    if demo.is_symlink() and args.export_dir is None:
        temporary=demo.with_suffix('.pt.tmp')
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(run/'calibrated.pt')
        temporary.replace(demo)
        write_json(ROOT/'runs/served_checkpoint.json',dict(stage=args.stage,checkpoint=str(run/'calibrated.pt'),
                   checkpoint_sha256=sha256(run/'calibrated.pt'),selection='validation, then validation-only calibration'))
    write_json(run/'checksums.json',{p.name:sha256(p) for p in run.iterdir() if p.is_file() and p.name!='checksums.json'})
    if args.stage==6:
        target=args.export_dir or ROOT/'models/samsone_clm_music'
        if target.exists():
            old=json.loads((target/'checksums.json').read_text())
            if old.get('best.pt') != sha256(run/'best.pt'):
                raise ValueError('Refusing to overwrite a different music export')
        target.mkdir(parents=True,exist_ok=True)
        for name in ('best.pt','last.pt','calibrated.pt','calibrated.json','evaluation_test.json',
                     'metrics.json','training_curves.png','live_audio.json'):
            shutil.copyfile(run/name,target/name)
        card['name']='samsone_clm_music'
        write_json(target/'card.json',card)
        write_json(target/'checksums.json',{p.name:sha256(p) for p in target.iterdir() if p.is_file() and p.name!='checksums.json'})
    print(json.dumps(dict(stage=args.stage,finalized=True,test_top1=test['correct']['overall']['top1'])),flush=True)


if __name__=='__main__':
    main()
