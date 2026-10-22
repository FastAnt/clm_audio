"""Publish immutable common model/data packages only after completed Stage 4."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import uuid

import torch

from samsone_clm.common import ROOT,commit_file,read_rows,sha256,write_json
from samsone_clm.resources import require_resources


def publish(directory, files, card):
    staging=directory.parent/(directory.name+'.export-'+uuid.uuid4().hex)
    staging.mkdir(parents=True)
    checksums={}
    try:
        for name, source in files.items():
            target=staging/name
            shutil.copyfile(source,target)
            with target.open('rb') as handle:os.fsync(handle.fileno())
            checksums[name]=sha256(target)
        write_json(staging/'checksums.json',checksums)
        write_json(staging/'card.json',card)
        if directory.exists():
            existing=json.loads((directory/'checksums.json').read_text())
            if existing!=checksums or any(sha256(directory/name)!=value for name,value in existing.items()):
                raise ValueError(f'Existing immutable export differs: {directory}')
            shutil.rmtree(staging)
        else:
            staging.replace(directory)
    except BaseException:
        if staging.exists():shutil.rmtree(staging)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--feature-manifest',type=Path,required=True)
    args=parser.parse_args()
    require_resources()
    report=json.loads((args.run_root/'stage4/metrics.json').read_text())
    if report.get('status')!='completed' or report['stage']!=4:
        raise ValueError('samsone_clm_common requires completed Stage 4')
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if ck['stage']!=4 or ck.get('overfit'):
        raise ValueError('Common checkpoint must be a trained Stage 4 checkpoint')
    if ck['step'] < math.ceil(report['train_samples']/report['configuration']['batch_size']):
        raise ValueError('Common best must have completed at least one full training pass')
    base=ROOT/'processed/v3'
    rows=read_rows(base/'stage4_all.jsonl')
    all_train=sum(row['split']=='train' for row in rows)
    if report['train_samples']!=all_train:
        raise ValueError('Stage 4 did not train on the complete immutable AudioMCQ train split')
    if not all(report['trained_heads_changed'].values()) or report['encoder_calls_during_training']!=0:
        raise ValueError('Common training architecture verification failed')
    required=['raw_test.json','calibrated_test.json','paired_audio_evaluation.json',
              'window_transfer.json','live_inference.json']
    for name in required:
        if not (args.run_root/name).exists():raise ValueError(f'Missing final verification artifact: {name}')
    model_dir=ROOT/'models/samsone_clm_common'
    data_dir=ROOT/'processed/samsone_clm_common'
    data_files={'canonical.jsonl':base/'canonical.jsonl','stage4_all.jsonl':base/'stage4_all.jsonl'}
    data_card=dict(name='samsone_clm_common',source='inclusionAI/AudioMCQ',filter='audio_contribution == strong',
                   train_samples=all_train,canonical_samples=sum(1 for _ in (base/'canonical.jsonl').open()),
                   split_policy='Existing immutable audio-disjoint train/validation/test; source audio is referenced, not copied',
                   feature_manifest=str(args.feature_manifest),feature_manifest_sha256=sha256(args.feature_manifest))
    publish(data_dir,data_files,data_card)
    model_files={'best.pt':args.checkpoint,'last.pt':args.run_root/'stage4/last.pt',
                 'calibrated.pt':args.checkpoint.parent/'calibrated.pt',
                 'calibration.json':args.checkpoint.parent/'calibrated.json',
                 'training_metrics.json':args.run_root/'stage4/metrics.json'}
    model_files.update({name:args.run_root/name for name in required})
    model_card=dict(name='samsone_clm_common',stage=4,parent='official pretrained CLM heads + frozen Samsone356M',
                    training_dataset=str(data_dir),shared_samsone_checkpoint=str(ROOT/'models/samsone_356m_model_bf16.ckpt'),
                    frozen_encoder=ck['feature_provenance'],architecture=ck['spec'],
                    train_samples=all_train,selected_step=ck['step'],
                    audio_dependence_gate_passed=report['audio_dependence_gate_passed'],
                    inference_checkpoint=str(model_dir/'calibrated.pt'),music_initialization=str(model_dir/'best.pt'),
                    quality_scope='Completion of full available AudioMCQ; capability limits are recorded in per-category evaluation')
    publish(model_dir,model_files,model_card)
    write_json(args.run_root/'common_export.json',dict(model=str(model_dir),dataset=str(data_dir),stage=4,completed=True))
    print(json.dumps(dict(model=str(model_dir),dataset=str(data_dir),completed=True)),flush=True)


if __name__=='__main__':main()
