"""Promote only fully committed, verified prefix shards to a reusable cache; no audio inference."""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file,save_file

from samsone_clm.common import ROOT,digest,read_rows,sha256,write_json
from samsone_clm.precompute import load_features


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    args=parser.parse_args()
    source=args.directory
    provenance=json.loads((source/'provenance.json').read_text())
    original=read_rows(args.dataset)
    choices=sorted({c for r in original for c in r['choices']})
    actions=sorted(source.glob('actions-*.safetensors'))
    if sum(len(load_file(str(p))['hidden']) for p in actions)!=len(choices):
        raise RuntimeError('Action shards are not complete')
    states=sorted(source.glob('correct-*.safetensors'))
    count=0
    for path in states:
        if int(path.stem.split('-')[1])!=count:raise RuntimeError('State shards have a gap')
        count+=len(load_file(str(path))['hidden'])
    rows=original[:count]
    for row in rows:
        # Only source-consistent rows can be promoted, even if a source changed after extraction.
        if sha256(Path(row['audio_path']))!=row['audio_sha256']:
            raise RuntimeError('Committed prefix includes changed source audio')
    provenance=dict(provenance,dataset_semantic_sha256=digest([{k:v for k,v in r.items() if k!='audio_sha256'} for r in rows]))
    directory=ROOT/'features'/('spike-recovered-'+digest(provenance)[:16])
    directory.mkdir(parents=True,exist_ok=True)
    for path in actions+states:
        target=directory/path.name
        if not target.exists():shutil.copyfile(path,target)
        if sha256(target)!=sha256(path):raise RuntimeError('Recovery copy mismatch')
    width=provenance['hidden_width']
    for path in states:
        values=load_file(str(path))['hidden']
        for mode in ('random','silence'):
            target=directory/path.name.replace('correct-',mode+'-')
            if not target.exists():save_file({'hidden':torch.zeros(len(values),width,dtype=torch.bfloat16)},str(target))
    ids={c:i for i,c in enumerate(choices)}
    manifest=dict(directory=str(directory),provenance=provenance,rows=rows,choices=choices,
                  candidate_indices=[[ids[c] for c in r['choices']] for r in rows],
                  ablation_available=[False]*count,random_audio_indices=list(range(count)),
                  shard_sha256={p.name:sha256(p) for p in sorted(directory.glob('*.safetensors'))},
                  recovery_scope='Validated committed prefix only; random/silence are unavailable placeholders')
    write_json(directory/'provenance.json',provenance)
    write_json(directory/'manifest.json',manifest)
    load_features(str(directory/'manifest.json'),dtype=torch.bfloat16)
    write_json(ROOT/'runs/v3/partial_feature_recovery.json',dict(states=count,actions=len(choices),directory=str(directory),
                original_directory=str(source),source_checksums_verified=True,all_shards_validated=True))
    print(dict(states=count,actions=len(choices),directory=str(directory)),flush=True)


if __name__=='__main__':main()
