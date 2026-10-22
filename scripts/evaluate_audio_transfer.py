"""Audio-identity clustered paired confidence intervals for correct/random/silence."""
import argparse
from pathlib import Path
import numpy as np
import torch

from samsone_clm.common import write_json
from samsone_clm.model import AudioCLM
from samsone_clm.precompute import load_features
from samsone_clm.resources import require_resources


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--feature-manifest',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--splits',nargs='+',choices=('validation','test'),default=['validation','test'])
    args=parser.parse_args()
    require_resources()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=AudioCLM.from_spec(ck['spec'])
    model.load_state_dict(ck['model_state'],strict=True)
    model.cuda().eval().requires_grad_(False)
    data,manifest=load_features(args.feature_manifest,dtype=torch.bfloat16)
    data={k:v.cuda() for k,v in data.items()}
    rows=manifest['rows']
    result=dict(checkpoint=args.checkpoint,method='paired bootstrap over audio_identity clusters; 2000 resamples, 95% intervals',splits={})
    for split in args.splits:
        indices=torch.tensor([i for i,r in enumerate(rows) if r['split']==split and data['ablation_available'][i]],device='cuda')
        hits={}
        with torch.no_grad():
            for mode in ('correct','random','silence'):
                predictions=torch.cat([model(data[mode][part],data['actions'][data['indices'][part]]).argmax(-1) for part in indices.split(64)])
                hits[mode]=(predictions==data['labels'][indices]).cpu().numpy()
        selected=[rows[i] for i in indices.tolist()]
        categories={}
        for kind in ('overall','music','sound','speech','temporal'):
            mask=np.array([kind=='overall' or r['question_type']==kind for r in selected])
            ids=[r['audio_identity'] for r,m in zip(selected,mask) if m]
            unique=sorted(set(ids)); mapping={v:i for i,v in enumerate(unique)}
            cluster=np.array([mapping[v] for v in ids]); counts=np.bincount(cluster,minlength=len(unique))
            stats={mode:float(hits[mode][mask].mean()) for mode in hits}
            gaps={}
            rng=np.random.default_rng(731)
            for mode in ('random','silence'):
                differences=hits['correct'][mask].astype(float)-hits[mode][mask]
                sums=np.bincount(cluster,weights=differences,minlength=len(unique))
                draws=rng.integers(0,len(unique),size=(2000,len(unique)))
                sampled=sums[draws].sum(1)/counts[draws].sum(1)
                gaps[mode]=dict(gap=float(differences.mean()),ci95=np.quantile(sampled,[.025,.975]).tolist(),
                                lost=int(((hits['correct'][mask]) & ~hits[mode][mask]).sum()),
                                gained=int((~hits['correct'][mask] & hits[mode][mask]).sum()))
            categories[kind]=dict(samples=int(mask.sum()),audio_clusters=len(unique),accuracy=stats,gaps=gaps)
        result['splits'][split]=categories
    write_json(args.output,result)
    print({split:value['overall'] for split,value in result['splits'].items()})


if __name__=='__main__':main()
