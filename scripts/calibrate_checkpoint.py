"""Fit a single positive score scale on validation; preserve rankings and original best.pt."""
import argparse
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from samsone_clm.common import write_json
from samsone_clm.metrics import metrics
from samsone_clm.model import AudioCLM
from samsone_clm.precompute import load_features
from samsone_clm.resources import require_resources


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--feature-manifest',required=True)
    args=parser.parse_args()
    require_resources()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=AudioCLM.from_spec(checkpoint['spec'])
    model.load_state_dict(checkpoint['model_state'],strict=True)
    model.cuda().eval().requires_grad_(False)
    data,manifest=load_features(args.feature_manifest,dtype=torch.bfloat16)
    ids=torch.tensor([i for i,r in enumerate(manifest['rows']) if r['split']=='validation'])
    with torch.no_grad():
        cosine=torch.cat([model(data['correct'][part].cuda(),data['actions'][data['indices'][part]].cuda(),
                                data['candidate_mask'][part].cuda()).cpu()/float(model.scale)
                          for part in ids.split(64)]).double()
    labels=data['labels'][ids]
    old_scale=float(model.scale)
    log_scale=torch.tensor(math.log(min(old_scale,20.)),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([log_scale],max_iter=100,line_search_fn='strong_wolfe')
    valid = torch.isfinite(cosine)
    finite_cosine = cosine.masked_fill(~valid, 0.)
    def closure():
        optimizer.zero_grad()
        loss=F.cross_entropy((log_scale.exp()*finite_cosine).masked_fill(~valid,float('-inf')),labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    scale=min(100.,max(.01,float(log_scale.detach().exp())))
    before=metrics((old_scale*cosine).float(),labels)
    after=metrics((scale*cosine).float(),labels)
    assert torch.equal((old_scale*cosine).argmax(-1),(scale*cosine).argmax(-1))
    assert after['loss']<=before['loss']+1e-5
    model.enable_learnable_scale()
    with torch.no_grad():model.logit_scale.fill_(math.log(scale))
    model.constrain_scale()
    checkpoint['model_state']={k:v.cpu() for k,v in model.state_dict().items()}
    checkpoint['spec']=model.spec
    checkpoint['calibration']=dict(method='single positive score scale fitted on validation only',
                                   samples=len(ids),old_scale=old_scale,scale=scale,validation_before=before,validation_after=after,
                                   rankings_unchanged=True,test_labels_used=False)
    output=Path(args.checkpoint).parent/'calibrated.pt'
    temporary=output.with_suffix('.pt.tmp')
    torch.save(checkpoint,temporary)
    temporary.replace(output)
    write_json(output.with_suffix('.json'),checkpoint['calibration'])
    print(checkpoint['calibration'])


if __name__=='__main__':main()
