import math

import torch
from torch import nn
from torch.nn import functional as F

from samsone_clm.objectives import bidirectional_choice_loss,canonical_action_ids
from samsone_clm.model import AudioCLM


class IdentityModel:
    scale = 2.
    def state(self,h): return F.normalize(h,dim=-1)
    def action(self,h): return F.normalize(h,dim=-1)


def test_unknown_candidates_are_masked_and_repeated_answers_are_positive():
    states=torch.tensor([[1.,0.],[.8,.2],[0.,1.]],requires_grad=True)
    actions=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]],requires_grad=True)
    choices=torch.tensor([[0,1],[1,0],[1,2]])
    labels=torch.tensor([0,1,0])
    loss,parts=bidirectional_choice_loss(IdentityModel(),states,actions,choices,labels)
    logits=2*F.normalize(states,dim=-1)@F.normalize(actions,dim=-1).T
    allowed=torch.tensor([[True,True,False],[True,True,False],[False,True,True]])
    masked=logits.masked_fill(~allowed,float('-inf'))
    expected_forward=F.cross_entropy(masked,torch.tensor([0,0,1]))
    reverse0=-F.log_softmax(masked[:,0],0)[:2].mean()
    reverse1=-F.log_softmax(masked[:,1],0)[2]
    assert torch.allclose(parts['forward'],expected_forward)
    assert torch.allclose(parts['reverse'],(reverse0+reverse1)/2)
    assert torch.allclose(loss,(expected_forward+(reverse0+reverse1)/2)/2)
    loss.backward()
    assert torch.isfinite(states.grad).all() and torch.isfinite(actions.grad).all()


def test_column_permutation_keeps_contrast_loss():
    model=IdentityModel()
    states=torch.eye(2)
    actions=torch.eye(2)
    a,_=bidirectional_choice_loss(model,states,actions,torch.tensor([[0,1],[0,1]]),torch.tensor([0,1]))
    b,_=bidirectional_choice_loss(model,states,actions,torch.tensor([[1,0],[0,1]]),torch.tensor([1,1]))
    assert torch.allclose(a,b)


def test_unmasked_choice_objective_matches_official_clm_multi_positive_recipe():
    states=torch.tensor([[1.,0.],[.8,.2],[0.,1.]])
    actions=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])
    choices=torch.tensor([[0,1],[1,0],[1,2]])
    labels=torch.tensor([0,1,0])
    loss,_=bidirectional_choice_loss(IdentityModel(),states,actions,choices,labels,mask_unknown=False)
    lg=2*F.normalize(states,dim=-1)@F.normalize(actions,dim=-1).T
    targets=torch.tensor([[1.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    forward=-(targets*F.log_softmax(lg,1)).sum(1).mean()
    w=targets.T
    keep=w.sum(1)>0
    w=w[keep]/w[keep].sum(1,keepdim=True)
    reverse=-(w*F.log_softmax(lg.T[keep],1)).sum(1).mean()
    assert torch.allclose(loss,(forward+reverse)/2)


def test_native_case_normalization_merges_identical_action_inputs():
    assert canonical_action_ids(['Female','female','male']).tolist()==[0,0,2]


def test_learnable_scale_is_frozen_stage1_and_can_decrease_from_cap():
    model=AudioCLM(960,adapter_kind='linear',learnable_scale=True)
    assert not model.logit_scale.requires_grad
    model.set_stage(2)
    assert model.logit_scale.requires_grad
    logits=model.scale*torch.tensor([[0.,1.]])
    F.cross_entropy(logits,torch.tensor([0])).backward()
    assert model.logit_scale.grad > 0
    before=float(model.scale.detach())
    with torch.no_grad(): model.logit_scale -= .001*model.logit_scale.grad
    model.constrain_scale()
    assert float(model.scale.detach()) < before
    restored=AudioCLM.from_spec(model.spec)
    restored.load_state_dict(model.state_dict(),strict=True)
    assert torch.equal(restored.scale,model.scale)
    import json
    json.dumps(model.spec)
