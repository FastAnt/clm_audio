import torch
from torch.nn import functional as F
from samsone_clm.model import AudioCLM
from samsone_clm.staged_train import parameter_groups, validate_initialization


def test_stage2_optimizer_updates_both_original_heads_and_shared_adapter():
    torch.set_num_threads(2)
    model = AudioCLM(960)
    model.set_stage(2)
    groups = parameter_groups(model, .001, .001)
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    before = [module.inp.weight.detach().clone() for module in (model.state_head, model.action_head)]
    optimizer = torch.optim.SGD(groups)
    F.cross_entropy(model(torch.randn(2, 960), torch.randn(2, 4, 960)), torch.tensor([0, 1])).backward()
    optimizer.step()
    for head, initial in zip((model.state_head, model.action_head), before):
        assert not torch.equal(head.inp.weight, initial)


def test_resume_rejects_different_stage_or_feature_manifest():
    import pytest
    from samsone_clm.common import digest
    from samsone_clm.staged_train import ENCODER_FIELDS
    manifest = {"provenance":{field:"fixture" for field in ENCODER_FIELDS}}
    ck = dict(stage=2, feature_provenance=manifest["provenance"], feature_manifest=digest(manifest))
    validate_initialization(ck, manifest, 2, resume=True)
    with pytest.raises(ValueError, match="Resume"):
        validate_initialization(ck, manifest, 3, resume=True)
    ck["feature_manifest"] = "changed"
    with pytest.raises(ValueError, match="Resume"):
        validate_initialization(ck, manifest, 2, resume=True)
