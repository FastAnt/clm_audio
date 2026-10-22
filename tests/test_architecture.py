import torch
from torch.nn import functional as F

from samsone_clm.encoder import FrozenSamsone
from samsone_clm.model import AudioCLM


def test_shared_adapter_gradients_and_stage_freeze():
    torch.set_num_threads(2)
    model = AudioCLM(960)
    x = torch.randn(3, 960)
    candidates = torch.randn(3, 4, 960)
    before = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("adapter.")}
    hits = []
    handle = model.adapter.register_forward_hook(lambda module, inputs, output: hits.append(id(module)))
    scores = model(x, candidates)
    F.cross_entropy(scores, torch.tensor([0, 1, 2])).backward()
    assert hits == [id(model.adapter), id(model.adapter)]
    assert sum(p.grad.norm().item() for p in model.adapter.parameters()) > 0
    assert all(p.grad is None for head in [model.state_head, model.action_head] for p in head.parameters())
    torch.optim.SGD(model.adapter.parameters(), lr=0.01).step()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    handle.remove()
    model.set_stage(2)
    model.zero_grad()
    F.cross_entropy(model(x, candidates), torch.tensor([0, 1, 2])).backward()
    for head in [model.state_head, model.action_head]:
        assert all(p.requires_grad for p in head.parameters())
        assert sum(p.grad.norm().item() for p in head.parameters()) > 0


def test_last_valid_handles_left_and_right_padding():
    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    pooled = FrozenSamsone._last_valid(hidden, mask)
    assert torch.equal(pooled, torch.stack([hidden[0, 3], hidden[1, 1]]))


def test_choices_are_permutation_equivariant():
    model = AudioCLM(960).eval()
    x, choices = torch.randn(2, 960), torch.randn(2, 4, 960)
    permutation = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(model(x, choices)[:, permutation], model(x, choices[:, permutation]), atol=1e-6)


def test_both_branches_backpropagate_into_same_adapter():
    model = AudioCLM(960)
    x = torch.randn(2, 960)
    for branch in (model.state, model.action):
        model.zero_grad()
        branch(x)[:, 0].sum().backward()
        assert model.adapter[1].weight.grad.norm().item() > 0
