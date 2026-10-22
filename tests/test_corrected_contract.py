import types

import torch

from samsone_clm.encoder import FrozenSamsone
from samsone_clm.model import AudioCLM


def test_both_heads_receive_unit_inputs_and_scale_is_preserved_on_reload():
    torch.set_num_threads(2)
    model = AudioCLM(960, adapter_kind="linear")
    norms = []
    handles = [head.register_forward_pre_hook(lambda module, inputs: norms.append(inputs[0].norm(dim=-1)))
               for head in (model.state_head, model.action_head)]
    inputs = torch.randn(3, 960)
    actions = torch.randn(3, 4, 960)
    scores = model(inputs, actions)
    for handle in handles:
        handle.remove()
    assert len(norms) == 2
    assert all(torch.allclose(value, torch.ones_like(value), atol=1e-6) for value in norms)
    assert model.scale == 100.0
    restored = AudioCLM.from_spec(model.spec)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.head_input_l2 and restored.scale == model.scale
    assert torch.allclose(scores, restored(inputs, actions))


def test_legacy_spec_is_not_silently_reinterpreted_as_corrected_contract():
    model = AudioCLM.from_spec(dict(samsone_width=960, bottleneck=960))
    assert model.scale == 1.0 and not model.head_input_l2
    assert model._adapt(torch.randn(3, 960)).norm(dim=-1).mean() > 10


def test_native_positions_ignore_padding_and_legacy_path_remains_available():
    encoder = FrozenSamsone.__new__(FrozenSamsone)
    encoder.width = 3
    encoder.legacy = False
    observed = {}
    def transformer(**kwargs):
        observed.update(kwargs)
        return types.SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])
    encoder.transformer = transformer
    embeds = torch.randn(2, 4, 3)
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    pooled = encoder._forward(embeds, mask)
    assert torch.equal(observed["position_ids"], torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1]]))
    assert torch.equal(pooled, torch.stack([embeds[0, 3], embeds[1, 1]]))
    encoder.legacy = True
    observed.clear()
    encoder._forward(embeds, mask)
    assert "position_ids" not in observed
