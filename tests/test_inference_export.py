import importlib.util
from pathlib import Path

import pytest
import torch

script = Path(__file__).resolve().parents[1] / 'scripts/export_inference.py'
spec = importlib.util.spec_from_file_location('inference_export', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_export_preserves_tensors_and_removes_training_identifiers(tmp_path):
    source, target = tmp_path / 'training.pt', tmp_path / 'inference.pt'
    tensor = torch.randn(3, 4)
    torch.save(dict(model_state={'adapter.weight': tensor}, spec={'samsone_width': 4},
                    feature_provenance={'recipe': 'fixture', 'dtype': 'bfloat16',
                                        'components': [{'path': '/private/features'}]},
                    optimizer={'state': 'private'}, args={'path': '/private/run'},
                    train_sample_ids=['private-id']), source)
    original = source.read_bytes()
    module.export(source, target)
    result = torch.load(target, weights_only=True)
    assert set(result) == {'model_state', 'spec', 'feature_provenance'}
    assert result['feature_provenance'] == {'recipe': 'fixture', 'dtype': 'bfloat16'}
    assert torch.equal(result['model_state']['adapter.weight'], tensor)
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        module.export(source, target)


def test_export_rejects_paths_in_inference_configuration(tmp_path):
    source, target = tmp_path / 'training.pt', tmp_path / 'inference.pt'
    torch.save(dict(model_state={'weight': torch.ones(1)},
                    spec={'nested': {'checkpoint': '/private/model.pt'}},
                    feature_provenance={}), source)
    with pytest.raises(ValueError, match='filesystem path'):
        module.export(source, target)
    assert not target.exists()
