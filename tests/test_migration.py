import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from samsone_clm import common
from samsone_clm.precompute import load_features
from samsone_clm.staged_train import ENCODER_FIELDS, validate_initialization


def test_relocation_keeps_exact_feature_manifest_and_shard_integrity(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    directory = tmp_path / 'features/spike-relocation-fixture'
    directory.mkdir(parents=True)
    for mode in ('correct', 'random', 'silence', 'actions'):
        save_file({'hidden': torch.ones(1, 3, dtype=torch.bfloat16)}, str(directory / f'{mode}-000000.safetensors'))
    manifest = dict(directory=str(common.ORIGINAL_DATA_ROOT / 'features/spike-relocation-fixture'),
                    rows=[dict(answer_index=0)], choices=['yes'], candidate_indices=[[0]],
                    provenance=dict(hidden_width=3), shard_sha256={p.name: common.sha256(p) for p in directory.glob('*.safetensors')})
    common.write_json(directory / 'manifest.json', manifest)
    original_digest = common.digest(manifest)
    loaded, stored = load_features(common.ORIGINAL_DATA_ROOT / 'features/spike-relocation-fixture/manifest.json')
    assert common.digest(stored) == original_digest
    assert torch.equal(loaded['correct'], torch.ones(1, 3))
    assert common.resolve_data_path('/another/project/file') == Path('/another/project/file')
    (directory / 'correct-000000.safetensors').write_bytes(b'damaged transfer')
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_features(directory / 'manifest.json')


def test_additional_stage4_requires_same_stage_dataset_and_encoder():
    manifest = dict(provenance={field: 'fixture' for field in ENCODER_FIELDS})
    checkpoint = dict(stage=4, feature_manifest=common.digest(manifest), feature_provenance=dict(manifest['provenance']))
    validate_initialization(checkpoint, manifest, 4, continuation=True)
    with pytest.raises(ValueError, match='Resume'):
        validate_initialization(checkpoint, manifest, 3, continuation=True)
    with pytest.raises(ValueError, match='mutually exclusive'):
        validate_initialization(checkpoint, manifest, 4, resume=True, continuation=True)
    checkpoint['feature_manifest'] = 'changed dataset'
    with pytest.raises(ValueError, match='exact feature manifest'):
        validate_initialization(checkpoint, manifest, 4, continuation=True)
    checkpoint['feature_provenance']['samsone_sha256'] = 'different frozen weights'
    with pytest.raises(ValueError, match='Frozen encoder mismatch'):
        validate_initialization(checkpoint, manifest, 4, continuation=True)
