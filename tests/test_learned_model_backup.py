import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/download_learned_models.py'
SPEC = importlib.util.spec_from_file_location('learned_model_download', SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_verified_copy_is_atomic_and_idempotent(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'learned state')
    target = tmp_path / 'models/model.pt'
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert module.verified_copy(source, target, expected, source.stat().st_size) == 'restored'
    source.unlink()
    assert module.verified_copy(source, target, expected, target.stat().st_size) == 'already verified'


def test_existing_model_is_never_overwritten(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'model.pt'
    source.write_bytes(b'new')
    target.write_bytes(b'old')
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        module.verified_copy(source, target, expected, 3)
    assert target.read_bytes() == b'old'


@pytest.mark.parametrize('bad_size,bad_sha', [(1, False), (3, True)])
def test_corrupt_download_is_not_published(tmp_path, bad_size, bad_sha):
    source, target = tmp_path / 'source', tmp_path / 'models/model.pt'
    source.write_bytes(b'new')
    expected = '0' * 64 if bad_sha else hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='checksum or size'):
        module.verified_copy(source, target, expected, bad_size)
    assert not target.exists()
    assert not list(target.parent.glob('.download-*'))


def test_catalog_matches_lfs_pointers_and_model_metadata():
    project = SCRIPT.parents[1]
    if not (project / 'checkpoints/catalog.json').exists():
        pytest.skip('This code-only export does not distribute private model binaries or their catalog')
    catalog = json.loads((project / 'checkpoints/catalog.json').read_text())
    assert set(catalog['models']) == {'samsone_clm_common', 'samsone_clm_music'}
    for name, model in catalog['models'].items():
        tensors = [x for x in model['files'] if x['storage'] == 'git-lfs']
        assert {x['name'] for x in tensors} == ({'calibrated.pt'} if catalog['format_version'] == 2 else {'best.pt', 'last.pt', 'calibrated.pt'})
        expected = json.loads((project / 'checkpoints' / name / 'checksums.json').read_text())
        for item in model['files']:
            path = project / item['path']
            if item['storage'] == 'git-lfs':
                if path.stat().st_size == item['bytes']:
                    assert module.matches(path, item['sha256'], item['bytes'])
                else:
                    pointer = path.read_text()
                    assert pointer.startswith('version https://git-lfs.github.com/spec/v1\n')
                    assert 'oid sha256:' + item['sha256'] + '\n' in pointer
                    assert 'size ' + str(item['bytes']) + '\n' in pointer
            else:
                assert module.matches(path, item['sha256'], item['bytes'])
            if item['name'] in expected:
                assert expected[item['name']] == item['sha256']
