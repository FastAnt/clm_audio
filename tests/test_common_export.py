"""An exported common parent must remain immutable during specialization."""
import importlib.util
from pathlib import Path

import pytest


module_path = Path(__file__).parents[1]/'scripts/export_common.py'
spec = importlib.util.spec_from_file_location('export_common', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_common_export_is_idempotent_and_refuses_changed_weights(tmp_path):
    source = tmp_path/'trained.pt'
    source.write_bytes(b'original common weights')
    output = tmp_path/'model'
    module.publish(output, {'best.pt':source}, {'name':'samsone_clm_common'})
    module.publish(output, {'best.pt':source}, {'name':'samsone_clm_common'})
    source.write_bytes(b'music finetuned weights')
    with pytest.raises(ValueError, match='immutable export differs'):
        module.publish(output, {'best.pt':source}, {'name':'samsone_clm_common'})
    assert (output/'best.pt').read_bytes() == b'original common weights'
    assert not list(tmp_path.glob('model.export-*'))
