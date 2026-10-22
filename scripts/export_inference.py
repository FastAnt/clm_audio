"""Export an inference-only checkpoint without machine paths or training state."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile

import torch

ENCODER_FIELDS = ('recipe', 'samsone_sha256', 'hidden_width', 'dtype', 'sample_rate',
                  'max_audio_samples', 'compute_dtype', 'position_recipe')


def check_metadata(value):
    if isinstance(value, dict):
        for key, item in value.items():
            check_metadata(key)
            check_metadata(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_metadata(item)
    elif isinstance(value, str):
        if value.startswith(('/', '~', '\\')) or ':\\' in value:
            raise ValueError('Absolute filesystem path found in inference metadata')
    elif value is not None and not isinstance(value, (int, float, bool)):
        raise ValueError('Unsupported inference metadata type')


def export(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing checkpoint')
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    state = checkpoint['model_state']
    if not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('Expected a tensor-only model_state')
    payload = dict(model_state=state, spec=checkpoint['spec'],
                   feature_provenance={k: checkpoint['feature_provenance'][k]
                                       for k in ENCODER_FIELDS if k in checkpoint['feature_provenance']})
    check_metadata(payload['spec'])
    check_metadata(payload['feature_provenance'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
        restored = torch.load(temporary, map_location='cpu', weights_only=True)
        if restored['spec'] != payload['spec'] or restored['feature_provenance'] != payload['feature_provenance']:
            raise ValueError('Metadata changed during serialization')
        if set(restored['model_state']) != set(state) or any(
            restored['model_state'][k].dtype != v.dtype or
            restored['model_state'][k].shape != v.shape or
            not torch.equal(restored['model_state'][k], v) for k, v in state.items()
        ):
            raise ValueError('Model tensors changed during serialization')
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = export(args.source, args.output)
    with destination.open('rb') as handle:
        print(hashlib.file_digest(handle, 'sha256').hexdigest(), destination.name)


if __name__ == '__main__':
    main()
