# Checkpoints and experiment artifacts

## Inference checkpoints

The selected common and music checkpoints are distributed through Git LFS.
Each contains three fields:

- `model_state`: trained adapter, state/action heads and logit scale.
- `spec`: dimensions, head configuration and scoring contract.
- `feature_provenance`: frozen Samsone version and feature-extraction contract.

These files support inference. They omit optimizer state, RNG, sample
identifiers and training-data locations, so they cannot resume training exactly.
Use a full training snapshot when exact resumption is required.

[export_inference.py](../scripts/export_inference.py) builds this format without
changing model tensors. [Verification results](INFERENCE_EXPORT_VERIFICATION.json)
record tensor equality and identical logits/probabilities on deterministic
hidden vectors. Repackaging changes the file SHA256; catalog.json and the
per-model checksums describe the distributed inference files.

## Historical results

Reports retain numerical results and original artifact hashes. Filesystem
locations in reports are illustrative; report copies are not exact-resume
manifests. Historical checkpoint hashes identify the original training files,
not the inference files in checkpoints/.

## Dependencies and licensing

Set SAMSONE_CLM_DATA to choose the data directory; the default is
/srv/data/samsone_clm. Official Samsone and CLM source revisions are pinned.

Apache-2.0 covers project-owned code and documentation. Third-party components,
source audio and model weights retain separate terms; see [THIRD_PARTY](../THIRD_PARTY.md)
and [NOTICE](../NOTICE). No unrestricted commercial model-weight license is asserted.
