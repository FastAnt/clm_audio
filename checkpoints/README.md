# Selected inference checkpoints

The selected common and music models are distributed as Git LFS files:

| Model | File | Purpose |
|---|---|---|
| samsone_clm_common | samsone_clm_common/calibrated.pt | General AudioMCQ ranking |
| samsone_clm_music | samsone_clm_music/calibrated.pt | FMA Large genre continuation |

These exports contain model_state, spec and the frozen encoder contract only.
Optimizer, RNG, training sample identifiers and machine paths are excluded.
They are inference-only, not exact-resume snapshots. Full training snapshots are required for exact resumption.

All tensors and configuration match the original calibrated checkpoints. The
file hashes differ because the archives were repackaged. See
[verification](../docs/INFERENCE_EXPORT_VERIFICATION.json) and catalog.json.

```bash
git lfs install
uv run python scripts/download_learned_models.py
uv run samsone-clm-serve --checkpoint "$SAMSONE_CLM_DATA/models/samsone_clm_music/calibrated.pt" --port 8800
```

Official Samsone and reference CLM-head downloads remain separate. The Apache
license for project code does not establish unrestricted commercial rights for
third-party models, source audio or learned weights. See THIRD_PARTY.md and NOTICE.
