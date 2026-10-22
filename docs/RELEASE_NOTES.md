# AudioMCQ common snapshot, 2026-10-03

Frozen Samsone356M → shared adapter → original pretrained CLM state/action heads. Stage 1–4 completed; Stage 4 uses all 181592 available strong train QA. No Qwen model is used. Music Stage5/6 remain planned.

This private release backs up the learned `samsone_clm_common` model:

- `best.pt`: raw Stage4 best, for additional training/music initialization.
- `last.pt`: Stage4 last, including optimizer/scheduler/RNG.
- `calibrated.pt`: inference; positive scale fitted on validation only.
- Original model cards, training/test reports, checksums, and dataset card/checksums.

Asset: `samsone_clm_common.tar` (~922 MB). SHA256 and byte count are committed in `docs/results/backup.json`. Extract under `SAMSONE_CLM_DATA`. The official Samsone356M/CLM pretrained checkpoints can be downloaded with `scripts/download_models.py`.

Source audio, canonical QA rows and cached features are transferred separately using `scripts/prepare_transfer.py`. The code, pinned upstream revisions, docs, raw metrics and plots are in the tagged Git commit. See `docs/REMOTE_MIGRATION.md`.

Reserved test (2048 balanced QA): correct62.79%, random59.03%, silence57.42%; positive audio gaps3.76/5.37pp. The strict ≥5pp gate against both ablations is not passed on test; temporal remains weak. This snapshot preserves a trained experimental result, not a claim of complete capability.
