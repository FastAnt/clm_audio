# Working instructions

Read README.md, docs/PROJECT_STATE.md and docs/REMOTE_MIGRATION.md first.
Historical report paths are examples. Report hashes identify the original
training artifacts; distributed inference files have separate checksums.

Architecture is fixed: frozen Samsone356M on both branches, one shared adapter,
and original pretrained CLM state/action heads. Stage 1 freezes the heads;
Stage 2+ trains both heads. Never run the encoder inside the training loop.
No Qwen model or alternative encoder is used.

Use Python 3.12+, uv.lock and relevant pytest checks. Large files, features,
environments and caches belong under SAMSONE_CLM_DATA. Keep datasets, audio,
credentials and model binaries out of Git. Official submodules stay pinned.
No training/precomputation should start merely to inspect this repository.
