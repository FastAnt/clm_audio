# Selected models and artifact backup — 2026-10-04

At the user's request, all trained results from the remote project were saved:
checkpoints from every run, selected and historical model exports, embedded
optimizer/scheduler/RNG state, metrics, logs, PNGs, and provenance. Datasets,
source/prepared audio, and frozen features were explicitly excluded from the
new offsite backup.

The selected common and music models are published in Git LFS under
[checkpoints/](../checkpoints/README.md). Each has unchanged `best.pt`, `last.pt`,
and `calibrated.pt`, plus cards, calibration, evaluation, and SHA256.
These six checkpoint objects total 1,940,459,526 bytes. This inventory describes
the two original selected exports; later exports are listed in the catalog.

| Export | Raw best SHA256 | Calibrated SHA256 |
|---|---|---|
| `samsone_clm_common` | `4f4137e03975054ffe725ec635589afa31c8ddfd11b1d71db20e863a43710e78` | `7f95b6d94bfc22da61afa41ff3119db4b087912550dc964f25ecbc0c5d8d5b98` |
| `samsone_clm_music` — FMA Large v2 | `a87f43181d3cd1b5ce98c0664e82a1c2fd05d8197754a28e540415e3ac4107b8` | `a1cc52344117b320c420aefc7a61f82a89f3d546b3430ac39c77fbd0e8dab71b` |

Common remains the originally selected AudioMCQ model. Validation did not select
the additional remote Stage 4 model as its replacement. Common test Top-1 is
62.79%, with +3.76/+5.37pp gaps: the strict test requirement of at least 5pp above
both ablations was not met; temporal performance is weak. FMA Large v2 was
selected on validation with retention of previous music tasks. Reserved test
was not used for selection.

Offsite package:
`/srv/data/samsone_clm/backups/runpod-2026-10-04/`.
The original snapshot contains 428 regular files / 20,466,904,972 bytes.
`inventory.json` and `SHA256SUMS` describe the package. After transfer,
`local_verification.json` records actual reads and verification of all local
bytes. `project.bundle` preserves Git history independently of GitHub; Git LFS
weights are also present as ordinary files under `data/models/`.

Inference requires the selected `calibrated.pt`, project code, and official
frozen Samsone356M. Further training also requires downloading/preparing data
and recomputing features, or using previously retained local feature stores.
A `last.pt` checkpoint does not replace the training dataset. A completed
OneCycle schedule cannot be extended with `--resume`: additional epochs start
separately from raw `best.pt` with a new optimizer and schedule.

Original checkpoints, embedded manifests, and SHA256 hashes are not rewritten.
Official pretrained weights and datasets are not added to Git. The historical
GitHub release `common-2026-10-03` is retained.

Verification completed: SHA256 matched for all 428 local files. Both Git LFS
exports were fetched from GitHub into an empty cache; all 6 tensor objects
matched their hashes. [Local proof](results/backup-2026-10-04/local_verification.json),
[GitHub proof](results/backup-2026-10-04/github_lfs_verification.json),
[inventory](results/backup-2026-10-04/inventory.json).

The remote web process was stopped. After backup, raw/processed/features were
removed, followed by models/runs/cache/tmp and the project Python environment
at the user's further request. Code and the Git LFS executable were retained.
The final cleanup left 499.61GiB free and no project GPU jobs.
[Final cleanup proof](results/backup-2026-10-04/remote_cleanup_final.json).
The initial [cleanup proof](results/backup-2026-10-04/remote_cleanup.json)
describes the earlier step when weights/runs were still on the server.

Selected exports were restored to the original workstation under
`/srv/data/samsone_clm/models/{samsone_clm_common,samsone_clm_music}/`;
all 22 files were checked against the Git catalog. The previous local music
export was preserved separately. The local demo was configured for CPU
inference with both models: [TEST_MUSIC](TEST_MUSIC.md). This authorization
covers local inference; it does not restart local training or precompute.
