# Migration to a remote GPU

The remote installation at `/home/ubuntu/dev_clm` was prepared, Stage 4-first
data verified, the additional Stage 4 run completed, and artifacts returned
locally. See [REMOTE_PROGRESS](REMOTE_PROGRESS.md) for results and musical
preparation. The runbook below preserves the original pre-migration plan;
statements about pending installation are historical. Training on the original
workstation remains paused.

## 1. Storage locations

| Object | Storage |
|---|---|
| Complete Python project, UI, tests, configs, uv.lock | This Git repository |
| Official source code | Pinned `vendor/samsone` and `vendor/CLM` submodules |
| Final metrics/cards/checksums/plots and v3 history | `docs/results/` in Git |
| Learned common best/last/calibrated and reports | Data volume + historical `common-2026-10-03` GitHub release; selected exports now also in Git LFS |
| Official Samsone356M / CLM heads checkpoints | Data volume; downloadable with SHA256 verification |
| Canonical datasets, source WAV, cached features | Data volume; transferred with `rsync` |
| Host Python environment and wheel caches | Recreate remotely; do not transfer ARM `.venv` |

Default data root: `/srv/data/samsone_clm`. Set a different absolute
`SAMSONE_CLM_DATA` remotely. `resolve_data_path()` remaps original data paths at
IO time **without changing** manifests, checkpoints, canonical rows, or their
hashes. Hash-matched resume does not require rewriting dataset paths.

## 2. Verifiable inventory

From the source project root:

```bash
export SAMSONE_CLM_DATA=/srv/data/samsone_clm
uv run --no-sync python scripts/prepare_transfer.py --profile training --verify
uv run --no-sync python scripts/prepare_transfer.py --profile full
```

`training` includes official weights, immutable common model/dataset exports,
canonical v3 snapshots, four v3 feature stores, and checkpoints/metrics for all
v3 runs. Stage 4 continuation needs cached hidden, not source audio. Verified
snapshot: **4437 files / 13.825 GB**.

`full` adds all available canonical WAV files, AudioMCQ metadata/provenance,
and downloaded music metadata. It supports Stages 5–6, replay/new-QA precompute,
live inference, and further audits. Duplicate raw archives, host caches and
environment, and older v1/v2 runs are excluded but retained on the source.

Files are written under `$SAMSONE_CLM_DATA/runs/migration/`:

- `PROFILE.json`: exact paths/bytes/SHA256 and checksum provenance.
- `PROFILE.files0`: NUL-separated paths for safe `rsync --from0`.
- `PROFILE.summary.json`: file counts and exact byte budget.

`--verify` reads actual file bytes. Without it, known WAV/feature/export SHA256
values come from provenance; this is **not a new full check of 190GB of audio**.
Run full verification after transfer and budget for reading all audio. The
inventory includes the repaired TACOS file. Unreadable historical NTFS JSON
entries under runs/v2 are excluded.

## 3. Clone and remote installation

After receiving the SSH host, check RAM, disk, and GPU/driver and choose a remote
data root. Example destination setup:

```bash
git clone --recurse-submodules https://github.com/FastAnt/clm_audio.git
cd clm_audio
export SAMSONE_CLM_DATA=/srv/data/samsone_clm
bash scripts/bootstrap_remote.sh --data-root "$SAMSONE_CLM_DATA"
```

While the repository is private, the destination needs GitHub access or a
prepared checkout transferred over SSH. Official revisions are pinned in Git
and `configs/migration.json`.

Bootstrap pins Python 3.12/uv.lock and cu130 PyTorch, places caches/environment
on the data root, and checks CUDA plus at least 20GiB available RAM/disk.
Check driver compatibility and actual GPU memory on the destination. At the
time of this original runbook, syntax/dry-run checks had passed and real setup
on the destination was still pending.

## 4. File transfer

Adapt these commands to the actual destination user and host:

```bash
rsync -a --partial --info=progress2 --from0 \
  --files-from=/srv/data/samsone_clm/runs/migration/full.files0 \
  /srv/data/samsone_clm/ remote-user@remote-host:/srv/data/samsone_clm/

rsync -a /srv/data/samsone_clm/runs/migration/ \
  remote-user@remote-host:/srv/data/samsone_clm/runs/migration/
```

Repeating the commands resumes partial files and transfers missing files.
Transfer the inventory JSON too. The `runs/v3/demo.pt` symlink holds an original
absolute path and is excluded; launch the API with explicit
`models/samsone_clm_common/calibrated.pt`.

Verify on the destination:

```bash
export SAMSONE_CLM_DATA=/srv/data/samsone_clm
uv run --no-sync python scripts/prepare_transfer.py \
  --inventory "$SAMSONE_CLM_DATA/runs/migration/full.json" --verify
uv run --no-sync pytest -q
```

Also compare source/destination encoder/head configurations and Stage 4 cached
predictions on identical examples. Common states do not need full precompute
again. Transfer checkpoint/optimizer bytes unchanged; `.pt` maps CPU to the
actual CUDA device when loaded.

## 5. Historical GitHub backup of common weights

Release tag: `common-2026-10-03`; asset: `samsone_clm_common.tar`. The archive
contains learned best/last/calibrated weights, model cards/results/checksums,
and dataset card/checksums. It excludes source audio, canonical rows, features,
and official pretrained weights.

```bash
gh release download common-2026-10-03 --repo FastAnt/clm_audio \
  --pattern samsone_clm_common.tar --dir "$SAMSONE_CLM_DATA/tmp"
tar -xf "$SAMSONE_CLM_DATA/tmp/samsone_clm_common.tar" -C "$SAMSONE_CLM_DATA"
uv run --no-sync python scripts/download_models.py
```

The archive restores learned weights/inference once audio inputs are available.
Exact training resume also needs the feature stores and canonical snapshots
listed in the inventory. Do not extract over a different model; common is
immutable. The archive SHA256 is recorded in `docs/results/backup.json`.

## 6. Additional Stage 4 epochs

Continue a completed Stage 4 from **raw best**, preserving the saved common
export and creating a separate output directory:

```bash
# Inspect only; this does not start training.
uv run --no-sync python scripts/continue_stage4.py --dry-run

# Execute on a prepared remote GPU after verifying transferred data.
uv run --no-sync python scripts/continue_stage4.py \
  --epochs 8 --out "$SAMSONE_CLM_DATA/runs/remote-stage4/stage4"
```

Use a fresh optimizer/schedule. Proposed LR: adapter 1e-4, heads 5e-5, scale 1e-3;
retain other v3 CE/pair settings. Early stopping patience 4; validate every
2838 updates, one full pass. This was a proposed experiment when written.
Validation selects whether common is replaced; do not reuse the already-seen
reserved test for that decision.

Restore an interrupted **continuation** run with:

```bash
uv run --no-sync python scripts/continue_stage4.py --resume \
  --out "$SAMSONE_CLM_DATA/runs/remote-stage4/stage4"
```

The wrapper restores saved hyperparameters, optimizer, scheduler, and RNG.
Stage, feature hash, and frozen encoder must match exactly. `--resume` cannot
add epochs to an old OneCycle schedule; another continuation requires a fresh
output directory.

In the original snapshot this run was not connected to the fixed v3 dashboard.
Monitor `progress.json`, `metrics.json`, and stdout. The destination launcher
and monitor need GPU/RAM/disk checks.

## 7. Musical Stages 5–6

Historical plan, subsequently implemented: permit Stages 5/6 in model/train/
train_all, prepare canonical musical QA with source-level splits, validate
local 10-second labels, record quality/confidence, and balance tasks/replay.
Stage 5 starts from selected raw common; Stage 6 starts from Stage 5 best.
Samsone stays frozen; the shared adapter and both original heads remain.

See [COMMON_MUSIC_PLAN](COMMON_MUSIC_PLAN.md). The specialization is named
`samsone_clm_music`; `samsone_clm_common` remains unchanged.
