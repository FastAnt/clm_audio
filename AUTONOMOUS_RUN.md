# State after the pause and migration preparation

Historical snapshot: Stages 1–4 are complete. The last supervisor finished
exporting `samsone_clm_common` on 2026-10-03 at 20:52 UTC. The user paused further
work; training/watchdog/web processes were stopped. Stages 5–6 had not started
at this point. Later results are recorded in the current README.

The implementation from `/opt/samsone-clm-backup` has been preserved
in this Git checkout. The original directory remains in place. Exact
configuration and metrics: [README](README.md),
[PROJECT_STATE](docs/PROJECT_STATE.md), [results archive](docs/results/v3/REPORT.md).

Large files: `/srv/data/samsone_clm/`. Immutable model:
`models/samsone_clm_common/`. Use raw best for further training and calibrated
weights for the API. Canonical dataset: `processed/v3/`; immutable dataset
export: `processed/samsone_clm_common/`. Cached full Stage 4 features:
`features/spike-a86e057eae695e43/`.

The next planned user action was to provide an SSH host. The
[transfer procedure](docs/REMOTE_MIGRATION.md), `scripts/bootstrap_remote.sh`,
`scripts/prepare_transfer.py`, and `scripts/continue_stage4.py` were prepared.
The latter creates a separate Stage 4 continuation run; it had not run in this
snapshot. Musical Stages 5–6 required implementation following the
[plan](docs/COMMON_MUSIC_PLAN.md).

Final common reserved test: correct 62.79% / random 59.03% / silence 57.42%;
gaps 3.76/5.37pp. The validation gate passed; the strict test gate of at least
5pp above both ablations did not. Temporal remains weak. These limitations
are retained in the Git snapshot.
