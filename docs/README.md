# Documentation

Recorded state on 2026-10-04: Stages 1–6, the window continuation, and FMA Large
v2 are complete. All trained artifacts were returned and verified; selected
weights were published in Git LFS. RunPod was cleared, and the demo was moved
to local CPU inference. The common export remains immutable.

| Document | Purpose |
|---|---|
| [PORTFOLIO](PORTFOLIO.md) | Evidence-backed project description and suggested CV wording |
| [ARTIFACTS](ARTIFACTS.md) | Inference checkpoint format, provenance and model verification |
| [CLM_TRANSFER_REVIEW](CLM_TRANSFER_REVIEW.md) | Comparison with official CLM, audits of two user tracks, and limitations of transfer |
| [NOVELTY_REVIEW](NOVELTY_REVIEW.md) | Related-work review, potential novelty, and missing evidence |
| [MODEL_BACKUP](MODEL_BACKUP.md) | Git LFS weights, verified offsite backup, and RunPod cleanup |
| [FMA Large v2](results/music-genre/REPORT.md) | Selected music export and genre-quality limitations |
| [TEST_MUSIC](TEST_MUSIC.md) | Checkpoints, web/API/CLI use, and music-model limitations |
| [Final window experiment](results/music-stage6-window/REPORT.md) | Full/window tests, selection, SHA256, plots, and preserved versions |
| [REMOTE_PROGRESS](REMOTE_PROGRESS.md) | Recorded remote results and artifact transfers |
| [PROJECT_STATE](PROJECT_STATE.md) | Architecture, Stage 1–4 results, limitations, and next steps |
| [REMOTE_MIGRATION](REMOTE_MIGRATION.md) | Transfer inventory, checksums, installation, and continuation |
| [COMMON_MUSIC_PLAN](COMMON_MUSIC_PLAN.md) | Musical QA preparation and Stage 5–6 plan |
| [MUSIC_DATA_EXPANSION](MUSIC_DATA_EXPANSION.md) | Sources, formats, sizes, and known overlaps |
| [V3_EXPERIMENT](V3_EXPERIMENT.md) | Corrected recipe; intermediate numbers describe 50K training |
| [SAMSONE_AUDIT](SAMSONE_AUDIT.md) | Official Samsone356M, pooling, and FP32 extraction checks |
| [results/v3](results/v3/REPORT.md) | Final raw JSON, configurations, logs, and PNGs |
| [results/common](results/common/card.json) | Immutable common export model card and checksums |

Historical documents remain available for reproducibility. They describe older
recipes and do not replace the final configuration:
[SPIKE_REPORT](SPIKE_REPORT.md), [ITERATION_1000](ITERATION_1000.md),
[REVIEW_AND_REPAIR](REVIEW_AND_REPAIR.md),
[AUDIO_CONTRAST_REPAIR](AUDIO_CONTRAST_REPAIR.md),
[TRAINING_PIPELINE](TRAINING_PIPELINE.md),
[HISTORICAL_README](HISTORICAL_README.md), and the original
[RESEARCH_PLAN](../RESEARCH_PLAN.md).

`diagrams/*-v2.png` describe the historical corrected-v2 recipe. The root README
specifies the pretrained-head architecture. Archived absolute paths record the
original host; preserving them maintains provenance. The IO resolver relocates
the data root without modifying those artifacts or hashes.
