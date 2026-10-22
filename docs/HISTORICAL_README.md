> Historical README from the original project. Current state: [README](../README.md).

# Samsone–CLM Audio System 1

Project: `/opt/samsone-clm-backup`. All large data, features,
checkpoints, environment, and caches: `/srv/data/samsone_clm/`.

```text
STATE:  audio + question → Samsone-356M FROZEN → hidden → shared adapter → CLM state_head → z_state
ACTION: choice text      → Samsone text path FROZEN → hidden → SAME adapter → CLM action_head → z_action
score = cosine(z_state, z_action)
```

Official Contrastive-LM/CLM heads provide pretrained initialization. Stage 1
trains the adapter; Stages 2–4 train adapter **and both heads**. Samsone always
stays frozen. Qwen is neither downloaded nor used. Dimensions come from actual
checkpoints/configurations. Training, ranking API, and web demo do not generate
text; separate diagnostics verify native Samsone inference.

The historical active experiment is under `runs/v3/`, status
`runs/v3/status.json`. Completed at this snapshot: 100-example native baseline,
256-example overfit, 10K Stage 1, three 10K Stage 2 pilots, and 50K training.
The user required 150K Stage 3 and full Stage 4, recording regression without
cancelling AudioMCQ completion, followed by immutable
`models/samsone_clm_common/`.
[Common → music plan](COMMON_MUSIC_PLAN.md),
[v3 recipe and supervision](V3_EXPERIMENT.md).
Historical `runs/stageN/` and `runs/v2/` stay separate.
[Native Samsone and hidden audit](SAMSONE_AUDIT.md).

## Commands

```bash
cd ~/tools/samsone-clm
uv sync
uv run python scripts/download_models.py
uv run python scripts/download_data.py
uv run python scripts/prepare_data.py
uv run python scripts/precompute.py
uv run python -m samsone_clm.train --stage 1
uv run python -m samsone_clm.train --stage 2
uv run python -m samsone_clm.train --stage 3
uv run python -m samsone_clm.train --stage 4
```

Continue the autonomous v3 experiment with resource checks, feature reuse,
and final reserved test:

```bash
uv run --no-sync python -m samsone_clm.train_all --resume
```

The older v2 supervisor is retained for historical reproduction:

```bash
uv run --no-sync python scripts/run_corrected_pipeline.py --epochs 12 --batch-size 16
```

After initial `uv sync`, use `uv run --no-sync` in an active environment to avoid
concurrent library changes. The supervisor resumes checkpoints/features;
do not run another supervisor simultaneously.

Reproducible diagnostic pilot with dedicated verified audio pairs per update:

```bash
uv run --no-sync python scripts/prepare_audio_contrast.py --pair-mode reciprocal-answers
uv run --no-sync python scripts/run_audio_contrast_pilot.py
```

## Data

QA labels come only from `inclusionAI/AudioMCQ`, strong subset. The downloader
retrieves metadata for all seven sources and available waveforms. Transport
mirrors supply audio only. MusicCaps timestamps are checked against official
Google CSV. Revisions/checksums stay under `raw/`. Missing, corrupt, and >30s
audio is excluded with a recorded reason; audio is not cropped to fit a label.

Canonical rows contain `audio_path`, `question`, `choices`, `answer_index`,
`question_type`, `source_dataset`, identity, and provenance. Source recordings,
including different YouTube segments, and exact PCM duplicates share one split:
80/10/10. A SQLite index supports resume; stage snapshots are immutable.

Hidden extraction uses FP32 and native position IDs; BF16 sharded safetensors
store the result. State is the final last-valid hidden of the official
 audio/question sequence plus ` answer: `. Action is choice text through the
same frozen transformer, **before the adapter**. Compatible features are reused
by PCM identity + question or choice text, with provenance/checksum checks.
Encoder calls during training = 0.

## Checkpoints and evaluation

Each v3 stage saves `runs/v3/stageN/best.pt`, `last.pt`, and `metrics.json`.
Stage 1 and pilots select best by validation Top-1 and CE. At 50K, retained
best/last are additionally compared by audio gate, then accuracy; selection
is stored separately without altering originals. Stages 3–4 use
`--selection-metric audio-grounded`: both ablations first trail by at least
5pp, then Top-1 and CE. Gradients and changes to both heads are checked.
Main sizes: 10K / 50K / 150K / all available strong training rows (181592).
Auxiliary pair rows are counted separately.

```bash
uv run python -m samsone_clm.evaluate --checkpoint /path/to/best.pt --feature-manifest /path/to/matching/manifest.json --split test
uv run pytest -q
```

Metrics: Top-1, Top-2, CE, mean correct probability, ECE, separately for
music/sound/speech/temporal/overall. Correct/random/silence use **the same**
2048 validation examples, balanced across four categories. The target is at
least 5pp advantage over both ablations. The 10K→50K transition completed;
Stages 3–4 finish available AudioMCQ while recording audio gaps/regressions.
Test does not select checkpoints.

## API and web

```python
from samsone_clm import rank
ranking = rank("example.wav", "what instrument is playing?", ["piano", "guitar", "strings", "synth"])
# [{"choice": ..., "probability": ...}, ...], descending probability
```

Default checkpoint: `runs/v3/demo.pt` if available, otherwise `runs/latest.pt`.
Override with `SAMSONE_CLM_CHECKPOINT`, `samsone_clm.api.configure(checkpoint)`,
or a separate `samsone_clm.infer.Ranker(checkpoint)`.

```bash
uv run --no-sync samsone-clm-serve --checkpoint /srv/data/samsone_clm/runs/v3/demo.pt --port 8800
```

Open `http://127.0.0.1:8800`: upload, playback/seek, question, arbitrary choices.
Analyze takes 5 seconds before and after the position, padding to exactly
10 seconds. Results show ranking, position, window/padding, latency, checkpoint.
`scripts/publish_checkpoints.py` points latest to selected trained Stage 2–4
weights; the server reloads heads without reloading Samsone.

Training monitor: `http://127.0.0.1:8800/training`, refresh every 3 seconds;
telemetry about every 5 seconds. It shows actual LR/batch/epochs, train/validation
CE, Top-1/Top-2, ECE, mean P(correct), audio ablation, gradients, checkpoints,
and separate precompute progress. Stage selection is manual or automatic.
The PNG button exports selected-stage curves to `runs/stageN/training_curves.png`.

```bash
uv run python scripts/plot_training.py --stage 2
```

JSON: `/api/training`; PNG: `/api/training/plot.png?stage=2`. The monitor reads
artifacts without starting encoding/training.

Questions/choices must match Samsone's original lowercase ASCII normalization
and vocabulary. Head inputs and final vectors are L2-normalized. Scale starts
from the official checkpoint, approximately 100, is frozen in Stage 1, trained
in Stages 2–4, and subsequently calibrated on validation. Probabilities are
softmax over scaled cosine scores among supplied choices. Positive-scale
calibration preserves ranking. Legacy checkpoints retain scale=1.
No additional models are used.

Pinned revisions: SamsungLabs/samsone
`a71e77ae9528ef411ae26cdeb02d7b90cc18a50c`; Contrastive-LM/CLM
`bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`. Original modules: `vendor/`.
Historical [256-example](SPIKE_REPORT.md) and [1000-example](ITERATION_1000.md)
reports describe the initial adapter-only checks.
