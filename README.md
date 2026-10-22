# Samsone–CLM Audio System 1

Rank candidate text answers to audio questions without generating a response.
This research project transfers pretrained CLM heads onto a frozen Samsone356M
encoder and investigates audio grounding, shortcut learning and music retrieval.

## Architecture

```text
audio + question → frozen Samsone356M → hidden → shared adapter → CLM state_head  → z_state
choice text      → frozen Samsone356M → hidden → shared adapter → CLM action_head → z_action

logits = positive_scale × cosine(z_state, z_action)
probabilities = softmax(logits across supplied choices)
```

The selected recipe uses a linear adapter (960 → 4096), L2-normalized head
inputs/outputs and pretrained 512-dimensional CLM projections. Dimensions are
read from checkpoint configuration. Stage 1 trains the adapter with frozen
heads. Subsequent stages train the adapter, both heads and logit scale. Samsone
is always frozen. Features are extracted once in FP32 and stored in sharded
BF16 files; training never invokes the encoder. No Qwen model is loaded.

## Results and limits

| Evaluation | Correct audio | Random audio | Silence |
|---|---:|---:|---:|
| General AudioMCQ reserved test, 2,048 QA | 62.79% | 59.03% | 57.42% |
| FMA Large genre balanced accuracy | 42.41% | 8.19% | 9.09% |

The general model fails the strict requirement of a 5 percentage point gap
above both audio ablations on test; temporal questions remain weak. Genre
results improve over the parent model but remain moderate and sensitive to
question wording. These results do not establish production reliability.
FMA genre test: 3,414 QA views of 569 independent recordings.

The FMA continuation used 44,968 usable recordings and 291,000 new training QA;
with replay, the mixture contained 404,473 QA. Training completed 32 epochs;
validation selected epoch 6. See [music results](docs/results/music-genre/REPORT.md),
[project state](docs/PROJECT_STATE.md) and [transfer review](docs/CLM_TRANSFER_REVIEW.md).

## Setup

Python 3.12+ and uv are required. The lockfile pins the tested dependencies;
GPU wheels target CUDA 13.0, so check your platform before installation.

```bash
git submodule update --init --recursive
export SAMSONE_CLM_DATA=/srv/data/samsone_clm
uv sync --locked
uv run python scripts/download_models.py
```

Check upstream model/source terms before downloading. Official source revisions
are pinned; encoder and reference-head weights are downloaded separately.

## Training and evaluation

```bash
uv run python scripts/download_data.py
uv run python scripts/prepare_data.py
uv run python scripts/precompute.py
uv run python -m samsone_clm.train --stage 1
uv run python -m samsone_clm.train_all --resume
uv run pytest
```

Perform the small overfit and audio-ablation checks before scaling a new run.
Historical recipes and later musical preparation are documented in
[documentation](docs/README.md). Source audio licenses and access vary.

## Inference and demo

Selected common/music inference checkpoints are included through Git LFS.
Restore them with `uv run python scripts/download_learned_models.py`.
See [checkpoints](checkpoints/README.md) for model files and verification.

```bash
uv run samsone-clm-serve --checkpoint /path/to/calibrated.pt --port 8800
```

Open http://127.0.0.1:8800/. Upload audio, choose a playback position, enter an
English question and candidates, and click Analyze. The demo ranks a 10-second
window centered on that position, padding boundaries with silence. It shows
probabilities, analyzed bounds, latency and checkpoint. `/training` displays
metrics from local run artifacts when available.

The Python API is implemented in [api.py](src/samsone_clm/api.py);
its `rank(audio, question, choices)` method returns sorted choices and probabilities.

## Checkpoints and licensing

The repository includes source code, tests, experiment reports and selected
inference checkpoints. Datasets, source audio and full training snapshots are
not included. [Artifact documentation](docs/ARTIFACTS.md) describes the
checkpoint format and provenance checks.

Project-owned code/docs: [Apache-2.0](LICENSE). Third-party components and model
weights retain separate terms: [NOTICE](NOTICE), [THIRD_PARTY](THIRD_PARTY.md).
Built on [SamsungLabs/Samsone](https://github.com/SamsungLabs/samsone) and
[Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM).
