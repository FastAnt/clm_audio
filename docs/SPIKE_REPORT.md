# Implementation spike — October 3, 2026

The approved architecture was implemented without Qwen: both branches use one
frozen Samsone-356M, the SAME shared adapter, and original CLM heads initialized
from the official checkpoint. Historical scoring used cosine directly, with
CE over cosine logits and no temperature/checkpoint logit scale.

Mechanical overfit on 256 samples passed. Audio dependence was not established:
correct/random/silence yielded identical accuracy. Large training had not
started. Transfer of pretrained CLM heads to frozen Samsone remained an open
hypothesis. A 100% training score established adapter memorization on this
cohort, not audio understanding or generalization.

| Measurement | Before training | After 300 steps |
|---|---:|---:|
| Train CE | 1.384619 | 0.425481 |
| Train Top-1 | 27.34% | 100% |
| Train Top-2 | 53.13% | 100% |
| Mean p(correct) | 0.25054 | 0.66661 |
| Train ECE, 15 bins | 0.01486 | 0.33339 |

Loss decreased approximately 69.3%. For four literal cosine logits the absolute
theoretical CE floor is about 0.34075, so the earlier loss <0.2 criterion does
not apply. High ECE means this overfit checkpoint's probabilities are not
calibrated confidence.

| Audio mode | Train Top-1, 256 | Validation Top-1, 30 | Validation CE |
|---|---:|---:|---:|
| Correct | 100% | 40% | 1.382156 |
| Random | 100% | 40% | 1.382278 |
| Silence | 100% | 40% | 1.382245 |

Accuracy gaps are zero, failing this run's specified audio-ablation criterion.
Loss/probability differences are tiny. This small validation cohort cannot
establish results for all AudioMCQ or show that subsequent training of both
heads must fail.

Metrics explicitly record `overfit_gate_passed=true`,
`audio_dependence_gate_passed=false`, and `ready_for_large_training=false`.
Test was not used for model selection.

Data: strong AudioMCQ, Clotho development source. Audio came from the
`gijs/clotho` transport mirror; annotations exclusively from
`inclusionAI/AudioMCQ`. The first source shard contains 640 audio files;
256 train + 30 validation + 30 test examples were prepared. All 316 have
distinct decoded PCM identities and no split overlap. Categories across the
spike: sound 299, music 10, speech 7, temporal 0. This does not cover all AudioMCQ.

Precompute produced 1230 unique choice-text representations and 316 state
representations per correct/random/silence mode. BF16 features were saved in
35 safetensors shards. Initial precompute took about 106 seconds excluding
model/data loading. A repeat confirmed reuse without frozen-encoder recompute.
Dataset cache keys use decoded PCM identity rather than irrelevant WAV-header
timestamps; waveform equivalence for the key change was checked separately.

Dimensions were read from actual configurations/weights: Samsone hidden 960,
CLM head input 4096, projection 512. One adapter has 4,868,928 parameters.
Samsone has 356,883,264 frozen parameters. Its internal audio/text components
were unchanged; the text-only branch uses the same checkpoint's weights.

Real-WAV checks:

- Repeated state extraction is deterministic.
- Replacing audio with silence changes hidden; mean L2 difference over 316 samples is 8.134.
- Changing the question changes hidden; fixture L2 difference is 19.561.
- State shape `[1,960]`, four actions `[4,960]`; no nonfinite values.
- Gradients propagate through both frozen heads into the SAME adapter.
- Both CLM heads are bitwise unchanged from original initialization after overfit.
- Training reads feature files without calling Samsone.
- Four unit tests cover shared adapter, stage freezing, gradients through both
  branches, candidate permutation, and left/right-padding pooling.
- Live `Ranker.rank(audio, question, choices)` matches cached ranking on a real
  WAV: maximum probability difference `6.24e-5`, within BF16/TF32 tolerance `1e-4`.
- Five new arbitrary choices traverse the frozen Samsone text path. This checks
  functionality, not answer accuracy on unseen candidates.
- `uv lock --check`, repeated preparation, precompute reuse, and separate
  evaluation commands passed.

Stage 2 configuration passed a gradient test for trainable adapter and both
original heads. Stages 2–4 were not actually trained in this spike. The demo
and large training stages remained subsequent work.

Artifacts:

- Best checkpoint (`/srv/data/samsone_clm/runs/overfit-256-seed42-lr0.001/best.pt`; external historical artifact)
- Last checkpoint (`/srv/data/samsone_clm/runs/overfit-256-seed42-lr0.001/last.pt`; external historical artifact)
- Full metrics and ablations (`/srv/data/samsone_clm/runs/overfit-256-seed42-lr0.001/metrics.json`; external historical artifact)
- Loss/accuracy plot (historical image omitted from this text-only export)
- Extraction proof (`/srv/data/samsone_clm/runs/extraction_proof.json`; external historical artifact)
- Inference proof (`/srv/data/samsone_clm/runs/inference_proof.json`; external historical artifact)
- Source/data provenance (`/srv/data/samsone_clm/raw/download_report.json`; external historical artifact)

Reproduce from `/opt/samsone-clm-backup`:

```bash
uv run python scripts/download_models.py
uv run python scripts/download_data.py
uv run python scripts/prepare_data.py
uv run python scripts/precompute.py
uv run python -m samsone_clm.train --stage 1 --overfit --samples 256
uv run python scripts/verify_inference.py
uv run pytest -q
```

Seeds, arguments, feature provenance, original-head metadata, and optimizer
state are saved with checkpoints. `uv.lock` pins the environment; all large
files, caches, and `.venv` live on the data disk. This is the historical recipe;
later v3 fixes restored input normalization and checkpoint scale.
