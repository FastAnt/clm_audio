# New iteration: 1000 samples — October 3, 2026

This run completed with the same architecture: frozen Samsone-356M, the SAME
trainable state/action adapter, and original official CLM state/action heads.
It is an independent Stage 1 overfit from original head initialization:
seed 42, LR 0.001, batch 64, AdamW without weight decay. Qwen was not used.
Both heads were frozen; score was literal cosine with no temperature scaling.

Training grew from 256 to 1000 examples; validation/test grew from 30 to 128
each. All are strong AudioMCQ/Clotho. Six Clotho development shards supplied
3839 audio files; complete Clotho strong metadata contains 1999 questions.
Splits use decoded PCM identity: 1256 selected examples have 1256 unique audio
identities. All 316 earlier examples occur in this cohort. No cross-split audio
overlap or rejects were found.

Training categories: sound 947, speech 31, music 22. Validation: sound 123,
speech 4, music 1. Test: sound 123, speech 3, music 2. Temporal is absent.
This is an AudioMCQ subset experiment, not an all-source/category evaluation.

Precompute: 4640 unique choices and 1256 states per audio mode, BF16 width 960,
sharded safetensors. Reused 1230 actions, 316 correct states, 316 silence states,
and one matching random audio/question pair after checking old provenance and
checksums. Only missing pairs were computed. Random pairing was rebuilt within
each split without identical audio identities. Precompute took 348.6 seconds
excluding model loading. Frozen parameters: 356883264; nonfinite count: 0;
mean correct/silence hidden L2 difference: 8.26378.

The overfit criterion stopped training at 1800 steps after 33.1 seconds,
excluding feature/model loading. As appropriate for diagnostic overfit, best
was selected by training loss. Training never called Samsone. Initial adapter
gradient norm was 0.53652; both heads remained bitwise equal to initialization.

| Training metric | Before training | Best, step 1800 |
|---|---:|---:|
| CE | 1.385126 | 0.381338 |
| Top-1 | 26.0% | 99.2% |
| Top-2 | 52.9% | 99.5% |
| Mean p(correct) | 0.250394 | 0.689966 |
| ECE, 15 bins | 0.002918 | 0.301974 |

CE fell 72.47%; the mechanical overfit gate passed. Late-step losses fluctuated
but stayed finite; final values describe the best checkpoint. Four cosine
logits have theoretical CE floor about 0.34075.

| Audio mode | Train Top-1, 1000 | Validation Top-1, 128 | Validation CE |
|---|---:|---:|---:|
| Correct | 99.2% | 50.0% | 1.190917 |
| Random | 99.2% | 50.0% | 1.190918 |
| Silence | 99.1% | 50.0% | 1.190925 |

Validation Top-2 is also identical at 72.65625%. Correct-audio mean P(correct)
is 0.361251 and ECE 0.108649. Per-category results are in metrics.json; tiny
music/speech cohorts do not support reliable category conclusions. Reserved
test was not used for checkpoint selection.

Audio ablation failed again: correct−random and correct−silence validation
accuracy gaps are zero. Under the specification, this iteration was unsuccessful.
Metrics record `overfit_gate_passed=true`, `audio_dependence_gate_passed=false`,
and `ready_for_large_training=false`. This is adapter-only Stage 1 on Clotho;
subsequent retraining of both heads had not yet been tested in Stages 2–4.

For a matched comparison, the earlier 256-example checkpoint was evaluated on
THE SAME 128 validation identities. Encoder provenance and the absence of
validation overlap with either checkpoint's training were verified:

| Checkpoint | Correct Top-1 | Random Top-1 | Silence Top-1 | Correct CE |
|---|---:|---:|---:|---:|
| 256 train | 49.21875% | 49.21875% | 48.4375% | 1.267442 |
| 1000 train | 50.0% | 50.0% | 50.0% | 1.190917 |

Correct accuracy improved by one answer out of 128, or 0.78125 percentage points.
This does not establish substantial progress or audio dependence. Earlier 40%
results used 30 validation examples and cannot be compared directly with 50%
on 128 examples.

Real-WAV inference passed: maximum live/cached probability difference 5.85914e-5,
below BF16/TF32 tolerance 1e-4. Five new choices passed functional checks, not
an accuracy benchmark. All 7 unit tests and the uv.lock check passed.

Artifacts:

- Best checkpoint (`/srv/data/samsone_clm/runs/overfit-1000-seed42-lr0.001/best.pt`; external historical artifact)
- Last checkpoint (`/srv/data/samsone_clm/runs/overfit-1000-seed42-lr0.001/last.pt`; external historical artifact)
- Metrics and ablations (`/srv/data/samsone_clm/runs/overfit-1000-seed42-lr0.001/metrics.json`; external historical artifact)
- Loss/accuracy plot (historical image omitted from this text-only export)
- Matched-validation comparison (`/srv/data/samsone_clm/runs/overfit-1000-seed42-lr0.001/comparison_previous.json`; external historical artifact)
- Inference verification (`/srv/data/samsone_clm/runs/overfit-1000-seed42-lr0.001/inference_proof.json`; external historical artifact)
- Training log (`/srv/data/samsone_clm/runs/overfit1000.log`; external historical artifact)
- Immutable feature manifest (`/srv/data/samsone_clm/features/spike-97b6f2ce7df94628/manifest.json`; external historical artifact)

Iteration commands were recorded in the historical README. Re-evaluation must
use the checkpoint's matching `--feature-manifest`. Earlier features and
checkpoints remain separately preserved.
