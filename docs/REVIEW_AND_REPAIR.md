# Samsone-356M verification and combined review — October 3, 2026

The working model is **Samsone356M**. The 134M check is retained only as history
of the initial README example; after the user's clarification, all subsequent
runs and the main project use 356M. Qwen is not run or downloaded.

## Official API reproduced directly

The original package is installed editable from `vendor/samsone`, without code
changes or reinstalling PyTorch/CUDA dependencies. Direct import works:
`from samsone import Samsone356M`. The default `~/.cache/samsone` symlinks to
`/srv/data/samsone_clm/cache/samsone`. The existing 356M checkpoint
is reused through a symlink, with SHA256 matching the verified official release.

```bash
cd ~/tools/samsone-clm
# When restoring the environment after uv sync:
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e vendor/samsone

# Minimal example using only the official API:
uv run --no-sync python scripts/official_samsone_example.py

# Reproducible audit with JSON, latencies, and multi-clip semantics:
uv run --no-sync python scripts/reproduce_samsone.py
```

The minimal script constructs **`Samsone356M()` without arguments** and invokes
it with one path and a list of two paths. Adapter/CLM are not involved.
`--no-sync` preserves the active environment; installing the official package
above does not re-resolve its training/mobile dependencies.

Observed native 356M results on CUDA/BF16:

| Input / prompt | Observation |
|---|---|
| `cow.wav`, `Caption the audio` | Describes a cow and calf mooing |
| `water.wav`, `Caption the audio` | Fragmentary `the audio`; this prompt fails |
| `water.wav`, `describe` | Describes rhythmic raindrops |
| `[cow.wav, water.wav]`, `what is the difference between audio files?` | Distinguishes animals in the first clip and flowing water in the second; adds unverified details and is truncated at 100 tokens |
| `[cow.wav, water.wav]`, `compare audio files` | Distinguishes mooing and rain; includes the unverified phrase `mooing machine` |

Hooks confirm that a list sends **two clips** through the audio encoder but
**one sample** through the LM, returning **one string**, rather than a batch
of independent questions. An FP32 diagnostic is also retained; it did not fix
the fragmentary water caption and changed some generated details. The API
and discrimination between these sounds work, but reliable native QA on
arbitrary audio is not established.

Reports under the data root: `runs/samsone_readme_356m/results.json`,
`runs/samsone_readme_356m_float32/results.json`, and
`runs/official_samsone_example_356m.log`. The earlier
[hidden-path audit](SAMSONE_AUDIT.md) remains relevant.

## Findings confirmed by both reviews

1. **The historical ranker failed the audio gate.** On the same 2000 validation
   rows, correct/random/silence accuracy is 62.8% / 62.8% / 62.55%. Raw hidden
   reacts to audio; final predictions rarely change. This supports a text-prior
   explanation but does not prove one cause or impossibility.
2. **Input L2 normalization was missing.** Upstream supplies unit-L2 embeddings.
   Our v1 adapter outputs have norms 64.11 (state), 64.33 (action). On the first
   512 validation rows, mean absolute state-head preactivation is 119.91;
   unit-L2 input reduces it to 1.87. The proportion with absolute GELU derivative
   <0.01 falls from 57.89% to 6.29%. Action values: 134.41 → 2.12 and
   62.84% → 13.37%. This violated the pretrained-head contract and was an
   implementation omission.
3. **Checkpoint logit scale was omitted.** Official `logit_scale=4.6132488`;
   upstream applies `exp(logit_scale).clamp(max=100)`. V1 used scale 1.
   The previously documented probability ceiling and CE floor describe our
   raw cosine-softmax implementation, not standard CLM.
4. **Stage subsets were not nested and held-out cohorts changed.** Only
   8648/10000 Stage 1 training rows appear in Stage 2. Validation overlaps
   355/1000 and test 349/1000. These rows keep their original splits in the
   full corpus. This is unstable top-N snapshot selection on a growing corpus,
   not established leakage. Stage 2 improved 51.95% → 62.8% on its own fixed
   validation cohort; direct comparison with Stage 1 validation is invalid.
5. **Precision and position IDs need controlled checks.** Single-example native
   hidden matches bitwise. Different prompt lengths in a batch cause 0.774%
   relative L2 discrepancy; native position IDs restore equality. An earlier
   batch audit measured 4.53% relative L2 in BF16 versus about 8.45e-6 in FP32.
   Storing BF16-computed results in FP32 does not remove compute error.
6. **Label noise and web-window mismatch exist.** One music-label conflict was
   verified against original source tags and bitwise PCM. Full-clip quality
   does not measure local 10-second-question quality.

Read-only head audit: `scripts/audit_clm_contract.py`, report
`runs/review_clm_contract.json`. Snapshot proof:
`runs/review_snapshot_comparison.json`. CUDA/TF32 settings match the historical
training/evaluation recipe.

Causal limits matter: a large positive GELU input is not a dead saturated
neuron; its derivative is close to 1. Small derivatives mainly concern the
negative region. Gradient norms across parameter groups do not establish
that heads were untrained; both heads measurably changed. However, pretrained
initialization received an incorrect input scale.

Post-hoc normalization/temperature checks on old weights are not retraining.
A positive scale **does not change argmax**. Applying scale 100 to old weights
raises CE to roughly 25 from 1.026, magnifying incorrect confidence. That is
not an inference-only repair of audio use. New training logits require fresh
overfit, validation, and audio-gate checks.

## Repairs within the fixed architecture

```text
audio + question -> frozen Samsone356M -> h_state -> SAME adapter -> L2 -> original state_head -> L2 -> z_state
choice text      -> frozen Samsone356M -> h_action -> SAME adapter -> L2 -> original action_head -> L2 -> z_action
similarity = cosine(z_state, z_action)
logits = min(exp(checkpoint.logit_scale), 100) * similarity
```

L2 and score calibration restore CLM's input/output contract without another
model or adapter. Initially fix scale to the checkpoint value. Stage 1 trains
only the adapter; Stages 2–4 train adapter and **both heads**. Samsone stays
frozen. Heads are retrained on Samsone representations rather than preserving
the old embedding space.

Sequence for the corrected experiment:

1. Fix the complete canonical corpus/checksum, immutable validation/test,
   and nested 10K ⊂ 50K ⊂ 150K ⊂ 181592 training subsets. Ensure sufficient
   representation of all four categories; report micro and macro metrics.
2. Introduce a separate encoder recipe with native position IDs. Compare FP32
   and BF16 compute on the same held-out cohort; retain BF16 sharded storage.
   Record FP32 selection in provenance; never mix v1/v2 features, checkpoints,
   and inference recipes.
3. Run corrected overfit on 256–1000 training examples with unit-L2 inputs and
   checkpoint scale. Check CE, Top-1, finite gradients, actual relative updates,
   and both head-input norms. An old best checkpoint cannot initialize a clean
   Stage 1 with the corrected contract.
4. Separate memorization and audio-use tests. Train a silence-feature control
   on the same rows, architecture, and optimizer budget to measure text-only
   strength. Native free-form answers and the external review's 9/40 word-overlap
   score are not reliable MCQ accuracy benchmarks.
5. Test identical questions/choices on different audio with different correct
   labels, balancing within groups. The full training inventory already has
   201 groups / 1186 rows. It is not an audited balanced 1000-example cohort:
   some groups are highly imbalanced and labels/evidence need checking.
   Silence should not win this task through question→choice priors.
6. If corrected plain CE retains the shortcut, add supervised audio-swap
   contrast on verified incompatible pairs: the correct candidate should score
   higher with its matching audio. Exclude false negatives. Do not assign the
   original positive label to random/silent audio, which would encourage
   audio-independent answers. Auxiliary loss leaves inference topology intact.
7. Scale only after that pilot. Use matching held-out ablations and paired
   confidence intervals at every stage, plus audio-hard question controls.
   Select LR/epochs through a short validation sweep, not two gradient norms.
   Test 10s web windows on questions with evidence inside the window.

These are causal tests, not guarantees that L2/temperature remove shortcuts.
They do not require a different encoder or architecture.

## Historical background jobs

Automatic v1 Stage 3/4 launch was suspended: supervisor PID 1547524 received
SIGSTOP; `runs/pipeline_hold.json` and `pipeline_status.json` record the reason.
Full precompute continued saving shards without data deletion. Old checkpoints
remain evidence of failed audio ablation. Suspending the supervisor does not
stop Samsone precompute or the inference server. Do not send SIGCONT before
configuring the corrected experiment: the old supervisor would resume old
stages automatically.

New review scripts did not alter production weights, old labels, or checkpoints.
This paragraph records the state at the review date, not current processes.
