# Autonomous v3 experiment

**Completion update:** Stages 3/4 completed, common was exported, and processes
were stopped. Final validation/test and separate continuation commands:
[PROJECT_STATE](PROJECT_STATE.md), [REMOTE_MIGRATION](REMOTE_MIGRATION.md).
Intermediate 50K numbers below are historical.

Artifacts: `/srv/data/samsone_clm/runs/v3/`. The user required full
AudioMCQ Stage 4 and immutable `samsone_clm_common`, followed separately by
Stages 5–6 → `samsone_clm_music`. The former evidence gate no longer blocks
Stages 3–4; regression is still a reported outcome.
[Detailed plan](COMMON_MUSIC_PLAN.md).
Status: `status.json`; final numbers: `REPORT.md` and `result.json`.
Plots: `training_comparison.png` and http://127.0.0.1:8800/training.

## Fixed architecture

Frozen Samsone356M produces hidden width 960. One shared linear adapter maps
960 → 4096, with L2-normalized output entering original CLM state/action heads.
Both 512-dimensional outputs are normalized. Ranking uses cosine; logits use
a positive scale. Stage 1 trains only the adapter; Stages 2–4 train adapter,
both heads, and scale. Qwen is not instantiated. Dimensions come from
checkpoints/configurations.

Generation is confined to the native diagnostic baseline, never training/API/
demo. Samsone computes features in FP32; storage uses BF16 shards. The encoder
is never called inside training.

## Completed checks

1. Native Samsone356M on 100 AudioMCQ validation examples, 25 per category:
   correct 50%, random 34%, silence 27%. The prompt includes choices and asks
   for an answer letter. Original outputs are retained. Sample count and
   prompts differ from ranking validation.
2. Production hidden matched hooks on official native prefill in four categories.
   Frozen weights and last-valid pooling were checked.
3. Adapter-only overfit 256 reached 100% training accuracy, CE 3.49 → 0.0062,
   with both frozen heads unchanged. Pure CLM InfoNCE also passed mechanical
   overfit at 99.61%.
4. Stage 1 trained on 10K. Stage 2 pilots compared CE, CE + masked bidirectional
   InfoNCE, and original choice CLM bidirectional InfoNCE with matching
   snapshots and initialization.
5. The 50K control completed; evaluation and the 150K scaling decision are in
   `scale_decision.json`. Failure of the old recipe does not establish impossibility.

The retained 50K CE checkpoint at step 8602 gives correct 62.65%, random 56.49%,
silence 56.74%, gaps 6.15/5.91pp, passing the overall validation audio gate.
Per-category correct/random/silence: music 74.61/63.87/64.06%, sound
59.77/57.42/58.40%, speech 68.95/59.96/56.84%, temporal 47.27/44.73/47.66%.
Temporal does not yet show measurable audio benefit. Validation permitted
Stage 3; this does not replace reserved test.

Validation-only calibration: scale 100 → 26.33, CE 1.542 → 0.906,
ECE 0.216 → 0.030, unchanged ranking. Paired bootstrap over 1834 audio clusters
(2048 QA) yields random-gap 95% CI [4.39,7.84]pp and silence-gap CI [4.17,7.55]pp.
Validation confidence intervals do not remove selection optimism on that split.

The provisional 50K API matched cached inference on four real WAV files across
categories, maximum probability error 1.18e-5. Upload/HTTP Range seeking and
start/end 10s padding passed. Central 10s windows change full-clip Top-1 in
12.5% of 40 cases: a stability measure, not crop-label accuracy.

## Loss and audio contrast

`ce`: per-example candidate CE. `infonce`: CE plus masked bidirectional InfoNCE,
with negatives limited to explicit distractors. `clm-infonce`: original CLM
choice bidirectional InfoNCE with a minibatch-wide candidate pool and multiple
positives for repeated answers, without additional main CE. All variants
share the same paired CE and swap-margin auxiliaries.

Every update explicitly samples an audio-pair minibatch: 32 pairs = 64 additional
rows. Partners have compatible questions/options but different correct answers;
columns align by text, not index. This removes the sparsity of pairs in random
minibatches. Pair labels also come from AudioMCQ. The overall pool covers 4955
rows; limited music/sound coverage remains a weakness.

Pure CLM loss canonicalizes action IDs using official Samsone normalization,
avoiding false negatives for identical normalized text. Auditing all 227353
usable rows found no within-question normalized-choice collisions.

AdamW, OneCycle, maximum 12 epochs, patience 4 validation evaluations.
Maximum LR: adapter 3e-4, each head 1e-4, scale 1e-3; weight decay 0.01;
main batch 64. Temperature is frozen in Stage 1 and trained in Stage 2+.
Checkpoints preserve optimizer, scheduler, RNG, and resume arguments.

## Data integrity

AudioMCQ: 571118 metadata rows, 257941 strong, 227353 usable: train 181592,
validation 22679, test 23082. PCM/source-identity splits and immutable nested
stage snapshots; fixed balanced validation/test of 2048 examples each.

A 201205-WAV hash audit found one TACOS checksum failure with a missing tail.
It was restored from original ZIP after published-MD5 and exact-PCM checks.
Three training QA rows were repaired in new immutable `processed/v3/`;
Stage 1 and all held-out rows were retained, as was the damaged original.
A verified prefix of 18432 states and 128538 actions was reused without
encoding again. WAV/checkpoint/JSON writes use temporary files, fsync, and
rename; WAV preparation locks out concurrent writers.

## Selection and scaling

Test does not select recipes/checkpoints. Among saved 50K best/last, selection
first requires at least 5pp advantage over both audio ablations, then maximizes
validation Top-1, minimum audio gap, and CE. The retained checkpoint is copied
into `*-audio-grounded/`; that is not a new training run. Original checkpoints
and metrics stay untouched. Stages 3–4 apply the same best-selection ordering.

The historical evidence gate required at least 0.5pp Top-1 improvement and
both audio gaps no worse than the pilot, with a 2pp floor. The updated user
instruction required Stages 3–4 to finish available AudioMCQ regardless;
gates/regressions remain recorded. Common exports best Stage 4 after at least
one complete pass.

After final selection: validation-only positive-scale calibration, reserved
correct/random/silence test, paired bootstrap by audio identity, live/cache
parity, and real-WAV web/window checks. Full clips and crops may have different
correct answers; assess window prediction stability without invalidly
transferring full-clip labels.

## Autonomous launcher and resource protection

```bash
cd ~/tools/samsone-clm
uv run --no-sync python -m samsone_clm.train_all --resume
```

Never change dependencies in an active training environment. The supervisor
refuses duplicate workers and reuses verified features/checkpoints. Jobs require
20 GiB available RAM/disk before launch and monitor a 12 GiB floor during work.
A watchdog logs `resource_history.jsonl` every 15 seconds and checks progress.
For unified-memory GPUs, monitor system MemAvailable. Low free memory with a
large reclaimable page cache does not itself establish OOM.

Supervisor/watchdog logs: `supervisor.log`, `watchdog.log`. Resource shutdown
signals only verified project workers. Historical unreadable NTFS JSON files
were preserved; no fsck was run on the mounted volume.
