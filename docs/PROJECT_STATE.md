# Project state

Recorded state on 2026-10-04: FMA Large v2 completed and was accepted on
validation. `samsone_clm_music` was updated; common remained unchanged.
Selected checkpoints were published in Git LFS. All trained runs/exports were
backed up offsite without datasets/audio/features at the user's request.
[Final music report](results/music-genre/REPORT.md),
[models and backup](MODEL_BACKUP.md). Historical chronology follows below.

User-track audit, 2026-10-04: the web server reads the current FMA weights;
PCM matches a fresh original decode; extracted hidden is bitwise equal to
official native prefill; both heads were trained. Genre errors nevertheless
reproduce and depend strongly on question/choice wording. Native Samsone
answers pop on Umbrella while the ranker selects classic; native inference
also answers pop on No White Flag.
[Audit and comparison with original CLM](CLM_TRANSFER_REVIEW.md).

Following verified backup, the RunPod project datasets/audio/features,
weights/runs, caches/environment, and project processes were removed. Cleanup
left 499.61GiB free. At the user's request, the demo was moved to the original
workstation: `http://127.0.0.1:8800/`, CPU, common/music FMA Large v2 selection.
Local Samsone–CLM training was not resumed. These statements record the
2026-10-04 installation; they are not a live server-health report.

Remote update, 2026-10-04: Stage 5, Stage 6, and Stage 6 adaptation to 10-second
windows completed. The validation-selected `samsone_clm_music` was returned
locally. Model/run SHA256, 73,450 referenced musical WAV files, and 16,528 window
WAV files were verified. Remote training finished and the demo was available.
[Results](results/music-stage6-window/REPORT.md), [testing](TEST_MUSIC.md),
[recorded progress](REMOTE_PROGRESS.md). The detailed common snapshot below
is historical.

Historical immutable snapshot: AudioMCQ v3 completed on 2026-10-03, with export
at 20:52 UTC. Training and the demo were stopped at the user's request. Code
was consolidated into this Git checkout from `/opt/samsone-clm-backup`;
the original directory and large data volume were retained.

## Architecture and corrected contract

- Encoder: official Samsone356M, 356,883,264 parameters, frozen throughout.
- State: official audio/question prefill plus ` answer: `; final normalized
  transformer hidden at the last valid token, width 960.
- Action: choice text through the same frozen Samsone text transformer, width 960.
- One shared linear adapter, 960 → 4096, followed by L2 normalization before each head.
- Original pretrained CLM state/action heads; configuration comes from the
  official checkpoint. Outputs have width 512 and are L2-normalized.
- Logits: cosine × positive scale. Stage 1 freezes both heads/scale;
  Stages 2–4 train both heads, adapter, and scale.
- Trainable parameters in Stage 2+: 22,823,937. Qwen is never run; its original
  encoder name remains in checkpoint configuration only as provenance.

FP32 extraction removes BF16 compute batch drift; BF16 storage saves memory.
Native position IDs, upstream prompt normalization/prune map, and audio
preprocessing are preserved. Feature SHA256 is verified on load. Actual
training records `encoder_calls=0`.

Fixes include scale, L2 head inputs, swap-column alignment for reordered
choices, and sparse-pair sampling. Thirty-two verified pairs per update provide
CE and swap-margin supervision; the main loss is candidate CE. Pretrained
head topology is retained.

## Data

| Level | QA rows |
|---|---:|
| All AudioMCQ metadata | 571118 |
| `audio_contribution == strong` | 257941 |
| Available usable source audio ≤30s | 227353 |
| Train | 181592 |
| Validation | 22679 |
| Test | 23082 |

Usable categories: speech 133283, sound 55040, temporal 24405, music 14625.
Musical training contains 11623 QA rows, all already included in Stages 2–4.

Splits use source-recording identity with exact PCM duplicate groups, before
stage subsampling. Stage snapshots are nested; balanced evaluation is fixed
at 2048 validation and 2048 test QA. Full Stage 4 features contain **185688 state
rows** and **401701 choice texts**: 181592 train + 4096 held-out. Feature store:
`features/spike-a86e057eae695e43/`.

The source audit checked 201205 unique WAV files. One TACOS file with a damaged
tail was restored from the original ZIP after verifying the published MD5.
Three training QA rows and PCM identities changed; Stage 1 and held-out labels
and splits did not. [Repair report](results/data_v3/repair_report.json).

## Training

| Stage / selected artifact | Main QA | Selected step | Validation correct/random/silence |
|---|---:|---:|---|
| Stage 1 | 10000 | 1727 | 56.01 / 53.42 / 53.96% |
| Stage 2, audio-selected retained CE | 50000 | 8602 | 62.65 / 56.49 / 56.74% |
| Stage 3 | 150000 | 14064 | 65.19 / 60.50 / 60.21% |
| Stage 4 | 181592 | 8514 | 65.19 / 59.67 / 58.01% |

Stage 3: 10 epochs, 23440 updates, 22.48 minutes, early stopping. Stage 4:
7 epochs, 19866 updates, 15.66 minutes, best at epoch 3. These timings cover
training on cached features only; precompute takes substantially longer.

Stage 4 recipe: batch 64, maximum 12 epochs, patience 4, AdamW weight decay 0.01,
adapter/head/scale LR 3e-4 / 1e-4 / 1e-3, OneCycleLR, gradient clip 1.0,
pair batch 32, pair CE weight 0.5, swap weight 1.0 / margin 0.05, reciprocal-answer
pairs. Best selection first requires both validation audio gaps ≥5pp, then
ranks by Top-1/minimum gap/CE. Validation follows a complete training pass.

Native Samsone diagnostic on 100 examples: correct 50% / random 34% / silence 27%.
Overfit 256: training accuracy 100%, CE 3.49 → 0.0062; Stage 1 heads unchanged.
Native and ranking evaluations have different sample counts and prompts;
they are not directly comparable measurements.

## Reserved test and limitations

The following describes calibrated Stage 4 on 2048 balanced test QA, unused
for checkpoint selection or calibration:

| Type | Correct | Random | Silence | Random gap | Silence gap |
|---|---:|---:|---:|---:|---:|
| Overall | 62.79% | 59.03% | 57.42% | +3.76pp | +5.37pp |
| Music | 68.95% | 61.91% | 59.38% | +7.03pp | +9.57pp |
| Sound | 66.80% | 64.45% | 64.45% | +2.34pp | +2.34pp |
| Speech | 72.85% | 64.45% | 62.50% | +8.40pp | +10.35pp |
| Temporal | 42.58% | 45.31% | 43.36% | −2.73pp | −0.78pp |

Paired bootstrap over 1841 test audio clusters: random-gap 95% CI [2.20,5.25]pp;
silence-gap CI [3.66,7.05]pp. Audio has a positive overall effect, but the test
threshold of at least 5pp above both ablations **was not passed**. The
`audio_dependence_gate_passed` training/model-card flag refers to validation,
not all test categories.

Validation-only scale calibration: 100 → 37.8852. Test CE 0.8601, ECE 0.04317,
Top-2 82.18%, mean P(correct) 0.53832. Probabilities are relative to the supplied
candidate set.

Live/cache parity was checked on four real WAV files. Upload, HTTP Range
playback/seeking, and boundary-padded 10-second windows passed functional
checks. Full-clip versus central-10s Top-1 differs in 7.5% of 40 validation
cases; that measures stability, not crop accuracy. Temporal crop labels need
offset adjustment and evidence checks. Samsone's lowercase English ASCII/pruned
vocabulary constrains supported questions and choices.

## Preserved artifacts and next steps

Immutable model: `models/samsone_clm_common/{best.pt,last.pt,calibrated.pt,...}`.
Raw best is the parent for subsequent training; calibrated weights serve
inference. Separate dataset export: `processed/samsone_clm_common/`. Git
contains JSON cards/checksums/results/PNGs. Selected weights are now in Git LFS;
canonical audio remains external. The historical `common-2026-10-03` GitHub
release also backs up learned common weights.

Historical continuation plan: additional Stage 4 epochs on a faster remote GPU
create a separate run with fresh optimizer/schedule, preserving common and
original evidence. `train_all --resume` does not extend completed schedules.
The wrapper is `scripts/continue_stage4.py`. This continuation subsequently ran;
see the recorded remote results above.

Historical snapshot: Stages 5–6 had not been implemented when common was
exported. Musical stages and window continuation subsequently completed;
prepared audio/features/checkpoints were returned before the later cleanup.
Common stays immutable. The music model uses audio substantially, but tonic and
tempo remain weak and genre ranking is wording-sensitive.
[Final window report](results/music-stage6-window/REPORT.md),
[original plan](COMMON_MUSIC_PLAN.md).
