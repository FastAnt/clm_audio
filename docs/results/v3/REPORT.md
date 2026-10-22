# Samsone-CLM v3 experiment

Architecture: frozen Samsone356M -> shared linear adapter -> original CLM state/action heads; cosine ranking with checkpoint/learned scale.

Native 100-example baseline: correct=50.0%, random=34.0%, silence=27.0%.
Native generation is a diagnostic only; training and ranking use no generation. Native baseline and 2048-row ranking validation have different sample counts and prompts.

| Run | Main train | Correct | Random | Silence | CE | Audio gaps (pp) |
|---|---:|---:|---:|---:|---:|---:|
| diagnostic-stage2-ce | 10000 | 57.32% | 52.69% | 52.20% | 2.1783 | 4.64 / 5.13 |
| diagnostic-stage2-clm-infonce | 10000 | 50.88% | 47.12% | 46.29% | 1.3723 | 3.76 / 4.59 |
| diagnostic-stage2-infonce | 10000 | 56.88% | 53.12% | 52.44% | 2.1747 | 3.76 / 4.44 |
| stage1 | 10000 | 56.01% | 53.42% | 53.96% | 2.0220 | 2.59 / 2.05 |
| stage2 | 50000 | 62.94% | 58.54% | 57.28% | 1.1975 | 4.39 / 5.66 |
| stage2-audio-grounded | 50000 | 62.65% | 56.49% | 56.74% | 1.5415 | 6.15 / 5.91 |
| stage2-clm-infonce | 50000 | 56.84% | 49.22% | 50.54% | 1.2005 | 7.62 / 6.30 |
| stage2-clm-infonce-audio-grounded | 50000 | 56.84% | 49.22% | 50.54% | 1.2005 | 7.62 / 6.30 |
| stage3 | 150000 | 65.19% | 60.50% | 60.21% | 1.1245 | 4.69 / 4.98 |
| stage4 | 181592 | 65.19% | 59.67% | 58.01% | 1.1428 | 5.52 / 7.18 |

50K retained-checkpoint selection is a separate artifact, not another training run.
Provisional 50K demo uses validation-only scale calibration. Validation CI and calibration are fitted/selected on validation; reserved test remains separate.
For the selected 50K CE checkpoint, music/speech show clear audio gains; sound is weaker and temporal has no measurable benefit. Passing the overall gate does not establish all-category success.

Provisional 50K: full-clip vs 10s center-window Top1 changed in 12.5% of 40 validation cases. This is stability, not crop accuracy.

Source audit checked 201205 unique WAVs. One TACOS WAV failed checksum and contained a missing tail.
It was restored from the original ZIP with its published MD5 and exact PCM verification. Three QA rows received corrected PCM IDs and duration. Stage1 and all held-out rows are unchanged.
A validated 18432-state prefix and all 128538 actions were recovered without rerunning Samsone. Original damaged artifacts were retained.

Detailed metrics, configuration, checksums, scale and head-change checks are in each run metrics.json.
Paired confidence intervals and reserved-test results are separate artifacts; they do not select the training recipe.

Masked bidirectional InfoNCE uses exact text IDs, multiple annotated positives and only explicit distractor relationships. CE remains part of the objective.
Stage1 freezes both heads and temperature; Stage2+ trains both heads, adapter and temperature. Encoder is never called during training.

Resource guard: >=20 GiB RAM/disk before jobs, supervised termination below 12 GiB during jobs; per-training checks every 100 updates.
Three old NTFS JSON entries could not be read. They were preserved; new experiment files use runs/v3. No mounted-volume filesystem repair was attempted.
