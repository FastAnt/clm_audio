# Audio contrast repair: controlled pilots

The inference architecture is unchanged: frozen Samsone-356M supplies state and action representations, one shared adapter maps its width to the original CLM head input width, and both pretrained CLM heads are trained in Stage 2. Head inputs and outputs are L2-normalized. The official checkpoint score scale is 100. Qwen is never instantiated.

## First controlled pilot: identical question and option set

The immutable base contains 10000 main training rows and the same 2048 validation and 2048 test rows as corrected Stage 1. Verified partners were selected from the full canonical TRAIN split only. There are 1186 directed partner rows; 1113 were not already in the base and were appended with `training_role=contrast_only`. Source-recording and decoded-PCM intersections with the complete canonical held-out splits are both zero.

Feature extraction reused 14096 state features and 38250 action features. Only 1113 state features and 65 action features were newly computed. Every held-out random/silence representation was reused. Computation is FP32; stored shards are BF16. Extraction took 177.37 seconds.

The run `runs/v2/diagnostic-stage2-10k-dense-swap` starts from `runs/v2/stage1/best.pt`. Each of 1884 optimizer updates uses 64 main examples plus 32 verified pair anchors and their partners (64 auxiliary rows). The model sees 120576 pair-row occurrences, rather than relying on rare partners in randomly drawn main batches. Both CLM heads changed; training invoked the frozen encoder zero times.

Settings: 12 epochs, AdamW, adapter LR 0.0003, head LR 0.0001, zero weight decay, global gradient clipping at 1. Main CE has weight 1, pair CE weight 0.5, swap loss weight 1. Swap margin is 0.05 in cosine units; scaled scores are divided by the checkpoint scale before computing the margin. Each swapped state is scored against the original row's candidates in their original order.

| Checkpoint | Correct audio Top-1 | Random | Silence |
|---|---:|---:|---:|
| Corrected Stage 1 | 54.64% | 53.96% | 53.17% |
| Prior Stage 2 pilot | 55.08% | 53.13% | 52.54% |
| Prior sparse swap pilot | 54.93% | 53.42% | 54.05% |
| Dedicated pair-batch pilot | 56.20% | 53.81% | 53.52% |

The new pilot's audio gaps are 2.39 and 2.69 percentage points, so it fails the required five-point gate. Its validation Top-2 is 78.32%, CE is 2.0477, mean correct probability 54.73%, and ECE 0.2840. Main-train Top-1 is 94.67%, CE 0.1409. These results do not establish successful audio generalization.

On the 1186 verified training partner rows, correct-audio accuracy is 92.33%; after swapping audio, accuracy against the original answer is 1.26%. The average correct-answer cosine gap is 0.08654; 80.19% of rows meet the 0.05 margin. These are training measurements and can reflect memorization.

Layer diagnostics on validation show increased sensitivity, not a complete loss of audio perturbations:

| Quantity: mean relative L2 change | Random audio | Silence |
|---|---:|---:|
| Frozen Samsone hidden | 0.1444 | 0.1830 |
| Dedicated-pilot adapter output | 0.1683 | 0.2134 |
| Dedicated-pilot state z | 0.2502 | 0.3065 |
| Prior Stage 2 pilot state z | 0.1604 | 0.2100 |

Random audio changes 273/2048 predictions: 125 correct answers are lost and 76 gained. Silence changes 319: 142 are lost and 87 gained. Sensitivity alone does not guarantee a correct audio-dependent answer.

Category gaps (correct minus random / silence): music +6.45/+6.45 points, sound -1.37/+0.20, speech +4.30/+4.88, temporal +0.20/-0.78. The verified training rows are highly skewed: speech 718, temporal 463, music 3, sound 2. Only eight validation rows match a trained contrast's exact normalized question and complete option set; this identifies limited exact-template coverage, not proof of a causal explanation.

## Expanded verified-pair diagnostic

The exact-option-set requirement is unnecessarily restrictive. The additional `reciprocal-answers` mode still requires the same normalized question and different recordings and PCM. For original correct answers A and B, BOTH option sets must contain BOTH A and B. Each annotation therefore explicitly establishes that the other's correct answer is a distractor. The remaining distractors may differ. Duplicate normalized choices are excluded. No arbitrary random-audio negative is assigned a guessed label.

This produces 4955 eligible train rows: speech 3968, temporal 969, music 14, sound 4. The expanded immutable pilot keeps the same 10K main training rows and held-out rows, adding 4733 auxiliary-only rows. It uses the same initialization and optimizer/loss settings as the first pilot. Its distinct run directory is `runs/v2/diagnostic-stage2-10k-reciprocal-swap`.

The expanded precompute reused 15209 state features and 38315 action features, newly computing 3620 states and 884 choices in 420.90 seconds. Both held-out ablations were reused completely. The expanded run completed 12 epochs in 192.85 seconds, with 1884/1884 updates receiving pair batches and both heads changing. The best checkpoint is step 314 (epoch 2), selected without consulting test labels.

| Expanded pilot validation metric | Result |
|---|---:|
| Correct audio Top-1 | 56.30% |
| Random audio Top-1 | 53.32% |
| Silence Top-1 | 53.27% |
| Correct minus random | +2.98 percentage points |
| Correct minus silence | +3.03 percentage points |
| Top-2, correct audio | 78.37% |
| CE, correct audio | 1.5038 |
| Mean correct probability | 52.00% |
| ECE | 0.1956 |

The five-point audio gate again failed. Category gaps (random / silence) are music +4.10/+4.69 points, sound +0.39/-0.39, speech +6.45/+4.88, temporal +0.98/+2.93. On the 4955 reciprocal training rows, the expanded best checkpoint achieves 74.63% correct-audio accuracy versus 15.36% accuracy against the original label after audio swap. These are training results, not independent proof of generalization. The expanded best checkpoint preserves audio perturbations in z_state (relative L2 0.1457 random / 0.1847 silence), while the more heavily fitted first pilot has greater sensitivity but poorer validation calibration. Sensitivity magnitude alone is therefore not a model-selection criterion.

The repair removes sparse delivery of the contrast objective: every update now receives verified positives and negatives. It does not establish a successful Audio System-1. The remaining evidence is consistent with overfitting and limited coverage of acoustic tasks in the verified pair pool, especially sound and music. It does not prove frozen Samsone is incapable of the task. Larger training was not launched after these failed gates.

Run/resume the expanded pilot and its automatic evaluation/audit:

```bash
uv run --no-sync python scripts/run_audio_contrast_pilot.py
```

If its audio gate passes, the supervisor continues the prescribed 50K, 150K and all-train stages. The main stage starts from the previous main stage's best checkpoint. All verified reciprocal partners are retained as separately counted auxiliary rows without changing primary snapshot membership or held-out data. If the gate fails, larger training is not launched and the supervisor records a completed diagnostic with the actual metrics.

Artifacts are under `/srv/data/samsone_clm/runs/v2/`: `audio_contrast_result.json`, `audio_contrast_curves.png`, `audio_contrast_signal_audit.json`; expanded-run files use the `_reciprocal` suffix. Each run also contains `best.pt`, `last.pt`, `metrics.json`, `progress.json` and an append-only log. Frozen data snapshots, pair inventories and split-audit reports are under `processed/v2/`.
