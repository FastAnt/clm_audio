# Music model and serving-window adaptation

Stages 5/6 completed on the same RunPod RTX3090. The validation-selected music
model was published locally/remotely as `models/samsone_clm_music` at this
historical milestone; the later FMA genre export supersedes that selection.
Raw best: `c9c00f307c12828eee9e66893bd3e7f4c6eba3a2a477d8e4ff4bdaafeed7f694`.
Inference checkpoint: `calibrated.pt`, SHA256
`d6219a325a26a45b9e2c7abe2eef1f7e68adb44cc58529287d1e5cc0485fb2c7`.

## Architecture and training

Frozen Samsone356M → SAME linear adapter 960→4096 → original pretrained CLM
state/action heads, with input/output L2, scale×cosine, and candidate ranking.
Both heads and adapter trained; `encoder_calls_during_training=0`. Qwen is unused.

Stage 5: 100424 musical training QA. Stage 6: 190653. Final Stage 6 window
adaptation retains this size: 44547 NSynth/Jamendo QA replace their full
representations with exact central-10s web `audio_window` representations;
remaining full representations stay. Training identities/splits/choices/labels
are preserved. NSynth 4s notes remain intact with silence padding; whole-track
Jamendo labels on 10s remain weak supervision. Samsone runs only in separate
FP32-compute/BF16-shard precompute.

Overfit 512 on the exact new manifest: 100% training accuracy, CE 0.0000586.
Parent: raw full32 Stage 6 best
`18d568e067ea0a6bf4497e415284afdb11c89cce03dbeb6e20b3680701ad397e`.
The first 32-epoch adaptation, LR 1e-4/5e-5, selected epoch 2; later epochs
hurt structural validation. That run/checkpoints remain separately preserved.

Final run `runs/music-window/stage6-window-lowlr`: 32 epochs, 23840 updates,
batch 256, 128 verified pairs/update, AdamW/OneCycleLR, maximum LR adapter 3e-5,
each head 1.5e-5, scale 3e-4, weight decay 0.01, clip 1,
CE + 0.5 pair CE + 1 swap loss, margin 0.05. Selected epoch 8 / step 5960.
Training commit: `fe3339b`.

## Validation selection

Selection requires audio gaps ≥5pp in every measured view/family, no more than
1pp full-family regression versus parent, and window macro improvement ≥1pp;
then it maximizes mixed balanced validation. Test and HTTP results do not
select the version.

The ≥50 task-support threshold counts original QA once: full/crop views do not
double support. This does not establish statistical independence among all
notes/augmentations. The previous checkpoint was re-evaluated with the same
corrected metric/cache separately from its immutable training reports.

| Balanced validation | Parent full32 | First window | Selected lower LR |
|---|---:|---:|---:|
| Full perception | 83.12% | 82.38% | 82.73% |
| Full structure | 43.89% | 43.05% | 43.47% |
| Serving-window perception | 69.74% | 78.83% | 79.34% |
| Mixed equal-family macro, corrected support | 60.16% | 61.83% | 62.25% |

Mixed macro describes the composed cohort and is not directly comparable with
full-only overall accuracy. [Version decision](window_refinement_selection.json)
retains both versions, baseline, and results. Calibration fits one scale on
validation only.

## Fixed test after selection

| View | QA rows | Correct audio | Random audio | Silence |
|---|---:|---:|---:|---:|
| Full clips | 3068 | 59.84% | 35.04% | 34.71% |
| Paired central-10s/padded notes | 1916 | 74.69% | 37.89% | 39.77% |

Window gaps: +36.80/+34.92pp; Top-2 88.47%, CE 0.7237, ECE 0.0451.
Full gaps: +24.80/+25.13pp; Top-2 77.38%, CE 1.0137, ECE 0.0453.
Balanced perception: full 80.84%, windows 80.15%; full structure 40.43%.
Window evaluation covers NSynth/Jamendo perception, **not** a separate
10s MusicBench structure test. These 1916 QA are paired views of original
full held-out QA, not 1916 new independent test examples.

Full-test accuracy fell from the preserved full32 model's 61.57% to 59.84%.
Validation retention does not guarantee test retention. Full32 and first-window
checkpoints are retained; test does not reopen version selection.
Audio dependence is substantial, but quality is not uniformly high:
full instrument family 54.72%, source 52.60%, vocals 92.97%, mode 60.16%,
tonic 13.67%, meter 57.81%, tempo 30.86%. Key/tempo remain weak. Genre cohorts
of 22–49 QA do not establish broad genre recognition.

A user MP3 additionally exposed question/option wording sensitivity. The original
failure reproduces and the new version does not fix it. New sources supplied
only 1,483 explicit genre QA on 793 audio identities; some AudioMCQ replay also
asks genre/style. Semantically equivalent questions change the top genre even
with a fixed full candidate set. [Diagnostic](user_genre_diagnostic.json),
[next plan](../../MUSIC_GENRE_FOLLOWUP.md).

Full-clip quality (historical image omitted from this text-only export)

10-second-window quality (historical image omitted from this text-only export)

## Artifacts and verification

- `models/samsone_clm_music/`: selected best/last/calibrated, metrics/card/checksums/PNG at this milestone.
- `models/samsone_clm_music_full32/`: preserved pre-window full-excerpt version.
- `models/samsone_clm_music_window/`: first window adaptation.
- `models/samsone_clm_music_window_lowlr/`: immutable source of this export.
- `runs/music-window/stage6-window-lowlr/`: optimizer/scheduler/RNG and resume history.
- `features/music-stage6-6e48fecc8cce6e96/manifest.json`: exact composed frozen feature store.

All 12 run and 9 export files per window version were returned locally and
SHA256-verified. Canonical musical QA, frozen features, and 73450 source WAV
files were returned; 16528 window WAV files were checked separately. Measured
source/PCM split leaks: 0. Immutable common raw hash stayed unchanged. Source
archives remained remotely at this historical milestone, before later cleanup.

Live Ranker checked 19 real held-out files; maximum live/cache probability
difference was 0.000150. HTTP checks passed 9 real instrument/vocal/tempo
windows: upload, Range seek, exact 10s, padding, candidate permutation, and
selected SHA. These are functional checks, not crop-label accuracy.
All 61 pytest tests passed on both machines. GPU training completed and the
demo was left running at this point.

Git stores small evidence/JSON/PNGs. At this snapshot, large weights lived
under the data root; selected exports were subsequently published through
Git LFS. Checksum lists retain their original large-file references.
[Testing](../../TEST_MUSIC.md), [prepared examples](test_examples.json).
