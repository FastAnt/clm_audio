# Stage 6 genre continuation: FMA Medium and FMA Large

FMA Large v2 completed on 2026-10-04. Architecture remained unchanged:
frozen Samsone356M → SAME linear adapter 960→4096 → original CLM state/action
heads, output 512 → L2/scaled cosine. Both heads and adapter trained; encoder
calls during training: 0.

FMA Medium: 21,958 recordings / 140,232 training QA; 32 epochs, selected genre
validation macro 42.79%. Audio dependence is substantial, but structure-replay
loss of 2.52pp exceeded the 1pp retention allowance, so that model was not
published as the selected export. All experimental checkpoints were retained.

FMA Large: 44,968 usable recordings, 36,375 training recordings / 291,000 new
training QA. Several 10s windows/question forms produce multiple QA per record.
With 101,850 replay QA and 11,623 AudioMCQ-music QA, the total is 404,473.
Artist/source/PCM-disjoint splits; measured leakage: 0. Labels are primary
track metadata weakly projected onto 10s windows, covering 11 broad genres.

V2: 32 epochs / 50,560 updates / 2872.45 seconds. Batch 256; adapter/head/scale
LR 3e-5 / 1.5e-5 / 3e-4; balanced genre CE plus verified pair CE/swap.
Selected epoch 6 / step 9480: validation genre macro 42.88%, versus parent
16.73%. All question variants pass the validation audio gate, and retention
of perception/structure passes its criterion. Test does not select checkpoints.
Medium/Large use different held-out cohorts; 42.79/42.88% are not measurements
of the same examples.

| Reserved test cohort | QA / independent recordings | Correct | Random | Silence |
|---|---|---:|---:|---:|
| All music tasks, ordinary Top-1 | 8398 QA | 58.62% | 25.28% | 25.72% |
| FMA genre, ordinary Top-1 | 3414 QA / 569 recordings | 46.22% | 9.64% | 11.25% |
| FMA genre, class macro over 11 genres | Same 3414 QA | 42.41% | 8.19% | 9.09% |

Overall test Top-2: 74.73%, CE 1.1896, ECE 0.01870.
Scale was calibrated only on validation, 66.771→47.099, preserving ranking.
Twenty real files passed live/cache and frozen-encoder checks. Genre accuracy
is still moderate; validation macro on two unseen question forms is about 28%.
Test your own tracks/windows/questions/choice synonyms: this experiment does
not establish broad, reliable genre recognition.

Selected export: `models/samsone_clm_music`; immutable source:
`models/samsone_clm_music_genre_v2`. The previous window/lower-LR export is
preserved as `models/samsone_clm_music_before_genre_v2` and under its original
name. Common raw SHA remains
`4f4137e03975054ffe725ec635589afa31c8ddfd11b1d71db20e863a43710e78`.

[Selected weights in Git LFS](../../../checkpoints/README.md),
[backup and restoration](../../MODEL_BACKUP.md),
[v2 test JSON](iteration-v2/evaluation_test.json),
[v2 metrics](iteration-v2/metrics.json),
[v2 live audio](iteration-v2/live_audio.json).

FMA Large training (historical image omitted from this text-only export)
