# samsone_clm_common → Stage 5 → Stage 6 → samsone_clm_music

**Historical snapshot:** Stages 1–4 are complete and `samsone_clm_common` has been
exported. Further execution moves to a remote GPU; additional Stage 4 epochs
use a separate run, as described in [REMOTE_MIGRATION](REMOTE_MIGRATION.md).
Stages 5–6 had not yet been implemented or started when this plan was written.
The agreed musical-data preparation plan is preserved below.

This follows the user's 2026-10-03 instruction: finish available AudioMCQ first,
preserve common, then run two music stages. Instruments/genre/mood and
tempo/rhythm/harmony have equal priority.

## 1. Finish AudioMCQ and preserve common

Stage 3 uses 150000 main training QA, initialized from selected Stage 2.
Stage 4 uses all available immutable strong training rows: 181592 main QA,
initialized from best Stage 3. Count auxiliary rows and pair updates separately.

The 257941 strong metadata rows are not all usable training examples. The usable
canonical corpus has 227353 rows, including 22679 validation and 23082 test.
Do not include held-out rows to increase the training count. Document missing
source audio and excluded labels in the dataset card, rather than counting
them as trained examples.

Under the updated user instruction, Stage 3 regression must be reported and
does not block Stage 4. Resource, provenance, and nonfinite guards remain
mandatory. Validate at the end of each full Stage 3–4 pass; the selected common
checkpoint must follow at least one complete training epoch. Select by audio
dependence and then accuracy, without test selection. This does not guarantee
success; report negative audio results honestly.

After Stage 4: validation-only calibration, reserved per-category test,
correct/random/silence controls, audio-identity paired bootstrap, live/cache
parity, and web-window checks. A separate immutable precompute can expand the
balanced 2048-row common test to all 23082 canonical test rows. Report balanced
and natural-distribution results separately, and do not tune hyperparameters
using these test results.

Artifacts:

```text
/srv/data/samsone_clm/
  models/samsone_clm_common/
    best.pt
    last.pt
    calibrated.pt
    card.json
    checksums.json
    training_metrics.json
    raw_test.json
    calibrated_test.json
    paired_audio_evaluation.json
  processed/samsone_clm_common/
    canonical.jsonl
    stage4_all.jsonl
    card.json
    checksums.json
```

Use `best.pt` to initialize Stage 5 and `calibrated.pt` for common inference.
Store the SHA256-linked frozen Samsone checkpoint once in shared model storage;
do not duplicate it in each head checkpoint. Exports are immutable: reject
attempts to overwrite them with different weights. Stages 5–6 write separate
directories and leave common unchanged.

At this snapshot, the controller had been updated to require Stage 4 and then
`export_common.py`. Stages 5–6 were still plans, not running jobs.

## 2. Fix musical sources and storage budget

Stage 5 focuses on perception: MTG-Jamendo supplies new tracks and genre/mood/
instrument annotations; NSynth is a bounded auxiliary source with clear
instrument-family/source labels.

Stage 6 focuses on structure: MusicBench tempo/beat/key/chord metadata and
known pitch/tempo variants, plus replay of selected Stage 5 examples.

Jamendo and MusicBench metadata have already been downloaded and verified
under `raw/music_expansion/`; source URLs, bytes, and SHA256 are preserved.
`MusicBench_train.json` is actually JSONL and must be read line by line.
Upstream Jamendo `autotagging.tsv` is a symlink; download its actual TSV target.

Measured inventory:

| Source | Size | Interpretation |
|---|---:|---|
| MTG-Jamendo autotagging | 55609 tracks / 3565 artists | Track-level uploader tags, not ready-made MCQ |
| Jamendo instrument tags | 25135 tracks | Multiple instruments and incomplete annotations are possible |
| MusicBench train | 52768 files / 5079 parent YouTube IDs | Pitch/tempo variants, not 53K independent songs |
| MusicBench after excluding AudioMCQ held-out parents from potential train | 50362 files / 4838 parents | Metadata eligibility only, not prepared QA |
| Official NSynth | 305979 isolated notes | Auxiliary data, not 306K songs |

Do not promise QA counts before auditing labels and clips. The first music
iteration targets tens of thousands of usable new clips and roughly 50–100K
high-quality QA if the audit supports it. Repeating templates without new audio
is not a substitute for coverage.

The full Jamendo high-quality archive is about 508 GB; lower-bitrate mono audio
is about 156 GB. With roughly 487 GiB free, avoid downloading the full
high-quality archive alongside duplicate extracted files. Begin with a
bounded task/class/artist subset and verified shard processing; budget raw,
processed, and feature storage in advance. Retain RAM/disk floors and watchdogs.
MusicBench audio is about 16.8 GB. Sample NSynth by instruments/classes instead
of including every note in every epoch.

## 3. Canonical contract and lineage

Preserve existing required API fields and add separate musical metadata:

```text
sample_id
audio_path
question
choices[]
answer_index
question_type = music
source_dataset
music_task = instrument | genre | mood | vocal | tempo | meter | key | mode | chord
source_track_id
parent_track_id
artist_id / instrument_id (where applicable)
split_group_id
window_start_s / window_end_s
silence_before_s / silence_after_s
pcm_identity / audio_sha256
label_origin / label_confidence
split
```

Record template and preprocessing versions. Labels, metadata, and paths never
enter the encoder: STATE contains only audio + question; ACTION only choice
text. Queries and choices must use supported English. Spell Unicode accidentals
as `sharp/flat`, because the official ASCII normalizer removes their symbols.
Upstream preserves ASCII `#`. Merge enharmonic equivalents such as C sharp /
D flat into one pitch class; do not offer them as different wrong/correct
choices. Check normalization collisions and answer-index/text consistency.

The main MCQ task needs exactly one unambiguous correct choice. Consider all
annotated positives when checking distractors. Reject ambiguous/multi-answer
rows or place them in a separately designed multi-positive experiment; do not
arbitrarily choose one correct label.

## 4. Split before cropping and augmentation

Order: source-parent/artist/instrument grouping → split → windows/variants →
PCM deduplication. A track and all its variants belong to one split. PCM-only
splitting is insufficient for pitch/tempo augmentation and different codecs.

MusicBench source IDs use the first 11 characters of the YouTube ID. Validate
augmentation suffixes separately: an original YouTube ID can itself end in an
underscore followed by digits, so an unrestricted suffix-removal regex corrupts
parent IDs.

Merge source groups with AudioMCQ:

- Common-training parents may appear only in musical train/replay, not an
  independent music validation/test split.
- AudioMCQ validation/test parents never enter musical training. Preserve their
  original split if reused in music evaluation.
- Assign new parents stable splits once; all variants inherit them. Keep the
  general reserved test unchanged after Stages 5–6.
- Group Jamendo by artist. Preserve official NSynth instrument-disjoint splits
  where possible and audit their actual identities. Check exact PCM duplicates;
  acoustic fingerprinting without a new ML model may identify codec duplicates.

MuChoMusic and HumMusQA are evaluation-only. Exclude their tracks and known
artists from new training. Check MusicCaps/MuChoMusic overlap with common;
do not claim independence where overlap exists.

## 5. Audio preprocessing and label alignment

Decode to mono 16 kHz with the existing decoder; publish verified PCM atomically.
Do not silently alter pitch, tempo, or loudness. The architectural context
remains at most 30 seconds.

Prefer annotated 10-second excerpts aligned with the web demo. Pad shorter clips
to 10 seconds and record offsets/padding to reproduce the input. For long songs,
choose a window and verify label applicability before cropping. A whole-track
`piano` tag does not establish piano in the introduction.

Preserve a MusicBench full excerpt when its ≤30s labels cannot be transferred
reliably to 10 seconds. Create separate 10-second examples only with valid local
annotations. Keep full-clip and window evaluation separate.

Store the actual variant in features/QA, not its base audio: tempo-shifted
waveforms use their own row's BPM/beats. Convert beat/chord timestamps into
window coordinates when offsets change. Pitch/tempo augmentations must not
retain unmodified key/BPM labels.

## 6. Stage 5: instruments, genre, mood, vocals

Jamendo: read all TSV tags, merge upstream aliases, define task taxonomies, and
account for co-occurring positives. Track-level uploader tags are weak labels,
not human-verified clip truth. Retain MCQ with usable local annotation and
unambiguous distractors. Missing instrument tags mean unknown; they do not
supply negative yes/no labels. Exclude incompatible labels and ambiguous clips.
Genre/mood labels can co-exist, so other tags are not automatically negatives.

If high-quality annotations cannot support the desired size, reduce the usable
count, add genuinely labeled sources, or obtain clip-level annotation. Missing
labels must not masquerade as automatically verified ground truth. Track weak
supervision separately; inclusion needs a distinct quality evaluation and
validation comparison.

NSynth: use annotated family/source for single-note MCQ. A `keyboard` family
is not a specific piano, and `string` is not a specific violin. Match questions
to annotation granularity. Balance families/sources and limit notes per
instrument/pitch/velocity. Padding a short note to 10 seconds preserves its
signal and family label. Isolated notes should not displace real arrangements.

Use grammatically parallel choices of similar length; shuffle while preserving
answer text. Balance correct positions and classes. Never reveal song, artist,
or source in the question. Use fixed task templates with valid paraphrases,
checking that question/choice text does not reveal the answer.

Before large Stage 5 training, audit a few hundred clips/labels across subtasks.
Run a 256–1000-example diagnostic initialized from common, training adapter and
both heads, with correct/random/silence on balanced held-out data. Keep the
common parent immutable.

Stage 5 starts from `models/samsone_clm_common/best.pt`. Samsone stays frozen;
shared adapter, both heads, and scale train. Initial sampling proposal:
80% accepted musical examples and 20% common-training replay, balancing musical
subtasks. Use lower LR than common training and select against music + common
validation. Save best/last/metrics/config/checksums; this is an intermediate
music model. Later user instructions restricted replay to music only.

## 7. Stage 6: tempo, rhythm, key, harmony

All 52768 MusicBench metadata rows passed beat/time and chord/time length
consistency checks. Fourteen rows have no beats; 3 have one beat position.
They cannot support meter inference. This is structural validation, not proof
that automatic musical labels are accurate.

Tempo: require finite BPM, strictly increasing beat times within decoded
duration, and agreement with inter-beat intervals while accounting for
subdivision. Start with stable ranges/categories; exclude ambiguous half/double
tempo and samples too close to bin boundaries.

Meter: require several fully supported bars in the excerpt. The maximum beat
index alone is insufficient. Reject changing meter or insufficient bars.
BPM and meter alone do not establish a rhythmic pattern.

Key/mode: require valid pitch-class/mode annotations. Initial `keyprob >=0.8`
filters weak automatic labels; it does not guarantee truth. Audit musical
correctness on a curated subset. Use actual shifted keys for augmented variants.
Unify enharmonic spelling; do not interchange major/minor or relative keys.
Test symbol normalization and duplicate choices before precompute.

Harmony: construct chord intervals from `(chords, chords_time)` and duration,
accounting for offsets/time stretches. One known chord is not a progression.
Ask progression questions only with multiple annotated changes in the selected
window. An anchored question uses the interval containing its anchor; reject
boundary/unknown chords. Root-only labels do not establish seventh/extended
voicings.

Controlled music pairs share the SAME question and canonical choices, but have
different audio and correct labels. Supplied variant metadata must establish
tempo/key differences. Pairs stay within train and the parent split. Swap-loss
gathers use canonical choice text/ID, not another row's column order. Random
audio and silence remain evaluation controls, not false supervised positives.

Initialize Stage 6 from best Stage 5. Proposed replay: 45% structural tasks,
45% accepted Stage 5 perception, 10% common-training replay. Final counts follow
the audit; quality and independent groups matter more than generated-template
counts.

## 8. Precompute, integration, monitoring

Compute one frozen hidden per `(actual PCM, question)` and unique choices through
Samsone's frozen text path. Preserve the encoder contract, FP32 computation,
and BF16 shards. Source SHA, question, and recipe define the cache key. New
crops/augmentations create new PCM identities. Training never runs Samsone.
Keep general and music feature manifests separate and immutable.

Implementation required after common: permit Stages 5/6 in `set_stage`,
stage configuration/selection, initialization checks, and `train_all`, without
topology changes. Add per-music-task metrics and class/parent-aware sampling.
The UI should show Stages 1–6 and actual main/auxiliary/replay counts. Add music
download/prepare/audit scripts and separate common/music run roots.

Meaningful checks cover parent/artist/instrument leakage, inherited variant
splits, missing tags not becoming verified negatives, timestamp/crop/tempo
transforms, enharmonic aliases/normalization collisions, swap-column alignment,
and immutable common export.

## 9. Preserve samsone_clm_music

After Stage 6: separate music-validation calibration; reserved tests, without
tuning, for instrument/genre/mood/vocal/tempo/meter/key/mode/chord and the
original common tasks. Report correct/random/silence and paired confidence
intervals by parent-track/artist/instrument groups. Window tests need actual
window labels, not answers inherited unquestioningly from full tracks.

Two key outcomes are musical quality/audio effects in both task families and
the change in common accuracy after specialization. Explicitly document
subtasks without an audio benefit; an aggregate music score must not hide weak
temporal/key/meter results.

Output: `models/samsone_clm_music/{best.pt,last.pt,calibrated.pt,card.json,checksums.json,metrics.json}`.
Parent SHA links back to common; the common export is unchanged. API/web select
an explicit common or music checkpoint. At the time of this plan, Stages 5–6
were not launched: verified Stage 4 common remained the first milestone.

## Primary sources

- AudioMCQ: https://huggingface.co/datasets/inclusionAI/AudioMCQ
- MTG-Jamendo: https://github.com/MTG/mtg-jamendo-dataset
- MusicBench: https://huggingface.co/datasets/amaai-lab/MusicBench
- NSynth: https://magenta.withgoogle.com/datasets/nsynth
- MuChoMusic: https://zenodo.org/records/12709974
- HumMusQA: https://huggingface.co/datasets/mtg-upf/HumMusQA
