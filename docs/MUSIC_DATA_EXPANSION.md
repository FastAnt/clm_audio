# Musical data: audit and expansion

Checked on 2026-10-03. The active AudioMCQ experiment remains unchanged during
precompute/training. This is a historical expansion plan, not a live status.

## Existing corpus size

AudioMCQ metadata: 571118 QA rows, 257941 strong. Usable canonical rows: 227353;
training: 181592.

Original metadata contains 15636 strong music QA. Preparation yielded 14625:

| Split | Music QA | Unique audio identities |
|---|---:|---:|
| Train | 11623 | 7377 |
| Validation | 1497 | 956 |
| Test | 1505 | 933 |

Canonical music QA by source: LP-MusicCaps-MTT 7894, CompA-R 5275, MusicCaps 1307,
AudioCaps 112, Clotho 37. The other 1011 original strong music rows are not
currently usable; exclusion reasons and audio availability need a separate
audit. Recovering them all would still provide only a small increase.

Stage 2 (50K), Stage 3 (150K), and full training already contain all 11623 music
training QA. Scaling increases sound/speech/temporal coverage, not musical
coverage. Musical specialization therefore needs new independent recordings.

## Candidate sources

### MTG-Jamendo: new tracks

The [official repository](https://github.com/MTG/mtg-jamendo-dataset) provides
about 55K tracks with genre/instrument/mood tags. Instrument tags cover 25135
tracks; the full autotagging base contains 55609. Artist metadata and official
splits are available. Full high-quality audio is approximately 508 GB;
lower-bitrate mono audio approximately 156 GB. With roughly 488 GiB free at
this snapshot, use a bounded subset or low-bitrate version with an explicit
storage budget.

These are tags, not ready-made MCQ. Training requires unambiguous positives;
tag absence does not imply a negative answer. Conversion needs taxonomy,
co-occurring-label filtering, and a quality audit. Labels must apply to the
selected ≤30s excerpt, not just the whole track. Artist/track-group splits
provide stronger musical controls than PCM hashes alone.

### MusicBench: tempo, key, controlled variants

The [official dataset](https://huggingface.co/datasets/amaai-lab/MusicBench)
has 52768 training entries, captions, extracted BPM/key/chord/beat features,
and a 16.8 GB audio archive. It expands approximately 5.5K MusicCaps base
samples with pitch/tempo/volume augmentation. Thus 53K entries are not 53K
independent songs.

It can support musical-attribute MCQ and same-question pairs with different
correct answers. Automatic annotation accuracy needs checking. All variants
of an original MusicCaps ID must share one split. Exclude AudioMCQ held-out
source IDs and their pitch/tempo variants from new training. Instrument labels
largely reuse existing MusicCaps coverage; its benefit is controlled musical
differences rather than independent recording diversity.

### NSynth: instruments and timbre

The [official dataset](https://magenta.withgoogle.com/datasets/nsynth) contains
305979 isolated notes: 289205 train, 12678 validation, 4096 test. Splits are
organized by instrument. WAV and family/source metadata support fixed
questions and clear distractors without an external generation model.

Use it as an instrument/timbre auxiliary source. Isolated notes are not full
arrangements, 306K songs, or replacements for genre/rhythm/harmony data.
Related pitch/source/instrument notes must retain their instrument split.

### Independent musical evaluation

[MuChoMusic](https://zenodo.org/records/12709974) contains 1187 human-validated
MCQ on 644 MusicCaps/SongDescriber tracks. It overlaps AudioMCQ MusicCaps;
source-ID/track checks are necessary before claiming independence.

[HumMusQA](https://huggingface.co/datasets/mtg-upf/HumMusQA) contains 320
expert-written MCQ on 108 Jamendo tracks, with 30–90s recordings. It is an
evaluation source, not a large training corpus. Samsone supports ≤30s: use
correctly annotated excerpts, not arbitrary crops with full-track answers.
Exclude these tracks/artists when expanding Jamendo training.

## Sequence

1. Complete the AudioMCQ 150K/full scaling branch with fixed snapshots/test protocol.
2. Recover usable missing AudioMCQ strong music rows and count genuinely new audio.
3. Prepare a separate immutable music-expansion manifest without retroactively
   changing existing splits. Count QA, PCM clips, parent tracks, and artists separately.
4. Choose Jamendo for new songs, MusicBench for controlled tempo/key pairs, and
   NSynth for instrument/timbre. Preserve frozen Samsone356M, shared adapter,
   and original trainable CLM heads.
5. Audit a few hundred labels/clips, frozen features, and overfit. Run separate
   music-focused fine-tuning with protection against speech domination and
   correct/random/silence evaluation per musical subtask.
6. Compare general and specialized checkpoints on reserved common and separate
   musical holdouts. Do not tune on test or count more epochs as more examples.

At the time of this plan, external audio archives had not been downloaded or
mixed into training. Later stages are documented in the final reports.
