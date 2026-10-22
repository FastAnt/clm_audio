# Samsone native inference and hidden-path audit — 2026-10-03

The Stage 2 audio ablation fails the experiment's audio-use criterion. On the same
2,000 validation samples, correct audio gives 1,256 correct answers (62.8%), random
audio also gives 1,256, and silence gives 1,251 (62.55%). Equal aggregate accuracy
does not imply equal predictions: random audio changes 12 answers, losing four
correct answers and gaining four; silence changes 12, losing seven and gaining two.

Earlier smoke/overfit checks verified tensor shapes, freezing, gradients and loss
reduction. They did not establish native Samsone semantic performance. The audits
below now provide direct evidence, with limitations made explicit.

## Native Samsone, without the adapter or CLM heads

`scripts/audit_samsone.py` calls the official Samsone-356M inference helper on real
audio. Generation is diagnostic-only; training, ranking API and demo still use
cosine candidate ranking, with the specified frozen Samsone architecture.

| Input | Native response, paraphrased | Observation |
| --- | --- | --- |
| Official `cow.wav` asset | Cows mooing | Plausible matching description |
| Official `water.wav` asset | Rhythmic raindrops | Plausible water/rain description |
| AudioMCQ AudioCaps, train-horn question | Approaching train blowing its horn | Matches the labeled event |
| AudioMCQ SpeechCraft, emotion question | Disgust, followed by neutral/dispassionate voice | Contradictory; do not count as a success |
| AudioMCQ LP-MusicCaps-MTT, instruments | Strings, violins and cellos, classical music | Conflicts with AudioMCQ label; original source tags support strings/classical |

These are five diagnostic examples, not a native-model accuracy benchmark. The
silence control also produces a vague, hallucinated description; native Samsone
is not assumed reliable on every input.

## Adapter input and native prefill equivalence

The state vector is `LlamaModel.last_hidden_state` after its final norm, pooled
at the last valid token of the official audio + normalized question + ` answer: `
sequence. Its width is 960. Forward pre-hooks show this exact vector, cast to
FP32, enters the shared adapter. The choice branch uses the same frozen text
transformer and the same adapter instance. Checkpoint-derived widths are
960 → adapter → 4096 → original CLM head → 512. All Samsone parameters are frozen.

For each of the five single-example cases, our state vector is **bitwise equal**
to the last-valid-token hidden captured during the first forward of official
native generation: maximum absolute error and relative L2 error are zero.

The separate batched audit also tests official `AudioLM.generate` with one
audio assigned to each sample (the helper's list-of-paths API instead assigns
multiple audio files to one sample). Batch size 2 with equal prompt lengths
matches exactly. In the five-example batch with different lengths, production
extraction differs from native prefill by mean relative L2 **0.0077403** (0.774%),
mean cosine **0.9999654**. Replaying the native inputs with its explicit
`position_ids` restores exact equality. Production direct forward currently
uses default positions, while native generation derives positions from the
padding mask. This discrepancy is recorded and remains in the existing v1
features; the audit does not silently change or mix feature recipes/checkpoints.
This small measured discrepancy does not establish the cause of failed ablation.

Native successful generation does **not** prove that one pooled final-token
vector is sufficient for candidate ranking. Native decoding can attend to all
prefill tokens through its attention cache. Our audit establishes the extraction
location and audio sensitivity, not the sufficiency of the pooled representation.

## Audio sensitivity through the trained Stage 2 model

`scripts/audit_audio_signal.py` evaluates the existing best Stage 2 checkpoint
and the same 2,000 cached validation samples without running Samsone again.

| Change from correct audio | Raw hidden, mean relative L2 | Adapter output | Normalized state vector | Mean absolute candidate probability change |
| --- | --- | --- | --- | --- |
| Random audio | 13.13% | 10.10% | 4.80% | 0.0863 percentage points |
| Silence | 18.50% | 14.40% | 6.28% | 0.1065 percentage points |

The raw representation reacts to audio. The trained projections and ranking
produce little aggregate audio dependence. Relative L2 values across layers
are descriptive diagnostics, not an information-theoretic attribution of blame
to one layer. Neither a native caption nor falling CE establishes audio-based
reasoning in the trained ranking model.

## Verified source/label conflict

For sample `856ceb6896c39a5735ca132e57e5d0398a3a954e8eb1e0e441fb0f81f49df63f`,
AudioMCQ's original metadata points to
`4/ensemble_mirable-triemer_six_cello_sonatas-06-sonata_2_largo-30-59.mp3`
but labels the main instruments **Electric guitars and drums**. The source MP3
from the downloaded archive and the prepared WAV decode to bitwise-identical
mono PCM: 465,984 samples at 16 kHz. This preparation step did not substitute
another recording.

The matching original MagnaTagATune annotation, clip ID 29083, has active tags
`no piano`, `classical`, `strings`, `violin`. These independently support the
native answer's broad instrument family and contradict the AudioMCQ guitar/drum
label. They do not verify every instrument detail in the generated answer.
This is one verified conflict, not an estimate of corpus-wide label noise.

## Reproduce and inspect

```bash
cd ~/tools/samsone-clm
uv run --no-sync python scripts/audit_samsone.py --device cuda
uv run --no-sync python scripts/audit_samsone_batch.py --device cuda
uv run --no-sync python scripts/audit_audio_signal.py
```

Reports are under `/srv/data/samsone_clm/runs/samsone_native_audit_cuda/`:
`native_cases.json`, `adapter_path_proof.json`, `batch_prefill_proof.json`,
`stage2_audio_signal.json`, `music_source_alignment.json`, `music_original_tags.json`.
Native cases include paths, prompts, full responses and measured errors.
The source-tag report includes its pinned annotation URL.

The main pipeline continues on the existing v1 feature recipe. At this audit,
Stage 1 and Stage 2 are complete; full-corpus precompute for Stage 3–4 is active.
No architecture, model weights, labels or existing cached features were changed
by these diagnostic scripts.
