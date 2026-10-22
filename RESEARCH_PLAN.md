# Audio System-1: research and implementation plan

**Historical research.** Stages 1–4 have since completed; see the current
[README](README.md). Qwen is now fully excluded. Both branches use frozen
Samsone-356M and the SAME trainable adapter; original CLM heads provide
initialization, frozen in Stage 1 and both trained in Stages 2–4. The original
research below is preserved; proposals involving Qwen or a permanently frozen
action head are superseded and must not be used as current instructions.

Checked on October 2, 2026. This analysis reviewed official code, configurations,
release assets, metadata APIs, and small JSONL samples. Full checkpoints had
not been downloaded; hidden extraction and training had not run. Dimensions
below were supported by code/configuration and still needed actual-weight
verification in the first implementation step.

## 1. Feasibility

The state chain `Samsone-356M frozen → trainable adapter → original CLM state head → z_state`
is technically implementable. Successful training and transfer to unseen audio
remain experimental hypotheses.

The original ambiguity was CLM's second branch: standard published ranking also
used `choices → frozen Qwen3-8B → frozen original action head → z_action`.
Qwen3-8B is the reference CLM encoder but would count as an additional model
under a literal reading of the specification. Without changing that reference
candidate recipe, forbidding it prevents arbitrary new choices. Caching existing
action vectors supports a fixed vocabulary but not unseen strings. The user
subsequently resolved this by requiring the Samsone text path and retraining
both heads; the historical proposal here is not the final architecture.

The initial proposal, conditional on permitting CLM's reference encoder, was
Qwen for choices only, with Qwen/action head/logit scale frozen. State would
stay as specified and only the original state head would unfreeze in Stage 2.

Official checkpoint: [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B),
file `CLM_v0.1-8B.pt`. Its model card binds the heads to Qwen3-8B last-token
embeddings. An adapter must learn compatibility after state-encoder replacement;
matching dimensions alone does not establish it.

## 2. Source-confirmed facts

| Component | Confirmed | Still requiring execution |
|---|---|---|
| Samsone | Samsone356M class, official BF16 PyTorch checkpoint, v1.0.0 release | Strict load, CUDA forward, hidden extraction |
| State representation | Configured text width 960; audio and question combined into one LM input | Actual shape, last-valid pooling, audio/question sensitivity |
| CLM | Original state/action heads, cfg, logit_scale, input width 4096 | Actual weight shapes and projection dimension |
| AudioMCQ | Seven metadata JSONL files, 571,118 published rows, separately hosted source audio | Exact usable/available strong count |
| Workstation | NVIDIA GB10, aarch64, Python 3.12.3, uv available | CUDA compatibility of selected PyTorch/decoder wheels |

Samsone sources: [repository](https://github.com/SamsungLabs/samsone),
[release](https://github.com/SamsungLabs/samsone/releases/tag/v1.0.0),
[inference/models.py](https://github.com/SamsungLabs/samsone/blob/main/src/samsone/inference/models.py),
[AudioLM](https://github.com/SamsungLabs/samsone/blob/main/src/samsone/models/multimodal/audio_lm.py).

Verified revisions: Samsone `a71e77ae9528ef411ae26cdeb02d7b90cc18a50c`;
CLM `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`.
Also pin Hugging Face revisions and weight SHA256 during implementation.

## 3. Extracting Samsone hidden representations

Public `Samsone356M.__call__` generates text. The project instead needs a wrapper
around the same loaded AudioLM, reproducing official preparation:

1. Decode, convert to mono, resample to 16 kHz; upstream uses 30-second crop/pad.
2. Reuse the checkpoint's audio encoder, internal projector, and separators.
3. Tokenize the question with the official tokenizer/pruned vocabulary map,
   preserving ` answer: `.
4. Combine embeddings/masks with `_combine_embeds_for_lm_input`.
5. Run one transformer forward with `inputs_embeds`, `attention_mask`, and
   `use_cache=False`, without generation or the correct answer in the input.
6. Pool final-layer hidden at the last valid prompt token after audio/question.
   For left padding, `mask.sum()-1` is incorrect; locate the last actual 1.

This is fixed pooling within the wrapper, not an additional trainable model.
Samsone's internal projector is not the external adapterLayer and stays frozen.

Official inference lowercases English and removes non-ASCII symbols using a
pruned vocabulary. Russian support cannot be promised. The UI can use Russian,
but initial questions/choices must be English. Do not silently reduce a Russian
question to an empty string.

On real WAV, verify eval-mode determinism, stable shape, finite values,
question sensitivity, and audio sensitivity with the question fixed. Numerical
hidden changes do not by themselves establish useful audio understanding.

## 4. Adapter and CLM ranking

Infer dimensions from loaded configurations and weight tensors. Current release
expectations are `D_samsone=960`, `D_clm_input=4096`; read projection dimension
from weights and check the action head. Do not hardcode these values.

Adapter proposal:
`LayerNorm(Ds) → Linear(Ds,b) → GELU → Linear(b,Dclm) → LayerNorm(Dclm)`.
Initially `b=Ds`, configurable by policy. This is about 4.9 million parameters
at expected widths. Hyperparameters remain explicit even with inferred boundary
dimensions.

The official embedder normalizes input vectors. L2-normalize adapter output
before the state head to match that convention, checking the pinned source.

Historical reference scoring proposal, superseded in its candidate backbone:

```python
h = frozen_samsone(audio, question)
x = normalize(adapter(h))
z_state = normalize(original_state_head(x))
z_action[j] = normalize(original_action_head(normalize(original_qwen(choice[j]))))
score[j] = exp(original_logit_scale).clamp(max=100) * dot(z_state, z_action[j])
loss = cross_entropy(scores, answer_index)
```

This is standard CLM scoring and the specified per-example candidate CE.
Original pretraining used bidirectional InfoNCE; the MCQ specification does not
require replacing CE with it. Check logits/probabilities against official
ranking on a text fixture before attaching Samsone.

Stage 1 pitfall: freeze state-head parameters with `requires_grad_(False)`, but
never wrap its training forward in `no_grad`/`inference_mode`. Gradients must
reach the adapter through the frozen head. Official
[HeadPair._project](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/heads.py)
uses no_grad for serving and is unsuitable for training. Construct heads with
the official factory and call their modules directly. Frozen extraction can
run without gradients; training reads h from disk.

## 5. Data and audio identity

[AudioMCQ](https://huggingface.co/datasets/inclusionAI/AudioMCQ) distributes
metadata, not a complete waveform collection. The seven JSONL files total
1,100,594,842 bytes according to the Hugging Face tree API. The
[paper](https://arxiv.org/html/2509.21060v3) reports 571,118 examples and 313,177
weak examples, implying 257,941 strong. This was a paper-derived count, not a
fresh recount of the snapshot.

Inspected rows use `audio-contribution`, not `audio_contribution`; support both
and normalize the key. `answer` is text; derive `answer_index` by matching
choices. Require exactly one match and reject ambiguous/malformed rows with
a reason.

Extend the canonical minimum with `sample_id`, `audio_identity`, `source_audio_id`,
`segment_start_s`, `segment_end_s`, `duration_s`, `audio_hash`, and `split`.
A source ID need not identify one question: CompA-R/TACOS reuse IDs/audio paths
across questions. Do not deduplicate by ID alone.

| Audio source | Retrieval/alignment |
|---|---|
| Clotho | [Zenodo 2.1](https://zenodo.org/records/4783391), WAV names and metadata, including renames |
| AudioCaps v2 | [Official site](https://audiocaps.github.io/), source ID and timestamps |
| CompA-R | Authors' AudioMCQ link and [their backup](https://huggingface.co/datasets/PinnHe/CompA-R-Backup) |
| MusicCaps | [Google metadata](https://huggingface.co/datasets/google/MusicCaps), YouTube ID and start/end |
| LP-MusicCaps MTT | Original MagnaTagATune, preserving relative MP3 paths |
| SpeechCraft | [LibriTTS-R](https://www.openslr.org/141/), split/speaker/chapter/utterance |
| TACOS | [Zenodo](https://zenodo.org/records/15379789), source audio ID |

Sequence: all metadata → strong filter → required-audio inventory → download
available required files → decode/integrity checks → canonical manifest.
`download_report.json` records requested/available/missing/corrupt counts by
source. Missing audio must not be replaced with silence in ordinary training.

Split source-identity groups once; the original proposal was approximately
90/5/5 with a seed. Groups include cross-dataset duplicates, source recordings,
and overlapping segments. Use IDs, offsets, and decoded PCM hashes: container
hashes miss codec variants, while exact PCM hashes miss overlapping excerpt
boundaries. Check source mappings and suspicious overlaps, using non-ML audio
fingerprints if needed.

Count questions and unique audio separately. Stratify sources/question types
without breaking groups. Stage 3 needs at least 150K valid strong TRAIN rows
after retrieval, deduplication, and splitting; availability is not guaranteed.

Samsone's README names ReasonAQA and AudioSkills as pretraining sources. Audit
known overlap with evaluation audio. Project-level audio-disjoint splits do
not guarantee that frozen pretraining never saw test recordings.

## 6. Storage, precompute, ablations

Code: `/opt/samsone-clm-backup`. All large data/model/feature/run/cache
files live under `/srv/data/samsone_clm/`, with `raw/`, `processed/`,
`features/`, `models/`, `runs/`, `cache/`. Redirect library/uv caches and large
download temporaries there. Samsone has a separate home cache: pass an explicit
checkpoint path and relocate its prune-map cache; setting HF_HOME alone is
insufficient. Give CLM explicit local checkpoint/cache paths.

Store pooled h as BF16/FP16 shards with a sample→shard/row manifest, atomic writes,
and resume. Proposed formats: safetensors and Parquet metadata. Cache identity
includes checkpoint SHA, tokenizer/prompt/extraction/preprocessing versions,
audio hash, crop boundaries, and question. Store h rather than z_state because
adapter/state-head weights change during training.

The historical fixed-action proposal encoded each unique choice once with its
frozen reference encoder/head and could cache final z_action. Keys include
recipes and both weight hashes. Such projected caches remain valid across
stages only if the action head stays frozen; the final architecture instead
trains both heads and caches pre-head hidden.

For held-out controls precompute `(random_audio_i, original_question_i)` and
`(silence_i, original_question_i)`. Random audio comes from the same held-out
split with a different identity and fixed permutation/seed, ideally repeated.
Do not shuffle existing h: those representations already contain other questions.
Silence means a zero waveform through the full pipeline, not a zero hidden.
Compute ablation features before training; never call Samsone in the training loop.

At 258K rows and width 960, 16-bit pooled h is approximately 0.50 GB.
Undeduplicated candidate z is at most about 1.06 GB for four choices and
projection width 512. Three full h modes total roughly 1.5 GB, although
held-out-only ablations usually suffice. Estimates exclude manifests/filesystem
overhead. Full sequences multiply storage by saved-token count; the first
experiment uses a fixed pooled vector.

Mono 16kHz int16 PCM needs 32KB/s: a 30s unique clip is about 0.96MB. Budget
by unique audio, including room for archives and extracted files.

## 7. Audio duration and the web window

Official Samsone inference crops/pads at 30 seconds. Some dataset questions
need more than 10 seconds: one inspected TACOS question asks orchestra duration
with answer `24.4s`. Assigning that answer to an arbitrary 10s crop is invalid.

Preserve original lengths/timestamps and use the annotated clip within the
supported context for offline training/evaluation. Do not silently crop
unsupported clips: report them and establish whether the required evidence
is available. Claiming all available samples requires this audit.

The web task uses a local window and needs a separate short-clip held-out
assessment or questions with confirmed evidence in the window. Full-clip test
quality does not measure the 10s demo. A uniform 10s task needs valid window
labels; four training stages do not fix label mismatch.

For playback t and duration T, the nominal window is `[t-5,t+5]`, real source
`[max(0,t-5),min(T,t+5)]`, left padding `max(0,5-t)`, right padding
`max(0,t+5-T)`. At 16kHz output is exactly 160000 samples. Do not shift the
window at a boundary. Snapshot playback position when Analyze is clicked,
even if playback continues.

## 8. Implementation sequence and gates

| Order | Work | Measurable result |
|---|---|---|
| 1 | Project, uv/Python/CUDA/decoder, pinned dependencies | CUDA smoke check and WAV decode |
| 2 | Official Samsone and CLM downloads; historical candidate encoder only if permitted | SHA manifest, strict checkpoint load |
| 3 | Samsone extraction, original CLM loading/scoring, adapter | Shapes, finite outputs, gradients/freezing, official score fixture |
| 4 | AudioMCQ metadata/audio, canonicalization/group splits | Coverage, rejects, no known identity overlap |
| 5 | Initial overfit/10K precompute and held-out ablations | Shards, reproducible loading, manifest |
| 6 | Overfit 128–512, frozen state head | Loss curve/accuracy, go/no-go |
| 7 | Stage 1 on 10K strong training rows | best.pt, last.pt, metrics.json, validation/ablations |
| 8 | Missing-feature computation with reuse after the gate | Complete immutable manifest before subsequent stages |
| 9 | Stage 2 on 50K from Stage 1 best | Adapter and original state-head training |
| 10 | Stage 3 on 150K from Stage 2 best | Matching validation protocol |
| 11 | Stage 4 on all valid strong TRAIN from Stage 3 best | Final checkpoint, held-out test |
| 12 | rank API, FastAPI/HTML/CSS/JS, real WAV | API/UI agreement, window/metadata/latency |

Full-corpus precompute need not precede initial overfit. Each unique pair is
computed once; growing inventory reuses existing pairs. Every stage begins
only after all of its required features exist.

Proposed mechanical overfit target: training CE <0.2 and Top-1 ≥95% on 128–512
valid samples within a bounded step budget, with a clear loss decrease.
`log(4)=1.386` is uniform-prediction reference loss, not a required initial value.
These are proposed thresholds, not guaranteed frozen-head results. On failure,
check labels, freezing/gradients, masks/pooling, norms, candidate recipe, and LR.
Do not unfreeze Samsone or bypass the original head.

After diagnostic overfit, start main Stage 1 with fresh adapter/optimizer;
retain overfit separately. Memorization does not establish alignment or
generalization. Scaling also requires validation quality and audio sensitivity.

Subsets are fixed and nested: 10K ⊂ 50K ⊂ 150K ⊂ all train; validation/test
stay unchanged and are excluded from Stage 4. The original proposal trained
adapter throughout, state head in Stages 2–4, and froze Samsone/action
encoder/action head/scale. That proposal was superseded by the two-trainable-head
architecture stated at the beginning of this document.

Initial optimizer proposal: AdamW, batch 128–512 subject to measurement,
adapter LR around 1e-3 with a short train/validation sweep; lower Stage 2
state-head LR, for example 1e-5–1e-4. Refine epochs/patience/batch from overfit
and pilots. BF16 storage/autocast, with FP32 normalization/logits/loss where
needed. Never tune on test.

Select best by validation Top-1, breaking ties with CE. `last.pt` stores
optimizer/scheduler/RNG/epoch/step; `best.pt` stores inference weights and
provenance. Stage transitions load best weights and build new parameter groups;
within-stage resume restores the optimizer exactly. `train_all --resume` checks
manifests, gates, completion flags, and atomic checkpoint completion.

## 9. Evaluation and audio dependence

Report Top-1, Top-2, sample-weighted CE, mean P(correct), and ECE by
music/sound/speech/temporal/overall, with QA and unique-audio counts. Overall
is micro-averaged; also report four-category macro due to imbalance. Fix ECE
bins in advance, for example 15, using maximum probability and Top-1
correctness, and show a reliability table.

For four choices, random Top-1 is 25% and Top-2 50%; exceeding chance does not
prove audio use. Authors' `strong` label comes from silent-audio tests of
other LALMs and does not guarantee dependence in this system.

Report correct/random/silence metrics, paired accuracy gaps, prediction
agreement, and audio-group bootstrap intervals. Proposed project gate:
at least 5 percentage points above both ablations and difference 95% CI
above zero, with per-type results. This practical 5pp requirement is a proposal,
not a literature result. Insufficient data for a convincing interval means
insufficient evidence rather than success.

Use test for final evaluation only. During training, check validation ablations.

## 10. Project and commands

```text
~/tools/samsone-clm/
  pyproject.toml
  uv.lock
  configs/{data,models,precompute,overfit,stage1,stage2,stage3,stage4}.yaml
  scripts/{download_data,prepare_data,precompute,train,evaluate,infer,server}.py
  src/samsone_clm/
    paths.py
    data/{download,prepare,identity,features}.py
    models/{samsone,clm,adapter}.py
    precompute.py
    train.py
    train_all.py
    evaluate.py
    infer.py
    server.py
    static/{index.html,style.css,app.js}
  tests/
  docs/
```

Scripts should be thin package wrappers. All specified commands share default
configuration/data root. Add `uv run python scripts/prepare_data.py` and
`uv run python -m samsone_clm.train --overfit --samples 256`. Download must not
start training automatically. Precompute checks canonical preparation and
explains missing inputs; idempotent preparation could support a two-command
workflow.

Expose `uv run samsone-clm-serve --checkpoint <checkpoint> --port 8800` through
project.scripts. At the research date these were an implementation contract,
not working commands.

`rank(audio, question, choices)` accepts at least two choices and returns sorted
choice/probability objects. New strings use the frozen candidate path and cache.
Probabilities are relative to supplied candidates, not guaranteed calibrated
truth probabilities for arbitrary unseen answers.

Web: upload, `<audio controls>`/seek, question, dynamic choices, Analyze.
The backend receives upload identity and playback-position snapshot, selects
and pads the window, and calls rank. Return playback position, nominal window,
source interval, padding, inference latency, and checkpoint name/SHA. Distinguish
model versus request latency and warm/cold cache. No generation is required.

Check grouped duplicates/overlaps, CE label mapping, gradient flow through the
frozen state head, unchanged frozen weights, no training-time Samsone calls,
feature/checkpoint consistency, resume, ECE, edge padding, and API/direct
agreement on a real WAV with arbitrary candidates.

## 11. Resources and timing estimates

The inspected workstation had 121Gi RAM with about 106Gi available, NVIDIA GB10,
driver 580.126.09, and about 878Gi free data storage. System Python lacked
Torch/Transformers/TorchAudio/TorchCodec; other environments were not inspected.
[NVIDIA's GB10 stack documentation](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/software-requirements.html)
describes ARM and unified memory. RAM and VRAM cannot be added as independent
resources; inspect actual PyTorch allocation after setup.

Upstream assets were approximately 714MB for Samsone and 75.6MB for CLM heads.
The historical Qwen3-8B BF16 estimate was around 16GB by parameter count,
excluding buffers/activations/caches; this backbone was later excluded entirely.
Adapter/head training is much smaller and needs neither encoder resident
after precompute.

Main costs are source retrieval, decoding, Samsone precompute, and unique-choice
encoding. Benchmark 500–1000 examples and estimate
`T_precompute ≈ N_pairs / measured_pairs_per_second`, accounting for choices
separately. At a hypothetical 5 pairs/s, 258K pairs take roughly 14.3 hours;
at 20 pairs/s, 3.6 hours, excluding downloads/choices. These are scenarios,
not GB10 measurements.

Engineering estimate: 1–2 working days for reproducible model/extraction/gradient
proof, 2–4 for resolvers and overfit, 1–3 for staged training/evaluation/resume,
1–2 for API/demo integration. Approximately 5–11 days assuming accessible
sources and no prolonged ARM dependency debugging. Download/precompute/training
wall time is additional. Recovering unavailable online audio and achieving
research quality cannot be guaranteed by a calendar estimate.

## 12. Original decisions required before implementation

1. Would the reference frozen Qwen3-8B be permitted for candidates? This was
   subsequently resolved by excluding Qwen and using Samsone for both branches.
2. Full-clip offline training and local 10s serving require separate evaluation,
   or valid window-level labels for a uniform 10s task.
3. Stage 4 means every usable strong TRAIN example, keeping validation/test held out.
4. Fix overfit/audio gates, checkpoint selection, and unsupported/missing-audio handling.

First implementation evidence: real WAV + question → verified Samsone hidden →
trainable adapter → unchanged original state head → candidate scores and
working backward. Follow with frozen-head overfit before scaling. These were
the original development gates; later outcomes are in the current reports.
