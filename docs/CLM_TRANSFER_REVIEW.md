# Original CLM comparison and user-track audit

2026-10-04. Reviewed the public official repository and pinned upstream
`bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`. `git ls-remote` confirmed it was then
the current HEAD. Encoder, weights, training recipe, and web UI were unchanged.
All new diagnostics ran on CPU with four PyTorch threads and CUDA hidden.

## Finding

Samsone wiring and the mathematical head contract are verified on two user
excerpts. Both trained heads changed and audio affects predictions. Arbitrary
genre ranking nevertheless remains unreliable: errors reproduce on user tracks
and change strongly with question/choice paraphrases.

Official CLM does not promise retained capabilities after replacing its
backbone. This experiment needs to learn audio/text alignment on frozen Samsone.
Pretrained-head transfer is a testable hypothesis; its initialization advantage
has not been isolated with a controlled initialization experiment.

## Official source contract

1. [README: Data Recipe](https://github.com/Contrastive-LM/CLM#data-recipe)
   binds reference heads to the encoder/pooling used for training. The original
   checkpoint uses Qwen3-8B last-token pooling. This project does not use that
   backbone: both branches use official frozen Samsone356M. The shared adapter
   maps 960→4096; widening cannot create information missing from its input.
2. [embedder.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/embedder.py)
   normalizes backbone embeddings; [heads.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/heads.py)
   normalizes head outputs and applies positive `exp(logit_scale)`, capped at 100.
   The selected model retains these norms and scaled cosine. Calibration to
   47.099 changes probabilities, not candidate order. Historical scale/L2 bugs
   do not describe current v3/music weights.
3. [schema.py](https://github.com/Contrastive-LM/CLM/blob/main/src/clm/schema.py)
   builds state from context + question and action from independent answer text
   or its description. There is no mandatory shared action prefix. Our branches
   follow this separation. State suffix ` answer: ` comes from native Samsone;
   prefill equivalence is checked below.
4. [train/finetune.py](https://github.com/Contrastive-LM/CLM/blob/main/train/finetune.py)
   supports bidirectional InfoNCE with distinct candidates/multiple reverse
   positives and per-question soft/hard CE. CE alone is therefore not a misuse
   of CLM. Our code retains both objectives, including original choice InfoNCE.
   On AudioMCQ 50K, that variant produced validation 56.84% and correct−random/
   silence gaps +7.62/+6.30pp; retained CE yielded 62.65% and +6.15/+5.91pp.
   This is a historical AudioMCQ control, not a matched FMA-objective experiment.
5. README describes broad state/action alignment, then hard negatives, then
   domain post-training with replay, and scaling results on a particular text
   benchmark. Those results do not guarantee that more musical data can remove
   frozen-representation limitations here.

## Fresh web-inference checks

Saved uploads match screenshots by duration/upload order; Umbrella also has
matching MP3 title/artist tags. Positions 114.33s and 121.68s are the visible
playback positions. Actual Analyze positions are cropped out of the screenshots,
so new probabilities are a nearby replay, not exact screenshot reproduction.

- Selected checkpoint SHA256:
  `a1cc52344117b320c420aefc7a61f82a89f3d546b3430ac39c77fbd0e8dab71b`.
- Fresh original MP3 decode is bitwise identical to the server's `.npy` for both
  uploads, maximum PCM error 0. Upload/decoding did not substitute audio.
- Exact 10s windows: Umbrella 109.33–119.33s, No White Flag 116.68–126.68s.
- HTTP and independent direct inference match: maximum probability errors
  3.64e-6 and 3.76e-7. Small text-batch numerical differences do not explain
  the genre changes.
- Hidden `[1,960]` is bitwise equal to the last-valid hidden from the first
  official native forward on both excerpts: maximum absolute error 0.
  The adapter receives the expected BF16→FP32 hidden; one shared adapter is
  used and head-input L2 norm is 1.
- FMA training changed 9,443,747/9,443,840 state-head parameters and
  9,443,712/9,443,840 action-head parameters. Changes do not prove quality,
  but exclude the claim that both heads remained frozen throughout.

Screenshot question: `what is genre of music?`. Five choices:
`pop / hip hop / rock / electronic / classic`.

| Excerpt | Selected FMA ranking | Same choices with `classical` replacing `classic` | Official native Samsone: `What genre is this music?` |
|---|---|---|---|
| Umbrella | classic 62.23%, pop 12.71% | pop 33.65%, electronic 30.91% | pop |
| No White Flag | pop 47.24%, classic 35.72%, rock 12.72% | pop 73.49%, rock 19.79% | pop |

`classic` is ambiguous and not automatically equivalent to `classical`.
However, replacing the word does not fix the whole task: No White Flag remains
pop. On Umbrella, the trained question `Which broad musical genre best describes this excerpt?`
with five canonical choices yields hip hop 35.73%, pop 29.54%. The problem
cannot be attributed solely to user wording or an unseen question.

For `describe`, official Samsone describes Umbrella as female vocal/pop with
bass/drums/keyboards/guitars, and the other excerpt as female vocal/pop with
guitar/bass/drums. These are independent generation diagnostics, not ground
truth; they do not exclude a native genre error on the second excerpt.
On silence, native genre inference says unclassifiable, while free description
hallucinates a constant source. Native inference is not assumed infallible.
Production ranking continues to use no generation.

Correct audio and silence produce substantially different state vectors:
cosine 0.5233 for Umbrella and 0.5694 for No White Flag under the screenshot
question. Silence with original choices yields classic 45.83%. Audio sensitivity
is real; correct genre identification does not follow automatically.

## Verified weaknesses in training/evaluation

- FMA training has four genre-question forms, deterministically coupled to
  choice wording: even IDs use canonical labels, odd IDs aliases. The coupling
  is visible in `scripts/prepare_genre_data.py`; its causal contribution still
  needs a controlled comparison.
- Track-level genres weakly supervise two 10s excerpts. That does not guarantee
  one clearly audible genre per excerpt. Transfer to commercial user-library
  tracks has not been measured on a separate representative benchmark.
- Reserved FMA genre macro is 42.41%, versus random 8.19% and silence 9.09%.
  Across the same 569 held-out recordings, per-question macro is
  53.73%, 48.57%, 52.10%, 40.56% for training forms, and 26.25%, 33.23% for
  unseen forms. Training has a real effect alongside pronounced language
  instability.
- Validation genre macro is 42.88% at selected epoch 6, versus 40.12% at epoch 32
  despite falling train objective. More identical epochs are not a sufficient
  demonstrated repair.
- Native decoding can use all prefill tokens and autoregressive steps. Its
  successful answer does not establish that one pooled hidden suffices for
  discriminative ranking. No architectural impossibility has been demonstrated.

## Next controlled experiment

1. Fix an independent user-like corpus with human labels on actual excerpts,
   including ambiguous/mixed genres. These already-inspected tracks become
   regression cases rather than fresh test examples.
2. Decouple question/choice wording and train independent combinations of true
   semantic aliases, longer descriptions, candidate subsets, and order.
3. Compare CE with original multi-positive bidirectional CLM loss on one music
   validation set, matching initialization/budget and audio controls. Repeated
   genre labels must not become false negatives.
4. Select for new-recording/paraphrase quality, audio dependence, and task
   retention. Start with a short pilot using the SAME adapter and original
   heads; frozen Samsone and the architecture remain unchanged.

Training remains paused at the user's instruction. This audit establishes
neither impossibility nor practical readiness for arbitrary musical search.

## Artifacts

[Two-track measurements](results/user-audio-audit/report.json),
[decode verification](results/user-audio-audit/decode_verification.json).
The full reproducer and 10-second WAV files are local under
`$SAMSONE_CLM_DATA/runs/local-demo/two-track-audit-2026-10-04/`.
No source audio or weights were copied into this report.
