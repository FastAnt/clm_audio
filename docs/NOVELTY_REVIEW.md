# Scientific novelty of Samsone–CLM

Search date: 2026-10-04. Sources are primary papers, official repositories, and
model cards. The search covered audio/text contrastive learning,
question-conditioned audio QA, frozen backbones, lightweight projection heads,
and exact Samsone/CLM combinations. Not finding an exact match does not establish
priority. This review is not a comparative training experiment.

## Assessment

Ranking audio/text pairs by embedding similarity is established. Question
conditioning, contrastive audio-MCQ supervision, and heads trained on frozen
encoders also have close antecedents. This project's potential contribution
is a controlled transfer of pretrained textual CLM decision heads to one small,
frozen Samsone356M audio-language backbone. Practical value depends on quality,
robustness, and full training/inference cost; a new combination of model names
alone is not a strong scientific claim.

The working architecture is unchanged:

```text
audio + question -> frozen Samsone356M -> shared adapter -> pretrained CLM state_head -> z_state
choice text      -> frozen Samsone356M -> shared adapter -> pretrained CLM action_head -> z_action
score = positive_scale * cosine(z_state, z_action)
```

Heads provide initialization and are retrained; the original CLM space is not
held fixed. The Qwen backbone is unused. Training does not run the encoder,
but frozen-feature precomputation is part of the total experiment cost.

## Closest related work

| Work | Overlap with this project | Implication |
|---|---|---|
| [CLAP, 2022](https://arxiv.org/abs/2206.04769) | Contrastive audio/text alignment and zero-shot classification using text labels | Arbitrary-label cosine ranking is established |
| [MuLan, 2022](https://arxiv.org/abs/2208.12415) | Joint music/text embeddings, tagging, retrieval | Musical specialization of embeddings is established |
| [TP-CLAP, 2026](https://arxiv.org/html/2607.25085) | Audio+question representations matched with separate answer embeddings, AudioMCQ supervision | Closest functional audio-QA predecessor, with different backbone/fusion |
| [AuroLA, 2026](https://arxiv.org/html/2602.18010v1) | A multimodal language model repurposed for audio/text embedding retrieval | Audio-language-model retrieval backbones have precedents; main training uses LoRA |
| [FIGMA, ACL 2026](https://aclanthology.org/2026.acl-long.2197.pdf) | Frozen audio/text encoders and about 22M trainable projection parameters for music retrieval | Small heads on frozen encoders are not a separate new idea |
| [Original CLM, 2026](https://github.com/Contrastive-LM/CLM) | Separate state/action heads, cached candidates, contrastive decisions | System One readout/cache design is inherited; multimodal support is on the roadmap |
| [Samsone, September 2026](https://arxiv.org/abs/2609.21666) | Small open audio-language backbone intended for devices | A recent backbone, but compact inference is already the authors' objective |

TP-CLAP uses 310K sound/music AudioMCQ question–choice pairs, initially freezes
encoders, then fine-tunes all parameters. The paper reports 219M parameters.
Our model is approximately 380M: smaller than original CLM-8B, but not smaller
than every audio ranker. TP-CLAP adds question conditioning through cross-attention;
our model lets Samsone process the question. FIGMA keeps roughly 800M backbone
parameters frozen and trains about 22M head parameters. These differences need
experimental comparison.

## Potential research hypothesis

Can pretrained discriminative heads from a textual CLM support useful,
audio-dependent decisions on a small frozen audio-language model after replacing
the backbone?

The reviewed primary sources did not contain the exact combination of
Samsone356M, one shared adapter, and pretrained CLM state/action heads.
Scientific value requires demonstrated transfer and a useful quality/cost
tradeoff. Original CLM initialization has not yet shown a controlled advantage:
loading pretrained weights and reducing loss do not isolate that contribution
from learning the adapter and heads.

A promising application is one local model answering different questions about
an audio clip: instruments, vocals, events, rhythm, and other attributes, with
cached answer embeddings. A genre classifier and such a general decision ranker
need different evaluations.

## Relation to observed failures

[RobustCLAP / NAACL 2025](https://aclanthology.org/2025.naacl-short.76/) documents
sensitivity to language variations and uses multiple paraphrase views for
robustness. It is related to fixing question/choice-alias sensitivity here,
but its retrieval experiments do not automatically establish the same effect
for our QA objective.

The [AudioMCQ paper](https://arxiv.org/abs/2509.21060) directly studies correct
answers obtainable from text without audio. Audio controls and protection
against textual shortcuts are necessary, but not themselves novel contributions.

Our FMA validation genre macro improved 16.73% → 42.88%, with reserved test
42.41%; a user-uploaded excerpt showed strong dependence on choice wording.
[The genre report](results/music-genre/REPORT.md) records the former;
the local diagnostic is
`$SAMSONE_CLM_DATA/runs/local-demo/genre-diagnostic-2026-10-04/report.json`.
FMA numbers cannot be compared directly to MMAU/MMAR accuracy in other papers.
Neither superiority over prior methods nor a robust general Audio System 1
has been demonstrated.

## Evidence still needed

1. Isolate pretrained CLM initialization with matching architecture, dimensions,
   backbone, data, and budget, changing only head initialization. This control
   has not run; the main model retains official initialization.
2. Evaluate question-conditioned QA on common external benchmarks and held-out
   recordings while auditing training/pretraining overlap. Match clips,
   choices, splits, and protocols across comparisons.
3. Measure audio contribution and stability across semantically equivalent
   questions/answers, candidate order, and candidate-set size. Ordinary
   accuracy is insufficient.
4. Compare full precompute+training cost, memory, state/choice latency with and
   without cache, throughput, and quality on the same hardware.
5. Test varied audio tasks. Fixed-label genre performance does not establish
   a generalizable decision model.

Assessment: low novelty in the general principle; a potentially interesting
cross-backbone transfer experiment; quality/efficiency advantages remain
unproven. A defensible description is a compact question-conditioned audio
decision model and a study of pretrained-head transfer, with explicit controls
and measured limitations.
