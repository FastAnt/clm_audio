# Portfolio and CV description

## Positioning

Describe Samsone–CLM as an applied ML research and engineering project on
non-generative audio–text ranking. The implementation combines existing
Samsone and CLM components; it is not a foundation model trained from scratch.
Attribute those components and distinguish the pretrained-head transfer
experiment from the separate native-sequence experiment.

The strongest demonstrated work is the reproducible pipeline: source audio
retrieval and integrity checks, recording-disjoint datasets, frozen-feature
caching, staged head/adapter training, ablations, calibration, reproducible
checkpoints, migration/backup, and a FastAPI demo.

## Suggested CV entry

**Samsone–CLM — Audio–Text Ranking Research Project**  
Python · PyTorch · FastAPI · contrastive learning · audio ML

- Built a reproducible audio–text ranking pipeline using frozen Samsone-356M,
  a shared adapter, and pretrained CLM state/action heads, with staged training
  on 181,592 general-audio QA and a 404,473-example musical training mixture.
- Implemented FP32 feature extraction with BF16 sharded storage, recording-level
  leakage checks, random-audio/silence ablations, validation-only calibration,
  resumable checkpoints, and an interactive 10-second-window FastAPI demo.
- Evaluated genre generalization and documented failure modes: the FMA music
  model reached 42.41% class-balanced test accuracy over 11 genres versus
  8.19% with random audio; robustness to question and choice wording remains
  an open issue.

The QA counts above are training examples, not independent recordings. The FMA
test contains 569 independent recordings with six question views, totaling
3414 QA. The metrics refer to `samsone_clm_music`, not the separate
`samsone_clm_music_v2` experiment.

For a shorter CV entry, keep the first two bullets and link to the results
report. Include only work you can explain and accurately attribute; mention
AI-assisted development when relevant rather than implying unsupported
individual contributions.

## Evidence to link

- [Implementation and setup](../README.md).
- [Music results and evaluation scope](results/music-genre/REPORT.md).
- [Official inference and hidden-extraction audit](SAMSONE_AUDIT.md).
- [Reproducible user-track failures](CLM_TRANSFER_REVIEW.md).
- [Checkpoint publication and verified backup](MODEL_BACKUP.md).
- [Checkpoint and artifact notes](ARTIFACTS.md).

A research prototype with measured negative results can demonstrate useful ML
engineering and experimental judgment. Claims of production readiness, broad
genre reliability, novel audio/text contrastive learning, or state-of-the-art
accuracy are not supported by the current evidence.
