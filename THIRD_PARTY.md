# Upstream dependencies and checkpoint provenance

Official code is preserved as Git submodules; there are no local edits to either upstream repository:

| Repository | Pinned commit | Used here |
|---|---|---|
| [SamsungLabs/samsone](https://github.com/SamsungLabs/samsone) | `a71e77ae9528ef411ae26cdeb02d7b90cc18a50c` | Frozen Samsone356M, native audio/text paths and diagnostic inference |
| [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) | `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7` | Official state/action head definitions and pretrained heads checkpoint |

CLM's Apache-2.0 license is included at `vendor/CLM/LICENSE`; its official model card also declares Apache-2.0 for the reference heads. The pinned Samsone checkout does not contain a standalone LICENSE file; its original notices/files are preserved in the submodule. Public availability alone is not a license grant. Dataset and checkpoint provenance remains attached to the source artifacts. This document does not relicense upstream code or data. See the [checkpoint and artifact notes](docs/ARTIFACTS.md) for outstanding licensing questions.

Official weights, fetched by `scripts/download_models.py` and validated before use:

| Weight | Bytes | SHA256 |
|---|---:|---|
| Samsone356M `samsone_356m_model_bf16.ckpt` | 713899355 | `8ed28e35aff5cb724bc84d1b2ea1559492e88e9c53abd1439f7e0a397214fea8` |
| CLM `CLM_v0.1-8B.pt` (heads only) | 75557149 | `b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5` |

The `8B` in the CLM checkpoint name describes its original training provenance. No Qwen8B weights are downloaded or instantiated by this project's pipeline.

TorchCodec 0.11 uses the official CPU wheel index for audio decoding. Default Linux x86 PyPI wheels require additional CUDA NPP libraries absent from the RunPod image; GPU video decoding is unnecessary here. PyTorch/Samsone inference and training remain CUDA. Installation source: [official TorchCodec instructions](https://github.com/meta-pytorch/torchcodec#installing-torchcodec).

General QA annotations come from `inclusionAI/AudioMCQ`, strong subset. Transport mirrors provide source audio only. Original source IDs, revisions, hashes, split identity and repair records are preserved. Music preparation and training subsequently completed; source formats/provenance are documented in `docs/MUSIC_DATA_EXPANSION.md`, `docs/REMOTE_PROGRESS.md`, and the musical result reports. Dataset-card licenses do not replace the licenses of underlying audio recordings.

Project-owned code and documentation in this export are Apache-2.0; see
[LICENSE](LICENSE) and [NOTICE](NOTICE). This grant does not cover third-party
code, source audio, annotations owned by others, or model weights.
