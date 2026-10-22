# Initial Stage6 checkpoint

Completed remotely on RTX3090 on2026-10-04. Training190653music-only QA; validation3005; fixed test3068. Shared adapter and both original CLM heads trained; Samsone356M stayed frozen and was never called inside training. Raw Stage5 best initializes this stage. Overfit512 passed100%, CE0.000098 before large training.

Main training6epochs/4470updates/242.85seconds; selected first epoch/step745 by validation. Adapter LR1e-4, heads5e-5, score scale1e-3, AdamW, batch256, candidate CE+pair CE+swap. Scale calibrated on validation only to59.98. Checkpoints and features returned locally and SHA256 verified.

| Fixed test | Correct audio | Random audio | Silence |
|---|---:|---:|---:|
| Overall3068QA | 60.63% | 34.49% | 34.75% |
| Equal-family balanced | 61.15% | 38.71% | 39.09% |

Perception balanced82.42%; structure39.88%. Instrument family60.43%, source51.56%, vocals93.75%, key mode57.81%, tonic8.98%, meter57.81%, tempo35.94%. Tonic is approximately chance among12choices; this task is not solved. Genres have small held-out cohorts. Jamendo full-track labels on excerpts are weak supervision; MusicBench structure labels are automatically estimated and filtered.

Audio dependence is substantial overall and passes both5pp gates in each family; structural gains7.57/6.55pp are smaller than perception gains37.32/37.58pp. These results support audio use, not universal musical understanding.

Live full-file ranking checked19task examples; max cached/live probability difference0.000240. HTTP demo additionally passed9fixed ten-second windows around start/middle/end, including padding and range seek. Warm75–138ms; first813ms while another GPU training job was running. Cropped functional checks do not measure full-clip label accuracy.

Raw best SHA256 `0c8362478011dfe0e245fae82721b4e0b1747a007e54c420017fed6306aaec88`. Calibrated SHA256 `89dd6ac9e8286f29c3db82e8179d50c36e29e99e3ac02108c7abab72f5bb3132`. Local export `/srv/data/samsone_clm/models/samsone_clm_music`.

A separate32-epoch Stage6 without early stopping subsequently completed with the same parent/features/hyperparameters, followed by serving-window adaptation. This initial export is preserved as `models/samsone_clm_music_stage6_initial`; current selection and limitations are in [the final report](../music-stage6-window/REPORT.md). Test did not select the version.
