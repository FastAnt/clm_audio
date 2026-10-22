# Selected Stage6 full-excerpt model

RTX3090, 32 epochs, 23,840 updates, 190,653 music-only train QA. The only difference from the first Stage6 run is patience 5→0; architecture, parent, features, LR, seed, batch and objective match. Validation selected epoch 12/step 8,940. Equal-family balanced validation improved 63.11→63.51%. Test did not select the version.

| Fixed test, 3,068 QA | Correct audio | Random audio | Silence |
|---|---:|---:|---:|
| Top-1 | 61.57% | 34.16% | 34.75% |

Correct-audio CE 1.0039, ECE 0.0215. Equal-family balanced accuracy 61.27%; perception 81.10%, structure 41.44%. Structure audio gaps 8.90/8.11 pp. Instrument family 60.78%, source 54.69%, vocals 93.75%; mode 57.81%, tonic 14.84%, meter 58.59%, tempo 35.16%. Tonic improved from 8.98% but remains weak; tempo also remains modest. Small genre cohorts do not support broad claims. Automatic structural annotations and full-track Jamendo labels on excerpts remain limitations.

Calibration fits validation only: scale 67.56→44.23 without changing rankings. Live full-file API checked 19 tasks, max cached/live probability difference 0.000072. HTTP checks passed nine 10-second windows, range seeking, silence padding and choice permutation. All run/export checksums were verified locally. Raw best SHA256 `18d568e067ea0a6bf4497e415284afdb11c89cce03dbeb6e20b3680701ad397e`; calibrated SHA256 `e6e51f492c52d0f81e74fc079e0c193764b452e405f2e50c983783a826e95ad8`.

This immutable full-excerpt export is now preserved as `models/samsone_clm_music_full32` and `models/samsone_clm_music_full32_before_windows`. Canonical `models/samsone_clm_music` now points to the completed window-adapted version described in [the final report](../music-stage6-window/REPORT.md). The first Stage6 version is preserved as `models/samsone_clm_music_stage6_initial`. All three music datasets and frozen feature stores were returned. SHA256 of all73,450referenced source audio files was verified; source archives remain reproducible on the remote host.

## Serving window diagnostic and continuation

256 validation QA, 32 per task, compared cached full-excerpt ranking to the exact padded central 10-second view. Instrument family fell 50→28.13%; vocals 96.88→65.63%; other tasks showed smaller differences. These are small diagnostic cohorts, and projected full-track labels on 10-second crops remain weak. No test labels were read and the chosen full-excerpt model was unchanged by this diagnostic.

An additional Stage6 continuation now adapts to serving windows. Audio-identity hashing selects half the NSynth/Jamendo train recordings to replace their full representation with an exact central 10-second view. Other recordings retain the full view; QA count remains 190,653. Source splits and answers are inherited. NSynth notes remain intact with silence padding. Jamendo labels explicitly record weak full-track→10-second projection. The original full held-out cohorts stay fixed, with paired crop views added separately; these are correlated observations, not new independent examples.

The parent is this selected raw Stage6 checkpoint. Samsone remains frozen; both original heads and the SAME adapter train. Acceptance requires all view/family audio gaps≥5 pp, each full-excerpt family's balanced accuracy drop≤1 pp and window balanced gain≥1 pp. Checkpoint selection then maximizes mixed validation quality. A fresh 512-sample overfit gate on the exact new feature manifest precedes training. The completed full-excerpt export remains the fallback.

Task audio ablations (historical image omitted from this text-only export)
