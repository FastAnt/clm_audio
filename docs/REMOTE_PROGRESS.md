# Remote continuation and musical specialization

Backup completed on 2026-10-04: 428 trained-artifact files / 20.47GB were returned
and SHA256-verified. Common/music models were pushed to Git LFS and verified
by downloading them again. Datasets/audio/features were excluded from this
backup at the user's request and removed remotely after verification. The web
process was stopped and the GPU released for another project.
[Evidence](results/backup-2026-10-04/), [restoration](MODEL_BACKUP.md).

Recorded state on 2026-10-04: FMA Large v2 completed and passed validation
acceptance; `samsone_clm_music` was updated while common stayed unchanged.
Selected checkpoints were published in Git LFS, and trained runs/exports
were backed up offsite without datasets/audio/features.
[Final music report](results/music-genre/REPORT.md),
[models and backup](MODEL_BACKUP.md). The chronology below is historical;
references to running jobs describe their state at each recorded milestone.

Updated 2026-10-04. Project: `/home/ubuntu/dev_clm`; remote data: `/home/ubuntu/dev_clm/data`. Local training remains paused; remote work is authorized by the user's migration request.

## Stage 4 outcome

RTX3090,24GB VRAM; container RAM limit116.42GiB; disk500GiB. RAM checks account for cgroups rather than trusting the host's `/proc/meminfo`.

1894 Stage4-first files (3.584GB) transferred and SHA256 verified. Official Samsone weights downloaded with their pinned SHA256. Fresh x86 Python3.12 environment uses the lockfile, official CPU TorchCodec decoding wheels and CUDA PyTorch.

Additional Stage4 finished at6epochs/17028updates/419seconds, early stopping4. Both heads and adapter changed; encoder calls during training=0. The new run is separate from immutable common exports.

| Validation | Original common | New remote selected |
|---|---:|---:|
| Top1 | 65.19% | 64.31% |
| Correct−random | 5.52pp | 5.18pp |
| Correct−silence | 7.18pp | 5.27pp |

**Original common is retained as Stage5 parent based on validation.** The new checkpoint was calibrated on validation, then evaluated once on fixed test: Top1 63.87%, random60.35%, silence58.59%, gaps3.52/5.27pp. Test does not alter the parent decision; the strict test gate remains unmet.

All remote run weights/logs/metrics returned locally to `$SAMSONE_CLM_DATA/runs/remote-stage4`; all13 remote SHA256 values verified. PNG generated from returned curves. [Small evidence](results/remote-stage4/migration_summary.json).

## Batch measurement

Disposable clones; actual candidate CE+pair CE+swap+backward+clip+AdamW. FP32 heads, BF16 cached features, TF32 disabled; 10warmup+200measured steps; pair counts scale with main batch. No encoder calls.

| Main batch | Main QA/sec | Peak allocated GiB |
|---|---:|---:|
| 64 | 2812 | 2.17 |
| 128 | 5432 | 2.22 |
| 256 | 11293 | 2.36 |
| 512 | 13344 | 2.61 |
| 1024 | 14451 | 3.19 |

Music default256. This measures throughput, not convergence, with4-choice common data. Music has2–12candidates. Bigger batches give fewer updates/epoch; proposed max32music epochs with early stopping. Precompute batch is a separate memory decision. Reproduce using `scripts/benchmark_batch_size.py`; no real checkpoint is overwritten.

## Musical preparation (music-only training)

Source downloads, preparation/auditing and frozen precompute finished. **Stage5, Stage6, full32 and both window continuations completed. All 73,450 referenced musical source files and 16,528 window WAV passed local SHA256 verification.** The validation-selected lower-LR window model is published as `samsone_clm_music` on both hosts. GPU training has ended; the real-audio demo remains running on the same remote host. [Final report](results/music-stage6-window/REPORT.md), [testing commands](TEST_MUSIC.md).

- MTG-Jamendo clean unanimous3-annotator classification labels;10671 referenced files downloaded individually with published per-track SHA256. Artist splits precede central30s excerpts. Full-track labels projected onto excerpts are weak, with scope recorded explicitly.
- Official NSynth JSON/WAV,20notes/instrument auxiliary selection. Keep official train; official valid/test share53held-out instruments, so group those instruments into two distinct evaluation splits. Exclude PCM shared across splits and contradictory same-PCM answers. Full4s-note family/source labels remain aligned.
- MusicBench pinned revision and published archive/metadata SHA256.52768metadata rows contain42611distinct audio locations; duplicate captions do not become independent QA. Augmentations share YouTube parent splits; existing common parent splits take precedence. BPM must agree with beat spacing; meter requires cycles; key confidence≥0.8. Automatic labels are marked as such. Unconfident chords remain audit metadata.

No uploader-tag absence becomes a negative. Captions never enter state prompts. The latest user instruction makes Stage5/6 training music-only; speech/sound/temporal common replay is excluded. Each QA stores source/split group, PCM/container hashes, label scope/evidence. Random audio uses a different source parent and PCM, rather than another augmentation of the same recording.

Code supports Stage5/6 with the SAME shared adapter and both original heads trainable. Ragged candidates use masks without mutating hashed manifests. Music metrics balance semantic answer texts, not shuffled option columns. All task metrics are reported; selection prefers tasks with≥50validation samples.

`compose_music_features.py` joins verified cached features and replay without190GB common audio or rerunning Samsone. Proposed mixtures: Stage5 80%new/20%AudioMCQ-music; Stage6 45%structure/45%Stage5/10%AudioMCQ-music, capped by independent QA. Reports show actual mixture counts. Component audio ablations retain their matched cohorts.

Canonical musical QA: Jamendo60981, NSynth42344, MusicBench111561, with0 measured split leaks. NSynth official validation/test share53instrument identities: those held-out instruments are repartitioned into distinct validation/test groups; official train instruments remain train. Source-parent and PCM duplicates are excluded across splits.

Native official Samsone356M caption and two-clip comparison ran successfully on real NSynth WAV. Final hidden width960, finite; frozen FP32 extraction withBF16 storage.30s-clip precompute benchmark: batch64 ≈48.7states/s,3.05GiB peak allocated; selected64. Training batch256 is a separate decision.

Sequence: frozen precompute →512-sample overfit with bothheads → Stage5 → calibration/test/live-file check → locally return/verify all run files → MusicBench precompute → Stage6 with Stage5 held-out retention → final evaluation and `samsone_clm_music` export. Larger training stops on a failed overfit gate. Resource monitor can stop only verified project worker groups, RAM floor15GiB/disk floor20GiB. Only completed project archives receive file-cache discard advice; no global cache dropping.

Supervisor: `scripts/remote_music_pipeline.py`; local `$SAMSONE_CLM_DATA/runs/remote_migration/music_status.json`, mirrored remote `data/runs/music/status.json`; detached phase logs/exit codes in remote `data/runs/music-preparation/`. Restart adopts an existing live project worker rather than launching it twice. Successful return markers contain verified SHA256 and gate progression. Final model and frozen feature stores are checked again on local return. Source archives are reproducible and need not be returned; prepared musical audio/QA/features are returned.

Entry points: `download_music.py`, `prepare_music.py`, `compose_music_features.py`, `train_music.py`, `monitor_resources.py`. Musical work runs on the authorized remote host. This preparation snapshot must not be interpreted as musical model results.

## Stage5 result

100424music-only train QA: NSynth38112 + Jamendo50689 + cached AudioMCQ-music11623. Overfit512 first constant-LR attempt stopped at89.45%/CE0.263; retained. Lower annealed LR passed at100%/CE0.00195, then large training started fresh from raw common. Both heads/adapter train; encoder_calls0.

Stage5 stopped after25epochs/9825updates/492.6seconds. Selected epoch20/step7860 by balanced musical validation. Validation Top1 77.74%; single scale validation-only84.43→59.76; validation ECE0.0220.

Reserved test2044QA: correct76.52%, random39.19%, silence41.10%; gaps37.33/35.42pp. Balanced musical task macro82.11%, with37.38/37.26pp gaps; this average gives each eligible task equal weight and is distinct from overall accuracy. Instrument family57.58% on561QA; source55.73% on192QA; vocals93.75% on128QA. Genre taxonomy cohorts22–49QA are too small for strong general claims. Jamendo full-track consensus on central excerpts remains weak supervision.

Live Ranker checked15real test files, maximum cached/live probability difference0.000174. All12large-run file SHA values, feature shards and successful/failed diagnostics verified on local return before Stage6. Best SHA `21006be9f0de3dee0fe783797945bb7dd6cc1d005b6c49b57c6bb33f2043510b`. Common unchanged. [Stage5 evidence](results/music-stage5/evaluation_test.json), curves (historical image omitted from this text-only export).

Prepared WAV decoding avoids per-sample ffmpeg subprocesses for verified native mono16k FLOAT/PCM16 WAV/WAVEX. Real-file PCM and encoder outputs matched bit-for-bit. Jamendo1.81×/MusicBench2.44× measured speedups; all other encodings retain ffmpeg fallback. Durable precompute shards survive restarts and seed the representation cache. [Measurements](results/music-preparation/jamendo_native_wav_benchmark.json).

## Stage6 initial result and additional run

190653music-only train QA: MusicBench90229 + NSynth/Jamendo88801 + AudioMCQ-music11623. Stage5 held-out tasks are retained. Overfit512 passed at100%/CE0.000098. Main training completed6epochs/4470updates/242.9seconds, best epoch1/step745; both heads and adapter changed, encoder calls0. Best SHA `0c8362478011dfe0e245fae82721b4e0b1747a007e54c420017fed6306aaec88`.

Validation3005QA: Top1 62.10%, equal-family balanced63.11%. Validation-only scale82.91→59.98. Fixed test3068QA: correct60.63%, random34.49%, silence34.75%; ECE0.0272. Equal-family balanced61.15%, perception82.42%, structure39.88%. Structure audio gaps7.57/6.55pp: audio contributes, but tonic8.98% across12choices is near chance; tempo35.94% across4bins is modest. Automatic structural labels remain a limitation. Stage5 and Stage6 overall accuracy are not comparable because their tasks and candidate counts differ.

Live API19real full files passed, max cached/live probability delta0.000240. Real HTTP demo tests passed9ten-second windows at beginning/middle/end, including silence padding and HTTP206seek. Warm inference75–138ms, first813ms under simultaneous GPU training; functional checks do not establish crop-label accuracy. All12run files, feature6 shards, diagnostics and initial musical export returned with verified checksums. [Initial Stage6 evidence](results/music-stage6/REPORT.md).

`stage6-full32` held architecture, parent, features, LR, objective and batch fixed; only patience changed5→0. Completed32epochs/23840updates; validation selected epoch12/8940updates, balanced63.51%. Fixed test correct61.57/random34.16/silence34.75%; balanced perception81.10/structure41.44%, tonic14.84%, tempo35.16%. The initial export is preserved. [Selected Stage6 evidence](results/music-stage6-full32/REPORT.md).

A256-QA validation diagnostic measured the full-excerpt→central10s mismatch: instrument family50→28.13%, vocals96.88→65.63%. `run_music_window_pipeline.py` prepares mixed train views, precomputes frozen representations, verifies a fresh overfit512, then trains both heads and adapter. Source splits and labels are inherited; note padding preserves notes, Jamendo crops remain weak supervision. QA count stays190653. Both full and crop validation views are reported; acceptance requires full balanced retention within1pp, window gain≥1pp, and all view/family audio gaps≥5pp. The full32 export remains the fallback, and test cannot select the continuation.

Window preparation 48,316 QA / 16,528 files passed split audit. A resource guard stopped work at 13.65 GiB available; advising only completed project archive/WAV cache pages restored 80.52 GiB without deleting data. Frozen precompute restarted safely and finished. The composed dataset keeps 190,653 train QA; paired views yield 4,858 validation / 4,984 test rows, without increasing independent original QA support.

Both 32-epoch window runs completed. First selected epoch2; the smaller LR run selected epoch8/step5960. LR adapter3e-5 / heads1.5e-5 / scale3e-4, batch256,128pairs/update, CE+pairCE+swap. Selection uses validation only, all view/family audio gaps≥5pp, each full-family retention within1pp and window gain≥1pp. Task-support eligibility counts original QA once across views; the earlier model was re-evaluated under the same corrected metric before comparison. Selected mixed balanced validation61.83→62.25%, window78.83→79.34%.

Fixed selected test: full3,068QA correct59.84/random35.04/silence34.71%; paired window1,916QA correct74.69/random37.89/silence39.77%. Window balanced perception80.15%, full perception80.84/structure40.43%. Full test fell from full32's61.57%; all prior exports remain preserved. Tonic13.67%, tempo30.86% remain weak. Jamendo excerpt labels are weak; crop structure was not independently evaluated.

All12run+9export files per window version SHA256 verified locally before publication. Current raw SHA `c9c00f307c12828eee9e66893bd3e7f4c6eba3a2a477d8e4ff4bdaafeed7f694`, calibrated `d6219a325a26a45b9e2c7abe2eef1f7e68adb44cc58529287d1e5cc0485fb2c7`. Live19files max probability difference0.000150; HTTP9windows passed range seek/padding/order/exact SHA. Pytest61passed locally and remotely. GPU training is completed; common remains unchanged.

User-uploaded genre example was reproduced on exactly158.71–168.71s. Old ranking gives classic instrumental59.58%; the same old model with canonical genre prompt/options gives pop55.64%, but the user's question with the same expanded options gives rap hiphop44.52%. The new model still fails the original case. This is a measured prompt/options generalization limitation, not repaired by more epochs. Only1,483 explicit new-source genre QA across the four taxonomies were trained; AudioMCQ music replay additionally contains1,274 genre/style keyword questions of heterogeneous types. [Diagnostic](results/music-stage6-window/user_genre_diagnostic.json), [counts](results/music-stage6-window/train_task_counts.json), [proposed genre follow-up](MUSIC_GENRE_FOLLOWUP.md). Historical status before the new request: no new genre training had been launched. The new remote continuation below supersedes that status.


## New FMA genre continuation, 2026-10-04

Remote-only work resumed after the user requested more data and genre training.
FMA Medium canonical preparation finished:21,958valid recordings,17,529train
recordings;140,232train/13,224validation/13,350test QA views; split leaks0.
Published archive SHA1 and actual SHA256 checked. FMA Large93.42GiB downloaded
and verified;45,129metadata-eligible primary-genre recordings are being checked.
Frozen Samsone feature extraction runs on the GPU while Large preparation runs
on CPU. Head optimization has not started in this snapshot.

The autonomous controller subsequently runs exact-feature overfit512, genre v1
training, validation acceptance/calibration/test/live-file checks, then Large v2.
Both CLM heads and shared adapter train; common stays immutable. Existing exports
and completed evidence are preserved. New artifacts remain remote under the
latest user instruction; no workstation processing or transfer is launched.
[Recipe and monitoring paths](GENRE_CONTINUATION.md).
