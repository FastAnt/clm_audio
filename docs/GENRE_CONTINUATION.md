# Genre continuation on the remote GPU

User authorized new music data and training on 2026-10-04 after the uploaded genre
example failed. All new download, preparation, precompute, training, tests and
artifacts stay on /home/ubuntu/dev_clm; the workstation runs a different workload.
The earlier local artifact returns remain valid historical evidence.

## Data and supervision

Official [FMA](https://github.com/mdeff/fma) metadata and medium/large audio archives
are downloaded with resume, published SHA1 and measured SHA256 verification.
Medium preparation completed: **21,958 valid recordings**, including **17,529
training recordings**. Canonical rows: train **140,232**, validation **13,224**,
test **13,350**; total **166,806**. These are question/window views, not 166,806
independent tracks. Source/PCM/artist splits are checked and protected old-corpus
PCM overlaps are excluded. Split leaks: zero.

Eleven official primary genres are used: blues, classical, country, electronic,
experimental, folk, hip hop, jazz, pop, rock, soul/rnb. Instrumental,
geographic and spoken labels are not treated as mutually exclusive music genres.
FMA Large has **45,129 metadata-eligible recordings** before audio validation;
report its final validated count separately. The archive contains additional
recordings without a usable primary genre.

Train recordings supply two exact 10-second windows and four question forms.
Validation/test use central 10-second windows and six question forms; two forms
are unseen in train. Train choice aliases include, for example, pop music and
rap hiphop; semantic labels remain canonical. Track-level genre projected to a
local window is explicitly weak supervision. It does not guarantee the genre is
equally apparent throughout every recording.

Evaluation ablations select whole recording/question groups with round-robin
semantic classes. Question robustness is compared on the same recordings, with
all six forms retained; a frequent genre cannot remove rare genres from the
audio-control cohort. The composer caps at 64 available recordings per genre.
All preserved replay validation/test cohorts are retained.

Existing MTG-Jamendo human genre annotations cover a much smaller pool than all
its audio. More positives must use actual annotation confidence. Missing tags
must never become asserted negative genre labels. This continuation first scales
the independent FMA pool; MTG expansion needs a separate quality-audited manifest.

## Fixed model and objective

Frozen official Samsone356M for both branches; FP32 feature extraction and BF16
sharded storage. The same shared adapter and original pretrained CLM state/action
heads continue from samsone_clm_music_window_lowlr/best.pt. No Qwen model is loaded.
Both heads and adapter train; Samsone is absent from the training loop.

Each iteration composes FMA with music replay, checks overfit512 on the exact
feature manifest, then runs up to 32 epochs:
batch256; adapter LR3e-5; heads LR1.5e-5; scale LR3e-4; OneCycle;
candidate CE + pair CE0.5 + swap1.0/margin0.05, 128 verified pairs/update.
New-genre CE uses normalized inverse-square-root class-frequency weights;
replay CE retains weight1.0. Class weighting affects the main CE, not reported
unweighted evaluation or the existing pair objective.

Checkpoint selection uses validation only: genre macro gain >=2 percentage
points, worst question form gain >=1 point, every question's correct-minus-random
and correct-minus-silence gap >=5 points, replay perception/structure retained
within1 point. Failed candidates are retained as separate runs and cannot replace
the working model. Selected models are calibrated only on validation; test runs
once after acceptance, followed by real-file live/cache checks.

## Autonomous execution

    cd /home/ubuntu/dev_clm
    export SAMSONE_CLM_DATA=/home/ubuntu/dev_clm/data
    .venv/bin/python scripts/run_genre_pipeline.py

The controller adopts the existing medium precompute worker, computes improved
grouped ablations by reusing verified correct-audio features, then composes,
overfits, trains and finalizes v1. FMA Large CPU preparation overlaps the GPU
work; large precompute and v2 training follow in sequence. There is one GPU
training/precompute worker at a time.

Logs/PID/status: SAMSONE_CLM_DATA/runs/music-genre/. Controller heartbeat also
updates runs/music/refinement_status.json, which the already-running demo reads.
Model runs are runs/music-genre/iteration-v1/stage6-genre-v1 and the v2 equivalent.
Accepted exports are models/samsone_clm_music_genre_v1 and ..._v2. Publication
archives the previous music export and atomically switches the demo checkpoint.
Immutable samsone_clm_common is checked before/after the controller.

RAM/disk are checked before jobs. A separate resource guard polls every15seconds
and stops only verified project worker groups below15GiB RAM or20GiB disk.
No global cache dropping, local compute or automatic environment upgrades occur.
A stopped controller can adopt live workers and completed exit records; failed
jobs require inspection before retry. Latest phase and actual progress are the
authoritative status; starting precompute is not starting head optimization.

## Dashboard history

The phase banner reports the current controller job. Graphs and PNG downloads
refer to the explicitly selected experiment. During FMA Large precompute, the
default preserves the completed FMA Medium curves; its32epochs are available as
Stage6 / FMA Medium / genre v1. The selector retains earlier full/window music
runs and switches to the new optimizer run when it starts.

Historical Stage1–3 metadata can be restored from the checked-in v3 evidence:

    .venv/bin/python scripts/restore_dashboard_history.py

This restores reports and curves, not absent tensor weights. Known experiment
IDs are used for PNG requests; arbitrary filesystem paths are rejected.
Existing audio uploads survive a web-process restart using the persisted
original file and decoded array. Training/precompute workers run independently
of the demo.
