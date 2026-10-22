# Testing the music model

Stage 5, Stage 6, and FMA Large v2 completed on a remote RTX3090. The selected
music export is `$SAMSONE_CLM_DATA/models/samsone_clm_music/`. For inference,
use `calibrated.pt`. `best.pt` contains raw trained weights; `last.pt` contains
the final step's state for exact resume. Official Samsone356M and CLM checkpoints
must also exist under the data root; learned checkpoints do not include Samsone.

## Local demo

On the original prepared workstation, open <http://127.0.0.1:8800/> after starting
the server if necessary. The recorded configuration runs **locally on CPU**,
without an SSH tunnel, leaving the GPU available for other training.
Select general `samsone_clm_common` or music `samsone_clm_music` (FMA Large v2,
the default). Upload audio, set the playback position, enter a question and
choices, and click Analyze. The analyzed window is exactly 10 seconds: 5 before
the position and 5 after, with silence padding at the boundaries. Answers are
ranked without generation. Switching models preserves the playback position
and uploaded audio. The result identifies the actual checkpoint and device.
The first analysis after switching includes weight loading: 5–6 seconds on the
verification file, followed by approximately 0.9–1.5 seconds for subsequent
analyses. Latency depends on audio and candidate count.

The remote demo and project GPU processes were stopped. Following verified
local backup, RunPod project datasets, audio/features, weights, runs, caches,
and the Python environment were removed. Code was retained.
[Restoring trained models](MODEL_BACKUP.md).

Start with English questions and clear choices, for example:

- `Are there vocals in this music?` → `vocals`, `instrumental`.
- `What instrument family is heard?` → `keyboard`, `guitar`, `brass`, `string`.
- `Does this music sound acoustic or electronic?` → `acoustic`, `electronic`.
- `Which broad musical genre best describes this excerpt?` → `classical`,
  `soul rnb`, `rock`, `experimental`, `jazz`, `pop`, `blues`, `hip hop`, `folk`,
  `country`, `electronic`.

Initially try recordings with one clearly audible instrument. These examples
are suggested user queries, not promises of correct answers. Arbitrary choices
are supported, but probabilities are normalized over the supplied choices;
adding alternatives changes probabilities. Samsone's native text path is
limited to English/ASCII input; Russian questions are unsupported here.

## Python API

In a prepared Python environment:

```python
from samsone_clm.api import configure, rank

configure("/srv/data/samsone_clm/models/samsone_clm_music/calibrated.pt", device="cpu")
print(rank("example.wav", "What instrument family is heard?",
           ["keyboard", "guitar", "brass", "string"]))
```

On the original remote host, replace the data root with
`/home/ubuntu/dev_clm/data`. The API processes the whole supplied clip, up to
30 seconds; the web demo selects the 10-second window.

## Restarting locally

```bash
cd /opt/samsone-clm
bash scripts/serve_local.sh
```

The script uses this checkout and the existing environment without installing
packages. Port 8800 must be available; do not start a second instance if the
demo is already running. Equivalent command in a prepared checkout:

```bash
PYTHONPATH=src uv run --no-sync python -m samsone_clm.server \
  --checkpoint "$SAMSONE_CLM_DATA/models/samsone_clm_music/calibrated.pt" \
  --device cpu --host 127.0.0.1 --port 8800
```

On the original workstation, code lives at `/opt/samsone-clm`
and data at `/srv/data/samsone_clm`. The prepared Python executable
is `/srv/data/samsone_clm/cache/environment/bin/python`; this
checkout's `.venv` points to it. Override the executable with
`SAMSONE_CLM_PYTHON` if needed. Local Samsone–CLM training remains paused.
These paths describe the original installation and may not exist in a fresh
checkout.

## Interpreting results

Evaluate full clips and 10-second windows separately. Paired views of one QA
are not additional independent examples. Jamendo labels describe whole tracks
and transfer weakly to short windows. MusicBench structural labels are
automatic. Key and precise tempo remain weak. Functional HTTP verification
checks upload, seeking, padding, candidate order, and checkpoint selection;
it does not establish the correctness of musical answers.

FMA Large v2 genre balanced accuracy is 42.41% on the reserved 11-genre test;
new question wording performs worse. This does not guarantee recognition of
arbitrary genres. Exact results and hashes:
[final genre report](results/music-genre/REPORT.md).
