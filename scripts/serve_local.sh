#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SAMSONE_CLM_DATA="${SAMSONE_CLM_DATA:-/srv/data/samsone_clm}"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
local_python="${SAMSONE_CLM_PYTHON:-$project_root/.venv/bin/python}"
if [[ ! -x "$local_python" ]]; then
  printf 'Set SAMSONE_CLM_PYTHON to the prepared Python environment.\n' >&2
  exit 1
fi

exec "$local_python" -m samsone_clm.server \
  --checkpoint "$SAMSONE_CLM_DATA/models/samsone_clm_music/calibrated.pt" \
  --device cpu --host 127.0.0.1 --port 8800 "$@"
