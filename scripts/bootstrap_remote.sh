#!/usr/bin/env bash
# Create a fresh host-specific environment; do not transfer the ARM .venv.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dry_run=false
while (($#)); do
  case "$1" in
    --data-root) export SAMSONE_CLM_DATA="${2:?Missing data root}"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) echo 'Usage: scripts/bootstrap_remote.sh [--data-root /absolute/path] [--dry-run]'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
export SAMSONE_CLM_DATA="${SAMSONE_CLM_DATA:-/srv/data/samsone_clm}"
[[ "$SAMSONE_CLM_DATA" = /* ]] || { echo 'Data root must be absolute' >&2; exit 2; }
export UV_CACHE_DIR="$SAMSONE_CLM_DATA/cache/uv"
export UV_PROJECT_ENVIRONMENT="$SAMSONE_CLM_DATA/cache/environment"
export UV_PYTHON_INSTALL_DIR="$SAMSONE_CLM_DATA/cache/python"
export HF_HOME="$SAMSONE_CLM_DATA/cache/huggingface"
export TORCH_HOME="$SAMSONE_CLM_DATA/cache/torch"
export TMPDIR="$SAMSONE_CLM_DATA/tmp"
export TOKENIZERS_PARALLELISM=false
cd -- "$project_dir"
if "$dry_run"; then
  printf 'Data root: %s\nEnvironment: %s\n' "$SAMSONE_CLM_DATA" "$UV_PROJECT_ENVIRONMENT"
  echo 'git submodule update --init --recursive'
  echo 'uv sync --frozen --python 3.12'
  echo 'Check PyTorch/CUDA and run pytest; no training starts.'
  exit 0
fi
for program in uv git ffmpeg; do
  command -v "$program" >/dev/null || { echo "Missing program: $program" >&2; exit 1; }
done
mkdir -p -- "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
git submodule update --init --recursive
uv sync --frozen --python 3.12
if [[ ! -e .venv && ! -L .venv ]]; then ln -s -- "$UV_PROJECT_ENVIRONMENT" .venv; fi
uv run --no-sync python - <<'PY'
import torch
from samsone_clm.resources import probe_storage
from samsone_clm.common import ROOT
print('Resources:', probe_storage(ROOT))
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Check the remote driver against the locked cu130 environment.')
device = torch.cuda.get_device_properties(0)
print('GPU:', device.name, 'VRAM GiB:', device.total_memory / 2**30)
PY
echo 'Environment ready. Transfer artifacts, then run prepare_transfer.py --verify and pytest.'
