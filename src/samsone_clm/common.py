from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ORIGINAL_DATA_ROOT = Path("/srv/data/samsone_clm")
ROOT = Path(os.environ.get("SAMSONE_CLM_DATA", str(ORIGINAL_DATA_ROOT))).expanduser().absolute()


def resolve_data_path(path: str | Path) -> Path:
    """Resolve saved local paths on another host without rewriting hashed artifacts."""
    path = Path(path).expanduser()
    try:
        return ROOT / path.relative_to(ORIGINAL_DATA_ROOT)
    except ValueError:
        return path


def setup() -> None:
    for folder in ("raw", "processed", "features", "models", "runs", "cache", "tmp"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    for key, value in {
        "HF_HOME": ROOT / "cache/huggingface", "TORCH_HOME": ROOT / "cache/torch",
        "TMPDIR": ROOT / "tmp", "UV_CACHE_DIR": ROOT / "cache/uv",
    }.items():
        os.environ.setdefault(key, str(value))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for repo in ("samsone", "CLM"):
        sys.path.insert(0, str(PROJECT / "vendor" / repo / "src"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with resolve_data_path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def commit_file(temporary, destination):
    """Flush a complete artifact before publishing its final name on the data volume."""
    temporary=Path(temporary)
    with temporary.open('rb') as handle:os.fsync(handle.fileno())
    temporary.replace(destination)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open('w') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_rows(path: Path) -> list[dict]:
    path = resolve_data_path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def identity_split(identity: str) -> str:
    bucket = int(digest(["audio-split-v1", identity])[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


setup()
