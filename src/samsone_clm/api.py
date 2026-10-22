import os
import threading
from .common import ROOT

_ranker = None
_lock = threading.RLock()


def configure(checkpoint=None, device="cuda"):
    global _ranker
    from .infer import Ranker
    with _lock:
        current=ROOT/'runs/v3/demo.pt'
        path = checkpoint or os.environ.get("SAMSONE_CLM_CHECKPOINT") or (current if current.exists() else ROOT/"runs/latest.pt")
        _ranker = Ranker(path, device=device)
        return _ranker


def rank(audio, question, choices):
    with _lock:
        if _ranker is None:
            configure()
        return _ranker.rank(audio, question, choices)
