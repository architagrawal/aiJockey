"""Cross-clip CLAP coherence scoring.

Penalizes Director plans whose adjacent clip pairs have low CLAP
cosine. Encourages narrative continuity within a set.

Toggle: AIJOCKEY_CLAP_COHERENCE=1
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_CLAP_COHERENCE", "0") == "1"


def _load_clap(cache_dir: str | Path, clip_id: str) -> np.ndarray | None:
    npz = Path(cache_dir) / f"{clip_id}.npz"
    if not npz.exists():
        lib = os.environ.get("AIJOCKEY_LIBRARY_CACHE") or "/cache"
        npz = Path(lib) / f"{clip_id}.npz"
    if not npz.exists():
        return None
    try:
        data = np.load(npz)
        clap = data.get("clap")
        if clap is None:
            return None
        return np.asarray(clap, dtype=np.float32)
    except Exception:
        return None


def coherence_score(clip_ids: list[str], cache_dir: str | Path) -> float:
    """Return mean adjacent CLAP-cosine across the sequence in [-1, 1]."""
    if not enabled() or len(clip_ids) < 2:
        return 0.0
    embs = [_load_clap(cache_dir, c) for c in clip_ids]
    sims = []
    prev = None
    for e in embs:
        if e is None or e.size == 0:
            prev = e
            continue
        if prev is not None and prev.size > 0:
            n1 = float(np.linalg.norm(prev)) + 1e-9
            n2 = float(np.linalg.norm(e)) + 1e-9
            sims.append(float((prev / n1) @ (e / n2)))
        prev = e
    if not sims:
        return 0.0
    return float(sum(sims) / len(sims))
