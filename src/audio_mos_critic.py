"""Audio MOS critic — AESCA fallback (AudioMOS DORA-MOS not released).

Research note (May 2026): DORA-MOS (AudioMOS Challenge 2025 winner) has
NOT released code or weights. Treat any hardcoded `AudioMOS/DORA-MOS`
ID as fictional.

This wrapper points at AESCA (Track-2 winner of AudioMOS Challenge,
github.com/CyberAgentAILab/aesca) instead. AESCA predicts the 4-axis
Audiobox-aesthetics scores, so it's effectively a re-implementation of
our existing critic — kept here only as a *second-opinion* head.

If AESCA is unavailable in the environment, this module degrades to
returning None so the rest of the pipeline ignores it.

Env knobs:
    AIJOCKEY_AUDIO_MOS_ENABLE   0|1  default 0
    AIJOCKEY_AESCA_REPO         path to cloned AESCA repo
    AIJOCKEY_AESCA_CKPT         AESCA checkpoint .pt
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

_LOCK = threading.Lock()
_PIPE = None
_LOAD_FAILED = False


def enabled() -> bool:
    if os.environ.get("AIJOCKEY_AUDIO_MOS_ENABLE", "0") != "1":
        return False
    return not _LOAD_FAILED


def _load(device: str | None = None):
    global _PIPE, _LOAD_FAILED
    if _PIPE is not None:
        return _PIPE
    if _LOAD_FAILED:
        return None
    with _LOCK:
        if _PIPE is not None:
            return _PIPE
        try:
            import torch
            repo = os.environ.get("AIJOCKEY_AESCA_REPO")
            ckpt = os.environ.get("AIJOCKEY_AESCA_CKPT")
            if not repo or not Path(repo).exists():
                print(f"[audio_mos/aesca] no repo at AIJOCKEY_AESCA_REPO={repo}")
                _LOAD_FAILED = True
                return None
            if not ckpt or not Path(ckpt).exists():
                print(f"[audio_mos/aesca] no ckpt at AIJOCKEY_AESCA_CKPT={ckpt}")
                _LOAD_FAILED = True
                return None
            if repo not in sys.path:
                sys.path.insert(0, repo)
            from aesca.inference import load_model as _load_aesca  # type: ignore
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model = _load_aesca(ckpt, device=device)
            _PIPE = {"model": model, "device": device}
            print(f"[audio_mos/aesca] loaded {ckpt} on {device}")
            return _PIPE
        except Exception as e:
            print(f"[audio_mos/aesca] load failed: {e}")
            _LOAD_FAILED = True
            return None


def score(audio_path: str | Path, device: str = "cuda") -> float | None:
    """Returns a MOS-like scalar (~1-5) or None. Aggregates AESCA's
    4-axis output to a single scalar via (PQ+CE)/2 for parity."""
    if not enabled():
        return None
    pipe = _load(device=device)
    if pipe is None:
        return None
    try:
        out = pipe["model"].score(str(audio_path))
        if not out:
            return None
        pq = float(out.get("PQ", 0.0))
        ce = float(out.get("CE", 0.0))
        return (pq + ce) / 2.0
    except Exception as e:
        print(f"[audio_mos/aesca] score failed for {audio_path}: {e}")
        return None
