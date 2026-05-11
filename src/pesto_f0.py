"""PESTO F0 estimator — 50× faster CREPE replacement.

Sony CSL Paris, ISMIR'23. Self-supervised pitch estimator, RPA 95%+
on MIR-1K/MDB/PTDB, MIT.

Env:
    AIJOCKEY_PESTO_ENABLE   0|1   default 0
    AIJOCKEY_PESTO_DEVICE   str   default 'cuda' if available
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_PIPE = None
_LOAD_FAILED = False


def enabled() -> bool:
    if os.environ.get("AIJOCKEY_PESTO_ENABLE", "0") != "1":
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
            import pesto  # type: ignore
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            _PIPE = {"pesto": pesto, "device": device}
            print(f"[pesto] loaded on {device}")
            return _PIPE
        except Exception as e:
            print(f"[pesto] load failed ({e.__class__.__name__}: {e})")
            _LOAD_FAILED = True
            return None


def predict_f0(audio_path: str | Path,
                 step_size_ms: float = 10.0,
                 device: str = "cuda") -> dict | None:
    """Returns {"timestamps_s", "f0_hz", "confidence"} or None on failure."""
    if not enabled():
        return None
    pipe = _load(device=device)
    if pipe is None:
        return None
    try:
        import torch
        import torchaudio
        wav, sr = torchaudio.load(str(audio_path))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if pipe["device"] == "cuda":
            wav = wav.cuda()
        ts, f0, conf, _ = pipe["pesto"].predict(
            wav, sr, step_size=step_size_ms,
        )
        return {
            "timestamps_s": ts.detach().cpu().numpy().tolist(),
            "f0_hz": f0.detach().cpu().numpy().tolist(),
            "confidence": conf.detach().cpu().numpy().tolist(),
        }
    except Exception as e:
        print(f"[pesto] predict failed for {audio_path}: {e}")
        return None
