"""Crowd-ambience overlay on low-energy / breakdown bars.

Looped festival-crowd noise mixed at -18 dB during low-energy segments
to add Tomorrowland atmosphere. No-op when no ambience source available.

Env:
    AIJOCKEY_CROWD_OVERLAY=1
    AIJOCKEY_CROWD_SAMPLE   path to mono/stereo WAV of crowd ambience
    AIJOCKEY_CROWD_DB       overlay level dB (default -18)
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

_LOCK = threading.Lock()
_SAMPLE_CACHE: tuple[np.ndarray, int] | None = None


def enabled() -> bool:
    return (os.environ.get("AIJOCKEY_CROWD_OVERLAY", "0") == "1"
             and os.environ.get("AIJOCKEY_CROWD_SAMPLE"))


def _load_sample(sr_target: int) -> tuple[np.ndarray, int] | None:
    global _SAMPLE_CACHE
    if _SAMPLE_CACHE is not None:
        wav, sr = _SAMPLE_CACHE
        if sr == sr_target:
            return _SAMPLE_CACHE
    with _LOCK:
        path = os.environ.get("AIJOCKEY_CROWD_SAMPLE")
        if not path or not Path(path).exists():
            return None
        try:
            import librosa
            wav, _ = librosa.load(path, sr=sr_target, mono=False)
            if wav.ndim == 1:
                wav = np.stack([wav, wav], axis=0)
            _SAMPLE_CACHE = (wav.astype(np.float32), sr_target)
            return _SAMPLE_CACHE
        except Exception as e:
            print(f"[crowd_overlay] load failed: {e}")
            return None


def overlay(x: np.ndarray, sr: int, *,
             energy_curve: np.ndarray | None = None,
             energy_thresh: float = 0.4,
             db: float | None = None) -> np.ndarray:
    """Mix crowd ambience under low-energy regions of x.

    Args:
        x: stereo (2, n).
        sr: sample rate.
        energy_curve: optional per-frame energy in [0, 1]; when below
            energy_thresh the crowd is mixed in. If None, mix at flat low level.
        energy_thresh: threshold for treating a frame as "breakdown".
        db: mix level (default from env AIJOCKEY_CROWD_DB or -18).

    Returns x with crowd mixed in (same shape).
    """
    if not enabled():
        return x
    cache = _load_sample(sr)
    if cache is None:
        return x
    crowd, _ = cache
    n = x.shape[1]
    # Loop crowd to match length
    if crowd.shape[1] < n:
        reps = int(np.ceil(n / crowd.shape[1]))
        crowd = np.tile(crowd, (1, reps))
    crowd = crowd[:, :n]
    if db is None:
        try:
            db = float(os.environ.get("AIJOCKEY_CROWD_DB", "-18"))
        except Exception:
            db = -18.0
    gain = float(10.0 ** (db / 20.0))
    if energy_curve is None:
        return (x + crowd * gain).astype(np.float32)
    # Build per-sample gain envelope from energy curve
    env = np.asarray(energy_curve, dtype=np.float32)
    # below threshold → full crowd; above → fade out
    mix = np.clip((energy_thresh - env) / max(energy_thresh, 1e-6), 0.0, 1.0)
    # Stretch mix to sample resolution
    samples_per_frame = max(1, n // max(1, len(mix)))
    full = np.repeat(mix, samples_per_frame)[:n]
    if len(full) < n:
        full = np.pad(full, (0, n - len(full)), mode="edge")
    return (x + crowd * gain * full[None, :]).astype(np.float32)
