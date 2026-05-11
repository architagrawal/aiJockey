"""Bass-mono collapse below crossover.

Standard DJ/club practice: sub-bass below ~120 Hz mono-summed so
phase issues don't cancel on big rigs. Applied as a mastering pass
BEFORE final stereo widener.

Toggle: AIJOCKEY_BASS_MONO=1, AIJOCKEY_BASS_MONO_HZ=120
"""
from __future__ import annotations

import os

import numpy as np
from scipy import signal as _sig


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_BASS_MONO", "0") == "1"


def collapse(x: np.ndarray, sr: int = 44100,
              crossover_hz: float | None = None) -> np.ndarray:
    """Mono-sum content below crossover_hz, keep highs stereo."""
    if not enabled() or x.ndim != 2 or x.shape[0] != 2:
        return x
    if crossover_hz is None:
        try:
            crossover_hz = float(os.environ.get("AIJOCKEY_BASS_MONO_HZ", "120"))
        except Exception:
            crossover_hz = 120.0
    nyq = sr / 2.0
    low_sos = _sig.butter(4, crossover_hz / nyq, btype="low", output="sos")
    high_sos = _sig.butter(4, crossover_hz / nyq, btype="high", output="sos")
    low = _sig.sosfiltfilt(low_sos, x, axis=-1).astype(np.float32)
    high = _sig.sosfiltfilt(high_sos, x, axis=-1).astype(np.float32)
    # Mono-sum the lows
    mono_low = ((low[0] + low[1]) * 0.5).astype(np.float32)
    low_mono = np.stack([mono_low, mono_low], axis=0)
    out = (low_mono + high).astype(np.float32)
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out
