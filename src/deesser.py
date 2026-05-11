"""Sibilance de-esser for vocal-collision junctions.

Detects 5-8 kHz transient energy spikes in vocal-heavy junctions and
applies a soft compressor on that band so harsh sibilance doesn't
bleed during transitions.

Toggle: AIJOCKEY_DEESSER=1
"""
from __future__ import annotations

import os

import numpy as np
from scipy import signal as _sig


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_DEESSER", "0") == "1"


def deess(x: np.ndarray, sr: int = 44100,
           band: tuple[float, float] = (5000.0, 8500.0),
           threshold_db: float = -20.0,
           ratio: float = 4.0,
           attack_ms: float = 2.0,
           release_ms: float = 30.0) -> np.ndarray:
    """Soft single-band compressor on the sibilance range.

    Args:
        x: stereo (2, n).
        sr: sample rate.
        band: low/high Hz of de-esser detection band.
        threshold_db: detector threshold.
        ratio: compression ratio.
        attack_ms, release_ms: envelope smoothing.

    Returns processed signal, same shape.
    """
    if not enabled() or x.ndim != 2:
        return x
    lo, hi = band
    nyq = sr / 2.0
    sos = _sig.butter(4, [max(1.0, lo) / nyq, min(0.99, hi / nyq)],
                       btype="band", output="sos")
    side = _sig.sosfiltfilt(sos, x, axis=-1).astype(np.float32)
    # Detector envelope
    rect = np.abs(side).mean(0)
    att = float(np.exp(-1.0 / (sr * attack_ms / 1000.0)))
    rel = float(np.exp(-1.0 / (sr * release_ms / 1000.0)))
    env = np.zeros_like(rect)
    g = 0.0
    for i, v in enumerate(rect):
        coeff = att if v > g else rel
        g = coeff * g + (1.0 - coeff) * float(v)
        env[i] = g
    # Convert to dB and compute gain reduction
    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))
    over = env_db - threshold_db
    gr_db = np.where(over > 0, over * (1.0 - 1.0 / ratio), 0.0)
    gr_lin = 10.0 ** (-gr_db / 20.0)
    side_comp = side * gr_lin[None, :]
    return (x - side + side_comp).astype(np.float32)
