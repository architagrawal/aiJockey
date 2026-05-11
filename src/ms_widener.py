"""Mid/Side stereo widener.

Decompose stereo into mid (L+R)/2 and side (L-R)/2, boost sides by
N dB, recompose. Opens up stereo image without phasey artifacts.

Toggle: AIJOCKEY_MS_WIDEN=1, AIJOCKEY_MS_WIDEN_DB=1.5
"""
from __future__ import annotations

import os

import numpy as np


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_MS_WIDEN", "0") == "1"


def widen(x: np.ndarray, db: float | None = None) -> np.ndarray:
    """Apply M/S widener on stereo `x` shape (2, n). Mono → no-op."""
    if x.ndim != 2 or x.shape[0] != 2:
        return x
    if db is None:
        try:
            db = float(os.environ.get("AIJOCKEY_MS_WIDEN_DB", "1.5"))
        except Exception:
            db = 1.5
    if db == 0.0:
        return x
    mid = ((x[0] + x[1]) * 0.5).astype(np.float32)
    side = ((x[0] - x[1]) * 0.5).astype(np.float32)
    gain = float(10.0 ** (db / 20.0))
    side = side * gain
    out = np.stack([mid + side, mid - side], axis=0).astype(np.float32)
    # Prevent clipping by gentle peak normalization
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out
