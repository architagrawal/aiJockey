"""EDM 'smile' EQ curve mastering pass.

Festival/EDM mastering trick: boost lows (kick + sub) + boost highs
(presence + air) + scoop mids (vocal/lead clarity zone). Produces the
classic Tomorrowland mainstage sound — bass thump + sparkly tops
without midrange mud.

Applied AFTER the multi-band compressor in master.py, BEFORE LUFS norm.

Env:
    AIJOCKEY_SMILE_EQ=1
    AIJOCKEY_SMILE_LOW_DB   default +2.0  (boost 30-120 Hz)
    AIJOCKEY_SMILE_HIGH_DB  default +1.5  (boost 8-16 kHz)
    AIJOCKEY_SMILE_MID_DB   default -1.5  (scoop 800-2k Hz)
"""
from __future__ import annotations

import os

import numpy as np
from scipy import signal as _sig


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_SMILE_EQ", "0") == "1"


def _shelf(x: np.ndarray, sr: int, freq_hz: float, gain_db: float,
            kind: str = "low") -> np.ndarray:
    """Apply a shelving filter (low or high) via bilinear-transformed
    second-order IIR (RBJ cookbook)."""
    a0 = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    S = 1.0
    alpha = sin_w0 / 2.0 * np.sqrt((a0 + 1.0 / a0) * (1.0 / S - 1) + 2.0)
    two_sqrtA_alpha = 2.0 * np.sqrt(a0) * alpha
    if kind == "low":
        b0 = a0 * ((a0 + 1) - (a0 - 1) * cos_w0 + two_sqrtA_alpha)
        b1 = 2 * a0 * ((a0 - 1) - (a0 + 1) * cos_w0)
        b2 = a0 * ((a0 + 1) - (a0 - 1) * cos_w0 - two_sqrtA_alpha)
        a0_c = (a0 + 1) + (a0 - 1) * cos_w0 + two_sqrtA_alpha
        a1 = -2 * ((a0 - 1) + (a0 + 1) * cos_w0)
        a2 = (a0 + 1) + (a0 - 1) * cos_w0 - two_sqrtA_alpha
    else:  # high
        b0 = a0 * ((a0 + 1) + (a0 - 1) * cos_w0 + two_sqrtA_alpha)
        b1 = -2 * a0 * ((a0 - 1) + (a0 + 1) * cos_w0)
        b2 = a0 * ((a0 + 1) + (a0 - 1) * cos_w0 - two_sqrtA_alpha)
        a0_c = (a0 + 1) - (a0 - 1) * cos_w0 + two_sqrtA_alpha
        a1 = 2 * ((a0 - 1) - (a0 + 1) * cos_w0)
        a2 = (a0 + 1) - (a0 - 1) * cos_w0 - two_sqrtA_alpha
    b = [b0 / a0_c, b1 / a0_c, b2 / a0_c]
    a = [1.0, a1 / a0_c, a2 / a0_c]
    return _sig.lfilter(b, a, x, axis=-1).astype(np.float32)


def _peaking(x: np.ndarray, sr: int, freq_hz: float, gain_db: float,
              q: float = 1.0) -> np.ndarray:
    """Peaking EQ (boost/cut at a single freq) via RBJ cookbook."""
    a0 = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b0 = 1 + alpha * a0
    b1 = -2 * cos_w0
    b2 = 1 - alpha * a0
    a0_c = 1 + alpha / a0
    a1 = -2 * cos_w0
    a2 = 1 - alpha / a0
    b = [b0 / a0_c, b1 / a0_c, b2 / a0_c]
    a = [1.0, a1 / a0_c, a2 / a0_c]
    return _sig.lfilter(b, a, x, axis=-1).astype(np.float32)


def apply(x: np.ndarray, sr: int = 44100) -> np.ndarray:
    """Apply smile EQ in-place style. Returns same shape."""
    if not enabled() or x.ndim != 2:
        return x
    try:
        low_db = float(os.environ.get("AIJOCKEY_SMILE_LOW_DB", "2.0"))
        high_db = float(os.environ.get("AIJOCKEY_SMILE_HIGH_DB", "1.5"))
        mid_db = float(os.environ.get("AIJOCKEY_SMILE_MID_DB", "-1.5"))
    except Exception:
        low_db, high_db, mid_db = 2.0, 1.5, -1.5
    y = x.astype(np.float32)
    y = _shelf(y, sr, 80.0, low_db, kind="low")
    y = _shelf(y, sr, 10000.0, high_db, kind="high")
    y = _peaking(y, sr, 1200.0, mid_db, q=0.9)
    # Prevent peak clip from boost
    peak = float(np.abs(y).max())
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)
