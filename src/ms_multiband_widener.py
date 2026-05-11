"""Multiband Mid/Side widener.

Splits stereo into 3 bands (sub / mid / high), applies per-band
side gain: sub COLLAPSED to mono (gain -inf side), mids near-flat,
highs WIDE. Club-safe; preserves kick punch + bass mono compatibility,
opens up cymbal/synth top end.

Toggle: AIJOCKEY_MS_MULTIBAND=1 (overrides single-band ms_widener)
Params (env):
    AIJOCKEY_MS_SUB_CROSSOVER     default 120 Hz
    AIJOCKEY_MS_HIGH_CROSSOVER    default 6000 Hz
    AIJOCKEY_MS_MID_DB            default 0
    AIJOCKEY_MS_HIGH_DB           default 2.5
"""
from __future__ import annotations

import os

import numpy as np
from scipy import signal as _sig


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_MS_MULTIBAND", "0") == "1"


def _split_bands(x: np.ndarray, sr: int,
                  sub_hz: float, high_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linkwitz-Riley 4th-order crossover into 3 bands."""
    nyq = sr / 2.0
    low_sos = _sig.butter(4, sub_hz / nyq, btype="low", output="sos")
    band_sos = _sig.butter(4, [sub_hz / nyq, high_hz / nyq],
                             btype="band", output="sos")
    high_sos = _sig.butter(4, high_hz / nyq, btype="high", output="sos")
    sub = _sig.sosfiltfilt(low_sos, x, axis=-1).astype(np.float32)
    mid = _sig.sosfiltfilt(band_sos, x, axis=-1).astype(np.float32)
    high = _sig.sosfiltfilt(high_sos, x, axis=-1).astype(np.float32)
    return sub, mid, high


def _apply_side_gain(band: np.ndarray, side_db: float | None = None,
                      collapse_to_mono: bool = False) -> np.ndarray:
    if band.ndim != 2 or band.shape[0] != 2:
        return band
    m = ((band[0] + band[1]) * 0.5).astype(np.float32)
    s = ((band[0] - band[1]) * 0.5).astype(np.float32)
    if collapse_to_mono:
        s = np.zeros_like(s)
    elif side_db is not None and side_db != 0.0:
        s = s * float(10.0 ** (side_db / 20.0))
    return np.stack([m + s, m - s], axis=0).astype(np.float32)


def widen(x: np.ndarray, sr: int = 44100) -> np.ndarray:
    """Apply multiband M/S widener. Mono → no-op."""
    if not enabled() or x.ndim != 2 or x.shape[0] != 2:
        return x
    try:
        sub_hz = float(os.environ.get("AIJOCKEY_MS_SUB_CROSSOVER", "120"))
        hi_hz = float(os.environ.get("AIJOCKEY_MS_HIGH_CROSSOVER", "6000"))
        mid_db = float(os.environ.get("AIJOCKEY_MS_MID_DB", "0"))
        hi_db = float(os.environ.get("AIJOCKEY_MS_HIGH_DB", "2.5"))
    except Exception:
        sub_hz, hi_hz, mid_db, hi_db = 120.0, 6000.0, 0.0, 2.5
    sub, mid, high = _split_bands(x, sr, sub_hz, hi_hz)
    sub = _apply_side_gain(sub, collapse_to_mono=True)
    mid = _apply_side_gain(mid, side_db=mid_db)
    high = _apply_side_gain(high, side_db=hi_db)
    out = (sub + mid + high).astype(np.float32)
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out
