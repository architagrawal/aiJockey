"""Sidechain ducking at junctions.

When two clips overlap at a transition, the incoming clip's kick band
(20-200 Hz) is detected as an envelope follower. That envelope is
used to duck the outgoing clip's low-mid (50-500 Hz) so the two basses
don't fight. Classic EDM "pump" feel.

Toggle: AIJOCKEY_SIDECHAIN_DUCK=1
"""
from __future__ import annotations

import numpy as np
from scipy import signal as _sig


def _band(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    nyq = sr / 2.0
    sos = _sig.butter(4, [max(1.0, lo) / nyq, min(0.99, hi / nyq)],
                       btype="band", output="sos")
    return _sig.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def _envelope(x: np.ndarray, sr: int, attack_ms: float = 5.0,
               release_ms: float = 80.0) -> np.ndarray:
    """Rectified one-pole envelope follower across last axis."""
    rect = np.abs(x).mean(0) if x.ndim == 2 else np.abs(x)
    att = float(np.exp(-1.0 / (sr * attack_ms / 1000.0)))
    rel = float(np.exp(-1.0 / (sr * release_ms / 1000.0)))
    env = np.zeros_like(rect, dtype=np.float32)
    g = 0.0
    for i, v in enumerate(rect):
        coeff = att if v > g else rel
        g = coeff * g + (1.0 - coeff) * float(v)
        env[i] = g
    return env


def sidechain_overlap(out_tail: np.ndarray, in_head: np.ndarray,
                       sr: int = 44100,
                       depth_db: float = -6.0,
                       attack_ms: float = 5.0,
                       release_ms: float = 80.0,
                       trigger_band: tuple[float, float] = (40.0, 200.0),
                       duck_band: tuple[float, float] = (50.0, 500.0),
                       stem_aware: bool | None = None
                       ) -> np.ndarray:
    """Apply sidechain duck of out_tail driven by in_head's trigger band.

    Returns ducked out_tail (same shape, same length as in_head).
    """
    import os as _os
    if stem_aware is None:
        stem_aware = _os.environ.get("AIJOCKEY_SIDECHAIN_STEM_AWARE", "0") == "1"
    n = min(out_tail.shape[1], in_head.shape[1])
    a = out_tail[:, :n].astype(np.float32)
    b = in_head[:, :n].astype(np.float32)
    trig_lo, trig_hi = trigger_band
    # Stem-aware mode narrows duck band to sub/kick only — preserves
    # mids and highs, more selective pump.
    if stem_aware:
        duck_band = (40.0, 180.0)
    duck_lo, duck_hi = duck_band
    # Envelope of B's kick band
    trig_band = _band(b, sr, trig_lo, trig_hi)
    env = _envelope(trig_band, sr, attack_ms, release_ms)
    if env.max() < 1e-6:
        return a
    env_norm = env / max(env.max(), 1e-6)
    # Gain reduction curve in linear: 1.0 down to 10^(depth/20)
    floor = float(10.0 ** (depth_db / 20.0))
    gr = 1.0 - (1.0 - floor) * env_norm
    # Apply only to A's duck band; pass-through the rest
    a_duck = _band(a, sr, duck_lo, duck_hi)
    a_rest = a - a_duck
    a_ducked = a_duck * gr[None, :]
    return (a_rest + a_ducked).astype(np.float32)
