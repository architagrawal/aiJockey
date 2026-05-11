"""Reverse-reverb impact (DROP-tier glue, no sample needed).

Take a brief tail of outgoing audio, render reverb-decayed version,
time-reverse it. Result swells INTO the downbeat — pre-drop suction
effect.

Toggle in transitions.py for build_riser_drop / drop tier.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as _sig


def _short_reverb(x: np.ndarray, sr: int,
                    decay_s: float = 1.8,
                    pre_delay_s: float = 0.0) -> np.ndarray:
    """Schroeder-style reverb tail: 4 comb filters + 2 allpass.
    Returns x with tail appended (length grows)."""
    n = x.shape[-1]
    decay_samples = int(decay_s * sr)
    pad = np.zeros((x.shape[0], decay_samples), dtype=np.float32) \
        if x.ndim == 2 else np.zeros(decay_samples, dtype=np.float32)
    y = np.concatenate([x.astype(np.float32), pad], axis=-1)
    comb_delays = [1116, 1188, 1277, 1356]   # ms-like sample counts
    g_arr = [0.84, 0.83, 0.82, 0.81]
    out = np.zeros_like(y)
    for d, g in zip(comb_delays, g_arr):
        buf = np.zeros_like(y)
        d_samp = d
        for i in range(d_samp, y.shape[-1]):
            if y.ndim == 2:
                buf[:, i] = y[:, i] + g * buf[:, i - d_samp]
            else:
                buf[i] = y[i] + g * buf[i - d_samp]
        out = out + buf * 0.25
    return out.astype(np.float32)


def reverse_reverb_pre_impact(source_tail: np.ndarray, sr: int = 44100,
                                 decay_s: float = 1.8,
                                 peak_db: float = -6.0) -> np.ndarray:
    """Generate reversed-reverb swell from source_tail.

    Returns waveform (2, n_reverb_samples) ending at amplitude peak —
    caller splices BEFORE the downbeat of incoming.
    """
    rev = _short_reverb(source_tail, sr, decay_s=decay_s)
    # Take just the reverb-tail portion (after dry signal)
    n_dry = source_tail.shape[-1]
    tail = rev[:, n_dry:] if rev.ndim == 2 else rev[n_dry:]
    if tail.shape[-1] < 1:
        return np.zeros((2, int(decay_s * sr)), dtype=np.float32)
    # Time-reverse
    reversed_tail = tail[:, ::-1] if tail.ndim == 2 else tail[::-1]
    reversed_tail = np.ascontiguousarray(reversed_tail).astype(np.float32)
    # Peak normalize
    gain = float(10.0 ** (peak_db / 20.0))
    peak = float(np.abs(reversed_tail).max() + 1e-9)
    return (reversed_tail * (gain / peak)).astype(np.float32)
