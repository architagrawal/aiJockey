"""Synthesized riser + snare-roll generators.

No sample bank needed. Build classic EDM tension elements from
filtered white noise + parameterized envelopes.

Two primitives:
    white_noise_riser(duration, sr) — pitched/filtered white noise
        sweeping up over duration; classic "pre-drop" sound.
    snare_roll(bars, beat_dur, sr) — accelerating snare-roll (16 ->
        32 -> 64 hits per bar) on synthesized snare-like burst.

Used by build_riser_drop transition + accent_hint injection.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as _sig


def _hp(x: np.ndarray, sr: int, freq: float) -> np.ndarray:
    sos = _sig.butter(4, freq / (sr / 2.0), btype="high", output="sos")
    return _sig.sosfilt(sos, x).astype(np.float32)


def _bp(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    sos = _sig.butter(4, [lo / (sr / 2.0), hi / (sr / 2.0)],
                       btype="band", output="sos")
    return _sig.sosfilt(sos, x).astype(np.float32)


def white_noise_riser(duration_seconds: float, sr: int = 44100,
                        start_hz: float = 200.0,
                        end_hz: float = 12000.0,
                        peak_db: float = -6.0,
                        stereo: bool = True,
                        seed: int = 42) -> np.ndarray:
    """Filtered white-noise sweep: low-passes start narrow, opens up
    progressively. Linear amp ramp 0 → peak across duration.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_seconds * sr)
    noise = rng.standard_normal((2 if stereo else 1, n)).astype(np.float32)
    # Apply time-varying band by chunking; cheap proxy for swept filter
    n_chunks = max(8, int(duration_seconds * 8))
    chunk_len = max(1, n // n_chunks)
    out = np.zeros_like(noise)
    cutoffs = np.linspace(start_hz, end_hz, n_chunks)
    for i, cut in enumerate(cutoffs):
        s = i * chunk_len
        e = min(n, s + chunk_len)
        if e <= s:
            continue
        chunk = noise[:, s:e]
        for c in range(chunk.shape[0]):
            chunk[c] = _bp(chunk[c], sr, max(40.0, cut * 0.5),
                            min(sr * 0.45, cut))
        out[:, s:e] = chunk
    # Amp ramp + final peak gain
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32) ** 1.6
    out = out * ramp[None, :]
    gain = float(10.0 ** (peak_db / 20.0))
    peak = float(np.abs(out).max() + 1e-9)
    out = out * (gain / peak)
    return out.astype(np.float32)


def _snare_hit(sr: int, length_s: float = 0.12,
                 seed: int = 0) -> np.ndarray:
    """Synth a snare burst: white noise + decay + body resonance."""
    rng = np.random.default_rng(seed)
    n = int(length_s * sr)
    noise = rng.standard_normal(n).astype(np.float32)
    noise = _hp(noise, sr, 200.0)
    body = np.sin(2 * np.pi * 180.0 * np.arange(n) / sr) * 0.6
    env = np.exp(-np.linspace(0, 6.0, n)).astype(np.float32)
    hit = (noise * 0.7 + body * 0.4) * env
    stereo = np.stack([hit, hit], axis=0)
    return stereo.astype(np.float32)


def snare_roll(bars: int = 4, beat_dur: float = 0.5,
                 sr: int = 44100,
                 start_hits_per_bar: int = 4,
                 end_hits_per_bar: int = 32,
                 peak_db: float = -6.0,
                 seed: int = 7) -> np.ndarray:
    """Accelerating snare-roll across `bars` bars.

    Hits-per-bar ramps from start to end exponentially.
    """
    bar_dur = beat_dur * 4.0
    total = int(bars * bar_dur * sr)
    out = np.zeros((2, total), dtype=np.float32)
    rng = np.random.default_rng(seed)
    pos = 0.0
    # exponential ramp through hit counts
    rates = np.geomspace(start_hits_per_bar, end_hits_per_bar, bars).astype(int)
    for bi, hpb in enumerate(rates):
        step = bar_dur / max(1, hpb)
        bar_start_s = bi * bar_dur
        for k in range(hpb):
            t_s = bar_start_s + k * step
            n_pos = int(t_s * sr)
            hit = _snare_hit(sr, length_s=min(step * 0.9, 0.12),
                              seed=int(rng.integers(0, 1 << 31)))
            end = min(total, n_pos + hit.shape[1])
            out[:, n_pos:end] += hit[:, : end - n_pos]
    gain = float(10.0 ** (peak_db / 20.0))
    peak = float(np.abs(out).max() + 1e-9)
    out = out * (gain / peak)
    return out.astype(np.float32)
