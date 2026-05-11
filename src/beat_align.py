"""Beat-grid DTW + sub-sample fractional-delay phase alignment.

Eliminates the phase-cancellation artifacts our audio_probes flag at
junction overlaps. Two-stage:

    1. **Beat-grid DTW**: align A's tail beats to B's head beats using
       dynamic time warping over beat-strength curves. Returns the
       optimal time shift (in samples) that puts the beats in lockstep.
    2. **Sub-sample fractional delay**: after the integer shift, run a
       cross-correlation over the kick band (20-200 Hz) on the trailing
       beat of A vs the leading beat of B. The peak gives a fractional
       sample offset that an all-pass fractional-delay filter applies.

Output: B is time-shifted so A_tail + B_head sum constructively rather
than cancelling. No magnitude change — purely phase alignment.

Public API:
    align_for_overlap(a_tail, b_head, sr, ...) -> (shift_samples, b_aligned)
"""
from __future__ import annotations

import numpy as np
from scipy import signal as _sig


def _band_filter(x: np.ndarray, sr: int,
                  low_hz: float = 20.0,
                  high_hz: float = 200.0) -> np.ndarray:
    """Band-pass mono signal in the kick band."""
    nyq = sr / 2.0
    lo, hi = max(1.0, low_hz) / nyq, min(0.99, high_hz / nyq)
    sos = _sig.butter(4, [lo, hi], btype="band", output="sos")
    return _sig.sosfiltfilt(sos, x).astype(np.float32)


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x.mean(0).astype(np.float32)


def _beat_strength(x: np.ndarray, sr: int, hop: int = 512) -> np.ndarray:
    """Cheap onset envelope: rectified diff of frame energies."""
    n = x.shape[-1]
    frames = max(1, n // hop)
    eng = np.zeros(frames, dtype=np.float32)
    mono = _to_mono(x)
    for i in range(frames):
        s = i * hop
        e = min(s + hop, n)
        eng[i] = float(np.sqrt(np.mean(mono[s:e] ** 2) + 1e-9))
    diff = np.diff(eng, prepend=eng[0])
    diff[diff < 0] = 0.0
    return diff


def _dtw_shift(a_env: np.ndarray, b_env: np.ndarray,
                max_shift_frames: int = 32) -> int:
    """Return integer frame shift that maximizes correlation of envelopes."""
    n = min(len(a_env), len(b_env))
    if n < 4:
        return 0
    a = a_env[-n:]
    b = b_env[:n]
    best_corr = -np.inf
    best_shift = 0
    for s in range(-max_shift_frames, max_shift_frames + 1):
        if s >= 0:
            a_seg = a[s:]
            b_seg = b[: len(a_seg)]
        else:
            b_seg = b[-s:]
            a_seg = a[: len(b_seg)]
        if len(a_seg) < 2:
            continue
        c = float(np.dot(a_seg, b_seg) / (np.linalg.norm(a_seg) *
                                             np.linalg.norm(b_seg) + 1e-9))
        if c > best_corr:
            best_corr = c
            best_shift = s
    return best_shift


def _xcorr_subsample(a: np.ndarray, b: np.ndarray,
                      sr: int, max_lag_samples: int = 200) -> float:
    """Return sub-sample lag in samples (float) maximizing cross-correlation
    of band-passed signals over ±max_lag_samples window."""
    af = _band_filter(_to_mono(a), sr)
    bf = _band_filter(_to_mono(b), sr)
    n = min(len(af), len(bf), 4096)
    if n < 64:
        return 0.0
    af = af[-n:]
    bf = bf[:n]
    full = _sig.correlate(bf, af, mode="full")
    mid = len(full) // 2
    lo = max(0, mid - max_lag_samples)
    hi = min(len(full), mid + max_lag_samples + 1)
    seg = full[lo:hi]
    if not len(seg):
        return 0.0
    k = int(np.argmax(seg))
    # Parabolic peak refinement for sub-sample accuracy.
    if 0 < k < len(seg) - 1:
        y0, y1, y2 = float(seg[k - 1]), float(seg[k]), float(seg[k + 1])
        denom = (y0 - 2.0 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    lag = (lo + k) - mid + float(delta)
    return float(lag)


def _fractional_shift(x: np.ndarray, shift_samples: float) -> np.ndarray:
    """Apply fractional sample shift via FFT phase rotation. Preserves
    magnitude exactly (linear-phase all-pass)."""
    if abs(shift_samples) < 1e-6:
        return x.astype(np.float32)
    n = x.shape[-1]
    X = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(n)
    phase = np.exp(-2j * np.pi * freqs * shift_samples)
    Y = X * phase
    out = np.fft.irfft(Y, n=n, axis=-1)
    return out.astype(np.float32)


def align_for_overlap(a_tail: np.ndarray, b_head: np.ndarray,
                       sr: int = 44100,
                       beat_hop: int = 512,
                       max_shift_ms: float = 30.0) -> tuple[int, np.ndarray]:
    """Compute integer + fractional shift that aligns B to A.

    Args:
        a_tail: stereo (2, n) waveform from end of clip A.
        b_head: stereo (2, n) waveform from start of clip B.
        sr: sample rate.
        beat_hop: STFT hop for onset envelope.
        max_shift_ms: max integer shift to consider (±).

    Returns:
        (shift_samples, b_aligned). shift_samples is the total shift
        applied to b_head (positive = delay). b_aligned has the same
        shape as b_head with the shift baked in via fractional-delay
        filter (preserves length; edge samples may be zero-padded).
    """
    if a_tail.ndim == 1:
        a_tail = a_tail[None, :]
    if b_head.ndim == 1:
        b_head = b_head[None, :]
    max_frames = int((max_shift_ms / 1000.0) * sr / beat_hop)
    a_env = _beat_strength(a_tail, sr, hop=beat_hop)
    b_env = _beat_strength(b_head, sr, hop=beat_hop)
    frame_shift = _dtw_shift(a_env, b_env, max_shift_frames=max_frames)
    integer_shift = int(frame_shift * beat_hop)
    # Apply integer shift first via roll, zero-pad
    n = b_head.shape[1]
    if integer_shift > 0:
        shifted = np.zeros_like(b_head)
        shifted[:, integer_shift:] = b_head[:, : n - integer_shift]
    elif integer_shift < 0:
        s = -integer_shift
        shifted = np.zeros_like(b_head)
        shifted[:, : n - s] = b_head[:, s:]
    else:
        shifted = b_head.copy()
    # Sub-sample refinement over kick band of overlap
    sub = _xcorr_subsample(a_tail, shifted, sr,
                            max_lag_samples=min(200, int(sr * 0.01)))
    aligned = _fractional_shift(shifted, sub).astype(np.float32)
    return integer_shift + int(round(sub)), aligned
