"""Sub-sample beat alignment for transition overlaps.

Targets the dominant artifact in the cohort baseline: spectral phasing
(93.3% of severity, mean 0.695). Cause when isolated:
    - Sub-sample beat jitter from tracker imprecision + rubberband stretch
    - Misaligned by 5-30 ms even when nominally beat-matched
    - Sums of misaligned periodic signals → comb-filter / phase cancel

This module fixes ONE of the three phase-cancellation sub-causes:
    1. SAMPLE JITTER (5-30 ms)        ← THIS MODULE
    2. Bar-grid mismatch              ← needs All-In-One section labels
    3. Harmonic content overlap       ← needs spectral overlap detection

Method (NOT full DTW — a fixed-point xcorr alignment is the right tool
when both windows are short and locally-stationary; full DTW is overkill):

    1. Take overlap-region prev_tail + cur_head (∼1-2 s @ 44.1 kHz).
    2. Compute RMS envelope of each (mono, smoothed).
    3. Cross-correlate envelopes.
    4. Find lag of peak within ±max_shift_ms.
    5. Reject if peak confidence < min_confidence (no real correlation).
    6. Apply integer-sample shift to cur_head; pad zeros at the freed edge.

Apply shift via numpy index slice — no rubberband, no SR conversion.
Pure O(N) array ops; ~5 ms per junction at SR=44.1k.

Env:
    AIJOCKEY_DTW_ALIGN          0|1     default 0 (opt-in until validated)
    AIJOCKEY_DTW_MAX_SHIFT_MS   float   default 50.0
    AIJOCKEY_DTW_MIN_CONFIDENCE float   default 0.3
    AIJOCKEY_DTW_ENVELOPE_MS    float   default 20.0  (RMS smoothing window)

Wire-in (single call site, src/execute.py:apply_transition):

    from dtw_align import align_overlap, enabled
    if enabled():
        cur_head_aligned, shift, conf = align_overlap(prev_tail, cur_head, sr=SR)
        if shift != 0:
            cur_head = cur_head_aligned   # adopt the shifted version
"""
from __future__ import annotations

import os
from typing import Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, "Tensor"]   # noqa: F821 — runtime tolerant


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def enabled() -> bool:
    """True if env opts in. Default OFF — must be validated per pool first."""
    return os.environ.get("AIJOCKEY_DTW_ALIGN", "0") == "1"


def align_overlap(prev_tail: np.ndarray,
                  cur_head: np.ndarray,
                  sr: int = 44100,
                  max_shift_ms: float | None = None,
                  min_confidence: float | None = None,
                  envelope_ms: float | None = None,
                  ) -> Tuple[np.ndarray, int, float]:
    """Align cur_head to prev_tail at sub-sample beat boundary.

    Inputs are (channels, T) numpy arrays OR (T,) mono. Stereo handled
    by mixing to mono for the envelope; the shift is then applied to
    the original (potentially stereo) cur_head.

    Returns:
        (shifted_cur_head, shift_samples, confidence)
        - shifted_cur_head : same shape as cur_head, content shifted by lag.
                              Original returned unchanged when no-op.
        - shift_samples    : int. positive = advanced (drop early samples).
                              negative = delayed (prepend zeros). 0 = no-op.
        - confidence       : float in [0, 1]. 0 = no correlation found.

    No-op cases (returns cur_head unchanged with shift=0):
        - Either input shorter than 250 ms.
        - Peak xcorr confidence < min_confidence.
        - Detected lag at the clamp boundary (peak likely wrong).

    Pure numpy. ~5 ms / junction at SR=44.1k for 2 s overlap.
    """
    if max_shift_ms is None:
        max_shift_ms = float(os.environ.get("AIJOCKEY_DTW_MAX_SHIFT_MS", "50"))
    if min_confidence is None:
        min_confidence = float(os.environ.get("AIJOCKEY_DTW_MIN_CONFIDENCE", "0.3"))
    if envelope_ms is None:
        envelope_ms = float(os.environ.get("AIJOCKEY_DTW_ENVELOPE_MS", "20"))

    prev_tail = np.asarray(prev_tail)
    cur_head = np.asarray(cur_head)

    max_lag = int(sr * max_shift_ms / 1000.0)

    # Envelopes — mono, smoothed RMS magnitude
    a = _envelope_mono(prev_tail, sr, envelope_ms)
    b = _envelope_mono(cur_head, sr, envelope_ms)

    n = int(min(len(a), len(b)))
    if n < sr // 4:    # < 250 ms — too short for reliable xcorr
        return cur_head, 0, 0.0
    a = a[:n]
    b = b[:n]

    lag, confidence = _xcorr_peak_lag(a, b, max_lag)

    # Reject low-confidence shifts (random correlation rather than real beat).
    if confidence < min_confidence:
        return cur_head, 0, confidence

    # Reject clamp-boundary peaks — usually means xcorr trailed off the
    # window, not a real alignment.
    if abs(lag) >= max_lag:
        return cur_head, 0, confidence

    if lag == 0:
        return cur_head, 0, confidence

    shifted = _apply_shift(cur_head, lag)
    return shifted, int(lag), float(confidence)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _envelope_mono(wav: np.ndarray, sr: int, win_ms: float) -> np.ndarray:
    """Smoothed RMS-style envelope. (C, T) or (T,) → (T,)."""
    if wav.ndim == 2:
        # Mix down to mono (same as librosa.to_mono)
        if wav.shape[0] in (1, 2):
            mono = wav.mean(axis=0)
        else:
            # Treat as (T, C) if shape doesn't look like channels-first
            mono = wav.mean(axis=-1) if wav.shape[-1] <= 8 else wav.mean(axis=0)
    elif wav.ndim == 1:
        mono = wav
    else:
        mono = wav.reshape(-1)

    mono = mono.astype(np.float32, copy=False)
    abs_w = np.abs(mono)

    win = max(1, int(sr * win_ms / 1000.0))
    if win <= 1 or win >= len(abs_w):
        return abs_w
    # Box-filter smoothing — fast O(N) via cumulative sum trick.
    csum = np.cumsum(np.insert(abs_w, 0, 0.0))
    smoothed = (csum[win:] - csum[:-win]) / win
    # Pad to original length so caller sees consistent indexing
    pad = len(abs_w) - len(smoothed)
    if pad > 0:
        smoothed = np.pad(smoothed, (0, pad), mode="edge")
    return smoothed.astype(np.float32, copy=False)


def _xcorr_peak_lag(a: np.ndarray, b: np.ndarray, max_lag: int
                    ) -> Tuple[int, float]:
    """Cross-correlate two same-length envelopes; return peak lag + confidence.

    lag > 0 → a leads b (advance b by lag samples to align).
    lag < 0 → b leads a (delay b by |lag| samples).
    confidence = peak_value / sqrt(energy(a) * energy(b)) ∈ [0, 1]-ish.

    Peak search restricted to [-max_lag, +max_lag] to avoid catching
    random correlations far away from the joint.
    """
    n = len(a)
    if n == 0:
        return 0, 0.0

    a0 = a - a.mean()
    b0 = b - b.mean()
    # Full xcorr: length 2n-1, zero lag at index n-1
    xc = np.correlate(a0, b0, mode="full")
    zero = n - 1
    lo = max(0, zero - max_lag)
    hi = min(len(xc), zero + max_lag + 1)
    if hi <= lo:
        return 0, 0.0

    window = xc[lo:hi]
    peak_idx_local = int(np.argmax(window))
    peak_idx = lo + peak_idx_local
    lag = peak_idx - zero
    peak_val = float(xc[peak_idx])

    # Normalized cross-correlation magnitude — gives a real [0, 1] number
    # when both signals have non-trivial energy.
    norm = float(np.sqrt(float((a0 ** 2).sum()) * float((b0 ** 2).sum())))
    if norm <= 1e-9:
        return 0, 0.0
    confidence = max(0.0, peak_val / norm)
    # Clip to roughly [0, 1] (xcorr peak can technically exceed norm
    # numerically due to mean removal; treat anything >1 as 1).
    confidence = min(1.0, confidence)
    return lag, confidence


def _apply_shift(wav: np.ndarray, shift: int) -> np.ndarray:
    """Index-shift cur_head by `shift` samples, pad zeros at the freed edge.

    shift > 0 — advance: drop first `shift`, pad end with zeros.
    shift < 0 — delay:   prepend `|shift|` zeros, drop end.
    shift == 0 — return unchanged.

    Length preserved. Stereo (C, T) handled.
    """
    if shift == 0:
        return wav
    if wav.ndim == 1:
        T = len(wav)
        out = np.zeros_like(wav)
        if shift > 0:
            n = max(0, T - shift)
            out[:n] = wav[shift:shift + n]
        else:
            absh = -shift
            n = max(0, T - absh)
            out[absh:absh + n] = wav[:n]
        return out
    elif wav.ndim == 2:
        T = wav.shape[1]
        out = np.zeros_like(wav)
        if shift > 0:
            n = max(0, T - shift)
            out[:, :n] = wav[:, shift:shift + n]
        else:
            absh = -shift
            n = max(0, T - absh)
            out[:, absh:absh + n] = wav[:, :n]
        return out
    raise ValueError(f"unsupported wav ndim={wav.ndim}")


# ---------------------------------------------------------------------------
# Diagnostic helper — useful for cohort A/B vs DTW-on comparison
# ---------------------------------------------------------------------------


def alignment_report(prev_tail: np.ndarray, cur_head: np.ndarray,
                     sr: int = 44100) -> dict:
    """Return alignment metadata WITHOUT applying the shift. Useful for
    logging the shift distribution across a cohort to see if DTW would
    have moved most junctions or just a few.

    Output:
        {
            'lag_samples': int,
            'lag_ms':      float,
            'confidence':  float,
            'would_apply': bool,
            'reason':      str,    # why apply / skip
        }
    """
    max_shift_ms = float(os.environ.get("AIJOCKEY_DTW_MAX_SHIFT_MS", "50"))
    min_confidence = float(os.environ.get("AIJOCKEY_DTW_MIN_CONFIDENCE", "0.3"))
    envelope_ms = float(os.environ.get("AIJOCKEY_DTW_ENVELOPE_MS", "20"))
    max_lag = int(sr * max_shift_ms / 1000.0)

    a = _envelope_mono(np.asarray(prev_tail), sr, envelope_ms)
    b = _envelope_mono(np.asarray(cur_head), sr, envelope_ms)
    n = int(min(len(a), len(b)))
    if n < sr // 4:
        return {"lag_samples": 0, "lag_ms": 0.0, "confidence": 0.0,
                "would_apply": False, "reason": "windows shorter than 250 ms"}
    lag, conf = _xcorr_peak_lag(a[:n], b[:n], max_lag)
    if conf < min_confidence:
        return {"lag_samples": lag, "lag_ms": 1000.0 * lag / sr,
                "confidence": conf, "would_apply": False,
                "reason": f"confidence {conf:.2f} < {min_confidence}"}
    if abs(lag) >= max_lag:
        return {"lag_samples": lag, "lag_ms": 1000.0 * lag / sr,
                "confidence": conf, "would_apply": False,
                "reason": "peak at clamp boundary, likely spurious"}
    return {"lag_samples": int(lag), "lag_ms": 1000.0 * lag / sr,
            "confidence": float(conf), "would_apply": (lag != 0),
            "reason": "ok" if lag != 0 else "zero lag"}


__all__ = [
    "enabled",
    "align_overlap",
    "alignment_report",
]
