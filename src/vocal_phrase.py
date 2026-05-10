"""Vocal-phrase-aware segment boundary adjuster.

Problem: section boundaries (chorus end @ 32.0s) and beat-grid downbeats
ignore where vocal LINES actually end. Cutting at the section/downbeat
boundary mid-vocal-phrase = audible word-chop = user dissatisfied.

Solution: detect vocal-stem RMS gaps near the nominal section boundary.
Adjust segment.end (and optionally start) to the nearest vocal-silence
within ±2 bars. Result: junction lands when the singer has finished
the line.

Public API:
    find_vocal_silence(vocals_stem, target_time_sec, sr,
                        search_window_sec=4.0,
                        silence_threshold_db=-30.0,
                        min_silence_ms=200.0)
        → adjusted_time_sec | None

    snap_segment_end_to_phrase_end(segment_dict, vocals_stem, sr,
                                     search_window_sec=4.0)
        → modified segment dict (start/end in seconds)

When `vocals_stem` is None / empty / RMS analysis fails: returns input
unchanged. Never raises. Cheap (~10 ms per call on 4-s window).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _to_mono(wav) -> np.ndarray | None:
    if wav is None:
        return None
    arr = np.asarray(wav, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        if arr.shape[0] in (1, 2):
            return arr.mean(axis=0).astype(np.float32)
        return arr.mean(axis=-1).astype(np.float32)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


def _rms_envelope(mono: np.ndarray, sr: int,
                   frame_ms: float = 20.0) -> np.ndarray:
    """Smoothed RMS envelope, decimated to one sample per frame.

    frame_ms=20 → 50 fps. Returns shape (T_frames,).
    """
    win = max(1, int(sr * frame_ms / 1000.0))
    if win >= len(mono):
        return np.array([float(np.sqrt(np.mean(mono ** 2 + 1e-12)))],
                         dtype=np.float32)
    sq = mono.astype(np.float32) ** 2
    csum = np.cumsum(np.insert(sq, 0, 0.0))
    rms = np.sqrt(np.maximum(0.0, (csum[win:] - csum[:-win]) / win) + 1e-12)
    return rms[::win].astype(np.float32)


def _db(x: np.ndarray) -> np.ndarray:
    """Convert RMS to dB. Floors at -120 dB to avoid log(0)."""
    return 20.0 * np.log10(np.maximum(x, 1e-6))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_vocal_silence(vocals_stem,
                       target_time_sec: float,
                       sr: int,
                       search_window_sec: float = 4.0,
                       silence_threshold_db: float = -30.0,
                       min_silence_ms: float = 200.0
                       ) -> float | None:
    """Find the nearest vocal-silence run to `target_time_sec`.

    Returns the start time of the silence run that begins closest
    (forward or backward) to `target_time_sec`. Useful for snapping a
    segment end to "the moment the singer stopped" rather than a clock
    boundary.

    Returns None when:
        - vocals_stem is None / empty
        - No silence run found within ±search_window_sec
        - Silence is shorter than min_silence_ms (transient gap, not a
          real phrase end)

    silence_threshold_db: dB level below which a frame counts as silent.
        -30 dB is a conservative threshold — quiet vocals (whispers,
        held notes fading) are still considered active. Lower threshold
        (-40, -50) catches MORE silences but may chop on breath gaps.

    min_silence_ms: minimum gap duration to count as a phrase end.
        200 ms typical. Below this is just a breath, not phrase-end.
    """
    mono = _to_mono(vocals_stem)
    if mono is None or len(mono) < sr // 4:
        return None

    frame_ms = 20.0
    rms = _rms_envelope(mono, sr, frame_ms=frame_ms)
    rms_db = _db(rms)
    silent = rms_db < silence_threshold_db

    target_frame = int(target_time_sec * 1000.0 / frame_ms)
    win_frames = int(search_window_sec * 1000.0 / frame_ms)
    lo = max(0, target_frame - win_frames)
    hi = min(len(silent), target_frame + win_frames + 1)
    if hi <= lo:
        return None

    min_silence_frames = max(1, int(min_silence_ms / frame_ms))

    # Scan for silence-RUNS within window. Track each run's (start_frame, length).
    runs: list[tuple[int, int]] = []
    in_silence = False
    start = 0
    for i in range(lo, hi):
        if silent[i]:
            if not in_silence:
                in_silence = True
                start = i
        else:
            if in_silence:
                in_silence = False
                if i - start >= min_silence_frames:
                    runs.append((start, i - start))
    if in_silence and (hi - start) >= min_silence_frames:
        runs.append((start, hi - start))

    if not runs:
        return None

    # Pick the silence run whose START is closest to target_frame
    best = min(runs, key=lambda r: abs(r[0] - target_frame))
    return float(best[0] * frame_ms / 1000.0)


def snap_segment_end_to_phrase_end(segment: dict,
                                    vocals_stem,
                                    sr: int,
                                    search_window_sec: float = 4.0,
                                    silence_threshold_db: float = -30.0,
                                    min_silence_ms: float = 200.0
                                    ) -> dict:
    """Adjust segment['end'] forward/backward to nearest vocal silence
    within ±search_window_sec. Returns a NEW segment dict; never mutates
    the input.

    When the segment has 'vocal_activity' < 0.10 (instrumental) OR
    vocals_stem is missing, returns input unchanged — no point snapping
    a non-vocal section.
    """
    seg = dict(segment)
    end = float(seg.get('end', 0.0))
    if end <= 0:
        return seg

    va = seg.get('vocal_activity')
    if isinstance(va, (int, float)) and va < 0.10:
        # Instrumental section — vocal phrase doesn't matter, leave as-is.
        return seg

    new_end = find_vocal_silence(
        vocals_stem, target_time_sec=end, sr=sr,
        search_window_sec=search_window_sec,
        silence_threshold_db=silence_threshold_db,
        min_silence_ms=min_silence_ms,
    )
    if new_end is None:
        return seg

    # Don't snap if the adjusted end is BEFORE start (degenerate)
    start = float(seg.get('start', 0.0))
    if new_end <= start + 1.0:    # require at least 1 sec of segment
        return seg

    seg['end'] = float(new_end)
    seg['vocal_phrase_snapped'] = True
    seg['vocal_phrase_drift_sec'] = float(new_end - end)
    return seg


def snap_segment_start_to_phrase_start(segment: dict,
                                        vocals_stem,
                                        sr: int,
                                        search_window_sec: float = 4.0,
                                        silence_threshold_db: float = -30.0,
                                        min_silence_ms: float = 200.0
                                        ) -> dict:
    """Mirror — adjust segment['start'] forward to land on vocal-silence
    boundary so the incoming clip's first vocal phrase plays in full.
    """
    seg = dict(segment)
    start = float(seg.get('start', 0.0))
    end = float(seg.get('end', 0.0))
    va = seg.get('vocal_activity')
    if isinstance(va, (int, float)) and va < 0.10:
        return seg

    new_start = find_vocal_silence(
        vocals_stem, target_time_sec=start, sr=sr,
        search_window_sec=search_window_sec,
        silence_threshold_db=silence_threshold_db,
        min_silence_ms=min_silence_ms,
    )
    if new_start is None or new_start >= end - 1.0:
        return seg

    seg['start'] = float(new_start)
    seg['vocal_phrase_snapped_start'] = True
    return seg


def has_vocal_phrase_at(vocals_stem, t_sec: float, sr: int,
                         lookahead_sec: float = 1.0,
                         threshold_db: float = -30.0) -> bool:
    """Quick check: is there an active vocal phrase at time `t_sec`?
    Useful for transition-time guard ("don't cut here, vocal is mid-line").
    """
    mono = _to_mono(vocals_stem)
    if mono is None:
        return False
    n_lookahead = int(lookahead_sec * sr)
    start_idx = max(0, int(t_sec * sr))
    end_idx = min(len(mono), start_idx + n_lookahead)
    if end_idx <= start_idx:
        return False
    chunk = mono[start_idx:end_idx]
    rms = float(np.sqrt(np.mean(chunk ** 2 + 1e-12)))
    rms_db = 20.0 * np.log10(max(rms, 1e-6))
    return rms_db >= threshold_db


__all__ = [
    'find_vocal_silence',
    'snap_segment_end_to_phrase_end',
    'snap_segment_start_to_phrase_start',
    'has_vocal_phrase_at',
]
