"""Echo throw / delay throw transition (pro DJ classic).

Per dj_research §3 MAJOR tier: outgoing's last 1-2 beats are caught in
a long delay (3/4 dotted or 1-bar) with feedback ~0.5, while the dry
signal of A fades out and B fades in clean. Effect = "echo of A's
last hit cascading over B's intro".

Toggle in transitions.py picker.
"""
from __future__ import annotations

import numpy as np


def _delay_line(x: np.ndarray, sr: int, delay_seconds: float,
                  feedback: float = 0.5,
                  decay_seconds: float = 4.0) -> np.ndarray:
    """Apply delay+feedback to mono/stereo signal in-place style.

    Returns signal of length original + decay_seconds * sr (so tail
    rings out).
    """
    n_in = x.shape[-1]
    n_decay = int(decay_seconds * sr)
    n_delay = max(1, int(delay_seconds * sr))
    pad = np.zeros((x.shape[0], n_decay), dtype=np.float32) if x.ndim == 2 \
        else np.zeros(n_decay, dtype=np.float32)
    y = np.concatenate([x.astype(np.float32),
                        pad], axis=-1).astype(np.float32)
    n = y.shape[-1]
    fb = float(max(0.0, min(0.95, feedback)))
    if x.ndim == 2:
        for c in range(y.shape[0]):
            for i in range(n_delay, n):
                y[c, i] = y[c, i] + fb * y[c, i - n_delay]
    else:
        for i in range(n_delay, n):
            y[i] = y[i] + fb * y[i - n_delay]
    return y


def echo_throw_transition(out_full: np.ndarray, in_full: np.ndarray,
                            sr: int, bars: int = 2, beat_dur: float = 0.5,
                            feedback: float = 0.55,
                            wet_db: float = -3.0) -> np.ndarray:
    """Echo-throw on the last (bars * 4 beats) of out_full, blend into in_full.

    Strategy:
        - Capture last `bars` bars of A.
        - Feed into a delay-line tuned to 3/4 dotted or 1-bar (use bar_dur).
        - Sum dry A (fading out) + wet delay tail.
        - At delay tail's halfway, fade in B.

    Returns full waveform: A_pre + echo_tail_overlap_with_B + B_post.
    """
    bar_dur = beat_dur * 4.0
    delay_s = bar_dur * 0.75   # 3/4 dotted = classic DJ echo
    n_capture = int(bars * bar_dur * sr)
    n = min(n_capture, out_full.shape[1])
    pre = out_full[:, :out_full.shape[1] - n]
    tail = out_full[:, -n:]

    decay_s = bar_dur * 4.0
    echo_tail = _delay_line(tail, sr, delay_s, feedback=feedback,
                              decay_seconds=decay_s)
    wet_gain = float(10.0 ** (wet_db / 20.0))
    # Fade out dry portion of tail (existing) while wet remains
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    dry_fade = (1.0 - t)[None, :]
    # Echo tail signal starts at length n + n_decay; original first n
    # already includes dry → multiply by dry_fade
    echo_tail[:, :n] = echo_tail[:, :n] * dry_fade
    echo_tail = echo_tail * wet_gain + 0  # wet level

    # Blend incoming B: full strength after first bar of delay-tail
    n_overlap = min(int(decay_s * sr), in_full.shape[1])
    overlap = np.zeros_like(echo_tail)
    in_seg = in_full[:, :overlap.shape[1]] if in_full.shape[1] >= overlap.shape[1] \
        else np.pad(in_full, ((0, 0),
                                (0, overlap.shape[1] - in_full.shape[1])))
    fade_in = np.linspace(0.0, 1.0, overlap.shape[1], dtype=np.float32)[None, :]
    overlap[:, :in_seg.shape[1]] = in_seg * fade_in[:, :in_seg.shape[1]]

    overlap_len = min(echo_tail.shape[1], overlap.shape[1])
    mixed = echo_tail[:, :overlap_len] + overlap[:, :overlap_len]

    rest_b = in_full[:, overlap_len:] if in_full.shape[1] > overlap_len \
        else np.zeros((in_full.shape[0], 0), dtype=np.float32)
    out = np.concatenate([pre, mixed, rest_b], axis=1).astype(np.float32)
    return out
