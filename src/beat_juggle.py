"""Beat juggle / loop-halve — LOOP tier tension build.

Per dj_research §3 LOOP tier: "1-beat or 1/2-beat loop on outgoing,
halve repeatedly (1 → 1/2 → 1/4 → 1/8) into incoming drop."

Takes last bar of A, loops it, halving the loop length every N
repeats. Increases tempo perception (each loop sounds faster) → drop
B on resolution.
"""
from __future__ import annotations

import numpy as np


def beat_juggle_transition(out_full: np.ndarray, in_full: np.ndarray,
                              sr: int, beat_dur: float,
                              start_subdivision: float = 1.0,
                              end_subdivision: float = 0.125,
                              n_steps: int = 4,
                              fade_to_b_seconds: float = 0.05
                              ) -> np.ndarray:
    """Halve loop length over n_steps, then snap to B on next downbeat.

    Args:
        out_full: stereo (2, n).
        in_full: stereo (2, n).
        sr: sample rate.
        beat_dur: seconds per beat.
        start_subdivision: initial loop length in beats (default 1).
        end_subdivision: final loop length in beats (default 1/8).
        n_steps: number of halving stages.
        fade_to_b_seconds: micro-fade to B at resolution.

    Returns concatenated waveform: A_pre + halving_loops + B_post.
    """
    bar_dur = beat_dur * 4.0
    bar_n = int(bar_dur * sr)
    a_len = out_full.shape[1]
    last_bar_start = max(0, a_len - bar_n)
    pre = out_full[:, :last_bar_start]
    bar = out_full[:, last_bar_start:]

    # Build subdivision schedule (halving)
    subs = []
    cur = start_subdivision
    for _ in range(n_steps):
        subs.append(cur)
        cur = max(end_subdivision, cur / 2.0)
    # Repeat each subdivision twice — DJ feel
    seq: list[np.ndarray] = []
    for sub in subs:
        loop_n = max(1, int(sub * beat_dur * sr))
        loop = bar[:, :loop_n]
        seq.append(loop)
        seq.append(loop)

    juggle = np.concatenate(seq, axis=1).astype(np.float32)

    # Micro-fade to B at end
    fade_n = max(1, int(fade_to_b_seconds * sr))
    if in_full.shape[1] >= fade_n and juggle.shape[1] >= fade_n:
        t = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)[None, :]
        juggle[:, -fade_n:] = juggle[:, -fade_n:] * (1.0 - t) + \
                                in_full[:, :fade_n] * t

    rest_b = in_full[:, fade_n:] if in_full.shape[1] > fade_n \
        else np.zeros((in_full.shape[0], 0), dtype=np.float32)
    return np.concatenate([pre, juggle, rest_b], axis=1).astype(np.float32)
