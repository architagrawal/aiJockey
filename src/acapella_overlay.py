"""Acapella overlay transition — vocal of A on instrumental of B.

Per dj_research §3 LOOP tier: "Loop outgoing's vocal phrase over
incoming's drums. Mashup feel." We extend the idea: drop A's vocal
stem over B's drums+bass+other stems for 8-16 bars. Vocal stem
required on A's side.

Caller passes per-clip stem dicts (already loaded by execute.py).

Toggle in execute.py via transition_in.name == 'acapella_overlay'.
"""
from __future__ import annotations

import numpy as np


def acapella_overlay(stems_a: dict, stems_b: dict, sr: int = 44100,
                       bars: int = 8, beat_dur: float = 0.5,
                       vocal_db: float = -3.0,
                       inst_db: float = 0.0) -> np.ndarray:
    """Mix A's vocals over B's instrumental for `bars` bars.

    Args:
        stems_a: must contain 'vocals' as (2, n) np.ndarray.
        stems_b: must contain drums/bass/other (or any non-vocals).
        sr, beat_dur: timing.
        bars: overlay length in bars (4 beats each).
        vocal_db, inst_db: per-side level dB.

    Returns stereo overlay region (2, n). Caller splices into output.
    """
    n_target = int(bars * 4 * beat_dur * sr)
    voc = stems_a.get("vocals")
    if voc is None:
        # No vocal stem → return B's full mix unchanged.
        b_full = sum(s for n, s in stems_b.items()).astype(np.float32)
        return b_full[:, :n_target]
    b_inst = sum(s for nm, s in stems_b.items() if nm != "vocals").astype(np.float32)
    n = min(n_target, voc.shape[1], b_inst.shape[1])
    voc_gain = float(10.0 ** (vocal_db / 20.0))
    inst_gain = float(10.0 ** (inst_db / 20.0))
    out = (voc[:, :n] * voc_gain + b_inst[:, :n] * inst_gain).astype(np.float32)
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out
