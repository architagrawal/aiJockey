"""Per-stem LUFS normalization before transition.

Sony FxNorm-automix pattern: equalize loudness of drums/bass/other/vocals
across clips so loud-clip-bullies-quiet-clip is fixed before any
overlap math.

Toggle: AIJOCKEY_STEM_NORM=1
Targets via env:
    AIJOCKEY_STEM_NORM_DRUMS    default -14
    AIJOCKEY_STEM_NORM_BASS     default -14
    AIJOCKEY_STEM_NORM_OTHER    default -16
    AIJOCKEY_STEM_NORM_VOCALS   default -18
"""
from __future__ import annotations

import os

import numpy as np

try:
    import pyloudnorm as pyln  # type: ignore
    _HAS_PYLN = True
except Exception:
    _HAS_PYLN = False


DEFAULTS = {"drums": -14.0, "bass": -14.0, "other": -16.0, "vocals": -18.0}


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_STEM_NORM", "0") == "1"


def _target_for(stem_name: str) -> float:
    key = f"AIJOCKEY_STEM_NORM_{stem_name.upper()}"
    try:
        return float(os.environ.get(key, DEFAULTS.get(stem_name, -16.0)))
    except Exception:
        return DEFAULTS.get(stem_name, -16.0)


def normalize_stems(stems: dict, sr: int = 44100) -> dict:
    """Returns dict with each stem LUFS-normalized to its per-stem target.

    Skips silently when pyloudnorm missing OR loudness undefined.
    """
    if not enabled() or not _HAS_PYLN:
        return stems
    out: dict = {}
    meter = pyln.Meter(sr)
    for name, s in stems.items():
        if not isinstance(s, np.ndarray) or s.ndim != 2:
            out[name] = s; continue
        try:
            loud = meter.integrated_loudness(s.T)
            if not np.isfinite(loud) or loud < -70:
                out[name] = s; continue
            target = _target_for(name)
            s_norm = pyln.normalize.loudness(s.T, loud, target).T.astype(
                np.float32)
            out[name] = s_norm
        except Exception:
            out[name] = s
    return out
