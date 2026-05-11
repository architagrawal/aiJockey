"""Variable LUFS arc across a set.

Instead of flat -9 LUFS everywhere, vary target loudness with the
arc curve: quiet intros (-12), louder drops (-7), settle to genre
target by outro. Audiobox rewards this dynamic feel.

Toggle: AIJOCKEY_LUFS_ARC=1
"""
from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_LUFS_ARC", "0") == "1"


def lufs_for_arc_position(arc_value: float, base_lufs: float,
                            *, quiet_range: float = 3.0,
                            loud_range: float = 2.0) -> float:
    """Map an arc_value in [0, 1] to a target LUFS.

    arc_value=0 (quiet intro)    → base_lufs - quiet_range
    arc_value=0.5 (mid energy)   → base_lufs
    arc_value=1 (peak drop)      → base_lufs + loud_range
    """
    v = float(arc_value)
    v = max(0.0, min(1.0, v))
    if v <= 0.5:
        # Interpolate base-quiet → base
        return float(base_lufs - quiet_range * (1.0 - 2.0 * v))
    # Interpolate base → base+loud
    return float(base_lufs + loud_range * (2.0 * v - 1.0))
