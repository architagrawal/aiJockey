"""Genre-pair-specific transition rules.

Lookup table mapping (genre_a, genre_b) -> preferred transition tier
or technique. Director / planner consults this when both clips have
known genre tags. Falls through to default tier selection when
unspecified.

Toggle: AIJOCKEY_GENRE_PAIR_RULES=1
"""
from __future__ import annotations

import os


# Frozenset key (a, b) preserved order: A→B direction
GENRE_PAIR_PREFS: dict[tuple[str, str], dict] = {
    # Smooth same-family transitions
    ("house", "tech_house"): {"tier": "minor", "tech": "crossfade", "bars": 8},
    ("tech_house", "house"): {"tier": "minor", "tech": "crossfade", "bars": 8},
    ("house", "deep_house"): {"tier": "minor", "tech": "long_crossfade", "bars": 16},
    ("techno", "tech_house"): {"tier": "minor", "tech": "eq_swap", "bars": 8},
    ("tech_house", "techno"): {"tier": "minor", "tech": "eq_swap", "bars": 8},
    ("chillstep", "future_bass"): {"tier": "major", "tech": "build_riser_drop", "bars": 16},
    ("dnb", "dubstep"): {"tier": "major", "tech": "spectral_hold", "bars": 4},
    ("dubstep", "dnb"): {"tier": "major", "tech": "spectral_hold", "bars": 4},
    ("trance", "progressive"): {"tier": "minor", "tech": "crossfade", "bars": 16},
    ("progressive", "trance"): {"tier": "minor", "tech": "long_crossfade", "bars": 16},
    ("future_bass", "dubstep"): {"tier": "drop", "tech": "build_riser_drop", "bars": 16},
    # Hard / jolting jumps that NEED a riser or bridge
    ("ambient", "edm"): {"tier": "drop", "tech": "build_riser_drop", "bars": 16},
    ("lofi", "dnb"): {"tier": "drop", "tech": "build_riser_drop", "bars": 16},
    ("classical", "techno"): {"tier": "major", "tech": "spectral_hold", "bars": 8},
    # Non-Western harmonic family — softer transitions, no aggressive techs
    ("bollywood", "edm"): {"tier": "major", "tech": "spectral_hold", "bars": 4},
    ("punjabi", "house"): {"tier": "minor", "tech": "long_crossfade", "bars": 16},
    ("hindustani", "ambient"): {"tier": "minor", "tech": "crossfade", "bars": 16},
    # Trap / hip-hop bridges
    ("trap", "future_bass"): {"tier": "major", "tech": "punch_in", "bars": 4},
    ("hip_hop", "trap"): {"tier": "minor", "tech": "eq_swap", "bars": 8},
}


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_GENRE_PAIR_RULES", "0") == "1"


def lookup(genre_a: str | None, genre_b: str | None) -> dict | None:
    """Return preferred transition spec or None when no rule matches."""
    if not enabled():
        return None
    ga = (genre_a or "").lower().strip()
    gb = (genre_b or "").lower().strip()
    if not ga or not gb:
        return None
    rec = GENRE_PAIR_PREFS.get((ga, gb))
    if rec is not None:
        return dict(rec)
    return None
