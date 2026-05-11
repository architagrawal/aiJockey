"""Genre-aware phrase length / structure conventions.

Per dj_research §1: genre variants set phrase grid expectations:
    House / Tech-house : 32-bar intro/outro, 16-bar breakdown.
    Techno             : groove-driven, no big drops, loop-heavy.
    Drum & Bass        : 32-bar intro, breakdown at 32, drop at 64.
    Trance / Big-room  : long build (16-32 bars riser+snare-roll), 1 drop.
    Hip-hop / Pop      : no DJ intro; use cuts/echo-outs/acapellas.

This module exposes per-genre defaults the planner / Director can use
to bias transition lengths, drop placement, and tier choices.

Toggle: AIJOCKEY_GENRE_PHRASE=1
"""
from __future__ import annotations

import os

# (intro_bars, outro_bars, breakdown_bars, drop_at_bar)
PHRASE_DEFAULTS: dict[str, dict] = {
    "house":       {"intro": 32, "outro": 32, "breakdown": 16, "drop_at": None, "preferred_tier": "minor"},
    "tech_house":  {"intro": 32, "outro": 32, "breakdown": 16, "drop_at": None, "preferred_tier": "minor"},
    "deep_house":  {"intro": 32, "outro": 32, "breakdown": 16, "drop_at": None, "preferred_tier": "minor"},
    "techno":      {"intro": 32, "outro": 32, "breakdown": 16, "drop_at": None, "preferred_tier": "minor"},
    "progressive": {"intro": 32, "outro": 32, "breakdown": 16, "drop_at": 32,   "preferred_tier": "major"},
    "dnb":         {"intro": 32, "outro": 32, "breakdown": 32, "drop_at": 64,   "preferred_tier": "drop"},
    "dubstep":     {"intro": 16, "outro": 16, "breakdown": 8,  "drop_at": 32,   "preferred_tier": "drop"},
    "trance":      {"intro": 32, "outro": 32, "breakdown": 16, "drop_at": 32,   "preferred_tier": "drop"},
    "edm":         {"intro": 16, "outro": 16, "breakdown": 8,  "drop_at": 32,   "preferred_tier": "drop"},
    "future_bass": {"intro": 16, "outro": 16, "breakdown": 8,  "drop_at": 32,   "preferred_tier": "drop"},
    "hardstyle":   {"intro": 16, "outro": 16, "breakdown": 8,  "drop_at": 16,   "preferred_tier": "drop"},
    "trap":        {"intro": 8,  "outro": 8,  "breakdown": 4,  "drop_at": None, "preferred_tier": "cut"},
    "hip_hop":     {"intro": 0,  "outro": 0,  "breakdown": 0,  "drop_at": None, "preferred_tier": "cut"},
    "pop":         {"intro": 0,  "outro": 0,  "breakdown": 0,  "drop_at": None, "preferred_tier": "cut"},
    "rnb":         {"intro": 4,  "outro": 4,  "breakdown": 4,  "drop_at": None, "preferred_tier": "minor"},
    "chillstep":   {"intro": 16, "outro": 16, "breakdown": 16, "drop_at": None, "preferred_tier": "minor"},
    "lofi":        {"intro": 4,  "outro": 4,  "breakdown": 0,  "drop_at": None, "preferred_tier": "minor"},
    "ambient":     {"intro": 8,  "outro": 8,  "breakdown": 0,  "drop_at": None, "preferred_tier": "minor"},
    "punjabi":     {"intro": 8,  "outro": 8,  "breakdown": 8,  "drop_at": None, "preferred_tier": "minor"},
    "bollywood":   {"intro": 8,  "outro": 8,  "breakdown": 8,  "drop_at": None, "preferred_tier": "minor"},
    "synthwave":   {"intro": 16, "outro": 16, "breakdown": 8,  "drop_at": None, "preferred_tier": "minor"},
    "retrowave":   {"intro": 16, "outro": 16, "breakdown": 8,  "drop_at": None, "preferred_tier": "minor"},
}


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_GENRE_PHRASE", "0") == "1"


def phrase_for(genre: str | None) -> dict:
    """Return phrase defaults for genre, fallback = house."""
    if not enabled():
        return PHRASE_DEFAULTS["house"]
    g = (genre or "").lower().strip()
    return PHRASE_DEFAULTS.get(g, PHRASE_DEFAULTS["house"])


def transition_bars(out_genre: str | None,
                     in_genre: str | None,
                     tier: str = "minor") -> int:
    """Recommended bar count for a transition between two genres at a tier.

    Strategy: take the SHORTER of the two genres' outro/intro lengths;
    for drop tier, default 8 bars (build window). For minor tier on
    house→house, full 32-bar long blend.
    """
    a = phrase_for(out_genre)
    b = phrase_for(in_genre)
    if tier == "drop":
        return 8
    if tier == "cut":
        return 1
    if tier == "major":
        return min(a["outro"], b["intro"], 16) or 8
    # minor: long blend
    return min(a["outro"], b["intro"], 32) or 8
