"""
Map LLM transition tier → allowlisted techniques (Tomorrowland-grade variety).

Tiers:
  - "minor"  : smooth blend (eq_swap, crossfade) — most mix points
  - "major"  : structurally significant (silence_drop, drum_break, filter_fade, echo_out)
  - "drop"   : building INTO an incoming drop (riser-style buildup → impact)
  - "cut"    : hard cut on the 1 (theatrical, on-beat) — used sparingly
  - "loop"   : loop_tighten / loop_callback (DJ stutter / repeat hook)

If the LLM's suggested tier is not one of these, it's normalized to "minor".
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Per-tier technique pools. Cycle by junction_idx for variety.
# ---------------------------------------------------------------------------

# Smooth transitions — used at most mix points so set doesn't sound choppy.
MINOR_TECHNIQUES: list[dict[str, Any]] = [
    {"name": "eq_swap", "bars": 12},
    {"name": "crossfade", "bars": 16},
    {"name": "eq_swap", "bars": 16},
]

# Genuinely DIFFERENT-sounding transitions for major moments.
# eq_swap removed from major to ensure audible distinction from minor.
MAJOR_TECHNIQUES: list[dict[str, Any]] = [
    {"name": "filter_fade", "bars": 16},
    {"name": "drum_break", "bars": 8},
    {"name": "silence_drop", "bars": 4, "silence_beats": 2},
    {"name": "echo_out", "bars": 8, "delay_beats": 0.5, "feedback": 0.55},
    {"name": "filter_fade", "bars": 24},
]

# Building INTO an incoming drop — riser/snare-roll buildup, then crashes
# into the next clip's drop section.
DROP_TECHNIQUES: list[dict[str, Any]] = [
    {"name": "loop_tighten", "bars": 16, "start_bars": 4},
    {"name": "drum_break", "bars": 16},
    {"name": "filter_fade", "bars": 24},
]

# Hard cut on the 1. Theatrical. Use sparingly.
CUT_TECHNIQUES: list[dict[str, Any]] = [
    {"name": "cut"},
    {"name": "silence_drop", "bars": 2, "silence_beats": 1},
]

# Loop-style tier: DJ stutter / hook callback.
LOOP_TECHNIQUES: list[dict[str, Any]] = [
    {"name": "loop_tighten", "bars": 16, "start_bars": 4},
    {"name": "loop_callback", "bars": 8, "repetitions": 2},
]

ALLOWED_TIERS = frozenset({"minor", "major", "drop", "cut", "loop"})


def tier_to_technique(tier: str, junction_idx: int) -> dict[str, Any]:
    t = (tier or "minor").lower().strip()
    if t not in ALLOWED_TIERS:
        t = "minor"
    pool = {
        "minor": MINOR_TECHNIQUES,
        "major": MAJOR_TECHNIQUES,
        "drop": DROP_TECHNIQUES,
        "cut": CUT_TECHNIQUES,
        "loop": LOOP_TECHNIQUES,
    }[t]
    base = pool[junction_idx % len(pool)]
    return dict(base)


ALLOWLIST_NAMES = frozenset(
    {"cut", "fade_in", "crossfade", "eq_swap", "filter_fade", "silence_drop",
     "drum_break", "mashup", "stem_swap", "echo_out", "spinback",
     "pitch_bend", "loop_tighten", "scratch_fill", "loop_callback"}
)


def maybe_apply_llm_technique_name(base: dict[str, Any], suggested: str | None) -> dict[str, Any]:
    if not suggested or not isinstance(suggested, str):
        return base
    name = suggested.strip()
    if name in ALLOWLIST_NAMES:
        merged = dict(base)
        merged["name"] = name
        bars = merged.get("bars", 16)
        if name in ("fade_in", "cut"):
            merged["bars"] = min(int(bars), 8)
        return merged
    return base
