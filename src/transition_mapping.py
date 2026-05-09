"""
Map LLM transition tier → allowlisted techniques (tasteful club defaults).

LLM selects only major|minor; code picks concrete DSP from transitions.execute.
"""

from __future__ import annotations

from typing import Any

MINOR_TECHNIQUES: list[dict[str, Any]] = [
    {"name": "eq_swap", "bars": 12},
    {"name": "crossfade", "bars": 16},
]

# Restrained majors: no spinback/airhorn defaults
MAJOR_TECHNIQUES: list[dict[str, Any]] = [
    {"name": "filter_fade", "bars": 16},
    {"name": "eq_swap", "bars": 24},
    {"name": "drum_break", "bars": 8},
]


def tier_to_technique(tier: str, junction_idx: int) -> dict[str, Any]:
    t = (tier or "minor").lower().strip()
    if t != "major":
        t = "minor"
        base = MINOR_TECHNIQUES[junction_idx % len(MINOR_TECHNIQUES)]
    else:
        base = MAJOR_TECHNIQUES[junction_idx % len(MAJOR_TECHNIQUES)]
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
