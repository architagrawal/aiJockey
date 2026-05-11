"""Section-role pair validator — refuse bad transitions before render.

Per dj_research §5 (Diagnosis 4): "Section role mismatch — drop-into-drop
without a planned double-drop, or breakdown-into-breakdown (energy crater).
Fix: validate (out_section, in_section) pair before approving tier."

This module returns:
    is_legal(out_section, in_section, tier, double_drop=False) -> bool
    recommended_tier(out_section, in_section) -> str

Used by planner to gate or reroute transitions whose section-pairing
will cause energy crashes or clashes.

Toggle: AIJOCKEY_SECTION_PAIR_VALIDATOR=1
"""
from __future__ import annotations

import os

# DJ research §1: canonical section vocabulary
INTRO = "intro"
OUTRO = "outro"
BREAKDOWN = "breakdown"
DROP = "drop"
VERSE = "verse"
CHORUS = "chorus"
BRIDGE = "bridge"
INSTRUMENTAL = "instrumental"
SOLO = "solo"
PRE_DROP = "pre_drop"
BUILD_UP = "build_up"
RIFF = "riff"

# Section role categories
_HIGH_ENERGY = {DROP, CHORUS, "peak"}
_LOW_ENERGY = {INTRO, OUTRO, BREAKDOWN, BRIDGE, "ambient", "interlude"}
_TRANSITIONAL = {OUTRO, INTRO, BREAKDOWN, BUILD_UP, PRE_DROP, BRIDGE}


def _norm(s: str | None) -> str:
    return (s or "").lower().strip()


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_SECTION_PAIR_VALIDATOR", "0") == "1"


def is_legal(out_section: str | None, in_section: str | None,
              tier: str = "minor", double_drop: bool = False) -> bool:
    """Returns True if the section pair is musically legal for this tier."""
    if not enabled():
        return True
    a, b = _norm(out_section), _norm(in_section)
    if not a or not b:
        return True   # unknown labels = soft-pass

    # First reject "energy crater" pairs (both low-energy, neither
    # outro→intro). These come before the transitional→transitional
    # soft-pass because crater is the worst named fault per dj_research.
    if a in _LOW_ENERGY and b in _LOW_ENERGY:
        # Golden exceptions: outro/breakdown → intro is the canonical
        # smooth transition. Bridge → intro also OK.
        if b == INTRO and a in {OUTRO, BREAKDOWN, BRIDGE}:
            return True
        return False

    # DROP → DROP is ONLY legal as a planned double-drop.
    if a == DROP and b == DROP:
        return bool(double_drop and tier in ("drop", "cut"))

    # GOLDEN: outgoing transitional region → incoming transitional region.
    if a in _TRANSITIONAL and b in _TRANSITIONAL:
        return True

    # CHORUS → CHORUS without cut/double-drop = stacked vocals clash.
    if a == CHORUS and b == CHORUS:
        return tier == "cut"

    # High-energy → low-energy: requires an outro/breakdown bridge.
    # Direct drop→intro is jarring unless tier=cut (deliberate hard cut).
    if a in _HIGH_ENERGY and b in _LOW_ENERGY:
        return tier in ("cut", "drop")

    # Low-energy → high-energy: ALWAYS legal (build-up payoff).
    if a in _LOW_ENERGY and b in _HIGH_ENERGY:
        return True

    return True   # default soft-pass for unmatched cases


def recommended_tier(out_section: str | None,
                       in_section: str | None) -> str:
    """Suggest the DJ-correct tier for a section pair.

    Per dj_research:
        outro → intro:           minor (long_crossfade)
        breakdown → intro:       major (eq_swap / filter_fade)
        breakdown → drop:        drop (build_riser_drop)
        drop → drop:             drop (double_drop only)
        drop → outro:            cut (hard cut)
        intro → drop:            drop
        chorus → verse:          minor (eq_swap)
    """
    a, b = _norm(out_section), _norm(in_section)
    if a == OUTRO and b == INTRO:
        return "minor"
    if a in {BREAKDOWN, BRIDGE} and b == INTRO:
        return "major"
    if a in {BREAKDOWN, BRIDGE, BUILD_UP} and b == DROP:
        return "drop"
    if a == DROP and b == DROP:
        return "drop"   # caller must set double_drop=True
    if a == DROP and b in _LOW_ENERGY:
        return "cut"
    if a in _LOW_ENERGY and b == DROP:
        return "drop"
    if a == CHORUS and b in {VERSE, BRIDGE, INSTRUMENTAL}:
        return "minor"
    return "minor"


def filter_timeline(timeline: list[dict], *,
                     director_double_drop: bool = False) -> list[dict]:
    """Walk a timeline and downgrade transitions whose (out_section,
    in_section) pair is illegal under is_legal(). Returns mutated copy.
    """
    if not enabled() or len(timeline) < 2:
        return timeline
    out = list(timeline)
    for i in range(1, len(out)):
        prev = out[i - 1]
        cur = out[i]
        seg_a = (prev.get("segment") or {}).get("type") or prev.get("section")
        seg_b = (cur.get("segment") or {}).get("type") or cur.get("section")
        tier = ((cur.get("transition_in") or {}).get("tier")
                or "minor")
        if not is_legal(seg_a, seg_b, tier,
                          double_drop=director_double_drop):
            new_tier = recommended_tier(seg_a, seg_b)
            ti = dict(cur.get("transition_in") or {})
            ti["tier"] = new_tier
            ti.setdefault("intent", "section-pair-rescue")
            ti["_rescued_from"] = tier
            cur["transition_in"] = ti
    return out
