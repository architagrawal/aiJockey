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


def tier_to_technique(tier: str, junction_idx: int,
                      vocal_active: bool = False,
                      section_label: str | None = None) -> dict[str, Any]:
    """Pick a technique for a tier + junction context.

    When `vocal_active=True` (vocal_activity > 0.30 on EITHER junction
    side), filters out vocal-unsafe techniques per catalog.json
    `vocal_safe=False` — chop, tape_stop, drum_replace, kickless_swap,
    spinback, forward_spin, build_riser_drop, snare_buildup, etc.

    When `section_label` provided (e.g. 'chorus', 'break', 'bridge'),
    additionally filters by catalog `incompatible_with` rules.

    Falls through to legacy hardcoded pools when catalog unavailable
    or the filtered set is empty for this context.
    """
    t = (tier or "minor").lower().strip()
    if t not in ALLOWED_TIERS:
        t = "minor"

    # Catalog-driven path (preferred when catalog reachable + filters fire).
    try:
        from transition_catalog import technique_for_context, get as cat_get
        ctx_pool = technique_for_context(
            tier=t,
            section_label=section_label,
            vocal_active=vocal_active,
            status="implemented",
        )
        if ctx_pool:
            chosen = ctx_pool[junction_idx % len(ctx_pool)]
            base: dict[str, Any] = {"name": chosen["name"], "tier": t}
            # Carry typical_bars[0] as default bars hint (downstream may override)
            tb = chosen.get("typical_bars") or [16]
            try:
                base["bars"] = int(tb[0]) if tb else 16
            except Exception:
                base["bars"] = 16
            # Pull a couple of common params from the catalog as sane defaults
            params = chosen.get("params") or {}
            for k, default in (("delay_beats", 0.5), ("feedback", 0.55),
                                ("silence_beats", 2)):
                if k in params and k not in base:
                    base[k] = default
            base["catalog_picked"] = True
            base["vocal_safe"] = bool(chosen.get("vocal_safe"))
            return base
    except Exception:
        # Catalog import error or other — fall through to legacy.
        pass

    # Legacy hardcoded pool (back-compat).
    pool = {
        "minor": MINOR_TECHNIQUES,
        "major": MAJOR_TECHNIQUES,
        "drop": DROP_TECHNIQUES,
        "cut": CUT_TECHNIQUES,
        "loop": LOOP_TECHNIQUES,
    }[t]
    base = dict(pool[junction_idx % len(pool)])
    base["tier"] = t
    return base


ALLOWLIST_NAMES = frozenset(
    {"cut", "fade_in", "crossfade", "eq_swap", "filter_fade", "silence_drop",
     "drum_break", "mashup", "stem_swap", "echo_out", "spinback",
     "pitch_bend", "loop_tighten", "scratch_fill", "loop_callback",
     # Tier-1 catalog upgrades (post-a00b5cd)
     "bass_swap", "highs_swap", "highpass_sweep_in", "punch_in", "chop",
     "loop_roll", "beat_juggle", "acapella_drop", "instrumental_swap",
     "kickless_swap", "drum_replace", "reverb_wash", "forward_spin",
     "tape_stop", "spectral_hold", "bpm_warp", "harmonic_overlay"}
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
