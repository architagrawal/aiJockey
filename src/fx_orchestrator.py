"""FX orchestrator — prevents effect stacking that makes mixes feel busy.

Real-DJ principle: ONE primary technique per transition, not all at
once. User feedback (2026-05-11): "too many noises at same time, no
plan". Audiobox PQ doesn't catch this — needs explicit budget.

Two mechanisms:
    1. MUTEX pairs — when both env flags ON, pick ONE per junction.
       (sidechain_duck XOR freq_duck)
       (deesser XOR spec_xfade — overlap on vocal-band processing)
    2. Per-set FX budget — only N% of junctions get aggressive fx,
       rest get plain crossfade.

Toggle: AIJOCKEY_FX_ORCHESTRATOR=1
"""
from __future__ import annotations

import os
import random


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_FX_ORCHESTRATOR", "0") == "1"


# Mutex groups: when 2+ are simultaneously enabled, only 1 wins per
# junction (chosen by junction-index hash for determinism).
MUTEX_GROUPS = {
    "overlap_processing": [
        ("AIJOCKEY_SIDECHAIN_DUCK", "sidechain"),
        ("AIJOCKEY_FREQ_DUCK", "freq_duck"),
    ],
    "vocal_clarity": [
        ("AIJOCKEY_DEESSER", "deesser"),
        ("AIJOCKEY_SPEC_XFADE", "spec_xfade"),
    ],
    "stereo_image": [
        ("AIJOCKEY_MS_WIDEN", "ms_widen"),
        ("AIJOCKEY_MS_MULTIBAND", "ms_multiband"),
    ],
}


def pick_for_junction(junction_idx: int,
                        group_name: str) -> str | None:
    """Return the single-winner fx name for `group_name` at this
    junction, or None if no fx in group is enabled."""
    if not enabled():
        return None
    group = MUTEX_GROUPS.get(group_name) or []
    active = [(env, name) for env, name in group
               if os.environ.get(env, "0") == "1"]
    if not active:
        return None
    if len(active) == 1:
        return active[0][1]
    # Deterministic pick by junction index hash
    return active[junction_idx % len(active)][1]


def is_fx_active(env_flag: str, junction_idx: int,
                   group_name: str | None = None) -> bool:
    """Wrapper that respects MUTEX_GROUPS + per-set FX budget.

    Replaces direct `os.environ.get(env_flag,'0')=='1'` checks in
    callsites — looks up the env flag, but if it belongs to a mutex
    group, defers to pick_for_junction's winner.
    """
    if not enabled():
        return os.environ.get(env_flag, "0") == "1"
    if os.environ.get(env_flag, "0") != "1":
        return False
    # If env is part of a group, check whether it wins this junction
    for gname, members in MUTEX_GROUPS.items():
        envs = {e for e, _ in members}
        if env_flag in envs:
            winner = pick_for_junction(junction_idx, gname)
            return winner == dict(members)[env_flag]
    return True


# ---------------------------------------------------------------------------
# Per-set FX budget — cap fraction of junctions that get aggressive fx
# ---------------------------------------------------------------------------

def junction_gets_fx(junction_idx: int, total_junctions: int,
                       fraction: float | None = None,
                       seed: int = 42) -> bool:
    """Returns True iff this junction is allowed to apply aggressive fx.

    `fraction` = portion of junctions that get fx. Default from
    AIJOCKEY_FX_BUDGET_FRACTION env (0.4 = 40% get fx, 60% clean).

    Real-DJ heuristic: peak/build junctions get fx, intermediate
    blends stay clean. Without section labels we use deterministic
    pseudo-random selection so the same junction always gets the
    same decision.
    """
    if not enabled():
        return True
    if fraction is None:
        try:
            fraction = float(os.environ.get(
                "AIJOCKEY_FX_BUDGET_FRACTION", "0.4"))
        except Exception:
            fraction = 0.4
    if fraction >= 1.0:
        return True
    if fraction <= 0.0:
        return False
    rng = random.Random(seed + junction_idx)
    return rng.random() < fraction
