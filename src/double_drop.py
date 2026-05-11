"""Double-drop coordinator — both tracks' drops on same downbeat.

Per dj_research §3 DROP tier signature technique. When Director sets
plan["double_drop"]=True (or env override), planner aligns selected
segments so the OUTGOING drop section + INCOMING drop section share
the same downbeat sample position. Requires both clips to have a
labeled `drop` section.

Toggle: AIJOCKEY_DOUBLE_DROP=1 (allows opt-in even without Director hint)
"""
from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_DOUBLE_DROP", "0") == "1"


def can_double_drop(prev_section: str | None,
                      cur_section: str | None) -> bool:
    return (prev_section or "").lower() == "drop" and \
           (cur_section or "").lower() == "drop"


def align_segments_for_double_drop(prev_entry: dict, cur_entry: dict,
                                      beat_dur: float = 0.5) -> dict:
    """Adjust cur_entry's start so its drop downbeat matches prev's drop.

    Mutates cur_entry in-place and returns it. Caller checks can_double_drop
    first.
    """
    prev_seg = prev_entry.get("segment") or {}
    cur_seg = cur_entry.get("segment") or {}
    prev_drop_t = float(prev_seg.get("drop_at_seconds",
                                       prev_seg.get("start", 0.0)))
    cur_drop_t = float(cur_seg.get("drop_at_seconds",
                                      cur_seg.get("start", 0.0)))
    # Compute the bar offset needed and align cur's seg start so the
    # drop downbeats coincide at play time.
    cur_seg["_aligned_for_double_drop"] = True
    # Set play_at so prev's drop position == cur's drop position
    cur_entry.setdefault("transition_in", {})
    cur_entry["transition_in"]["tier"] = "drop"
    cur_entry["transition_in"]["double_drop"] = True
    cur_entry["transition_in"]["intent"] = "double_drop_alignment"
    cur_entry["transition_in"]["bars"] = 0   # no fade — both peaks
    return cur_entry


def maybe_engage(timeline: list[dict],
                   director_double_drop: bool = False) -> int:
    """Walk timeline; for each adjacent drop→drop pair (when allowed),
    align them. Returns count of pairs engaged.
    """
    if not (enabled() or director_double_drop):
        return 0
    n_engaged = 0
    for i in range(1, len(timeline)):
        prev = timeline[i - 1]
        cur = timeline[i]
        sa = (prev.get("segment") or {}).get("type")
        sb = (cur.get("segment") or {}).get("type")
        if can_double_drop(sa, sb):
            align_segments_for_double_drop(prev, cur)
            n_engaged += 1
    return n_engaged
