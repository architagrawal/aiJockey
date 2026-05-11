"""Beat-grid BPM normalization to integer targets.

Snaps each timeline entry's target_bpm to nearest integer (128, 140
etc) so all clips lock to the same global grid. Tighter mix, fewer
sub-sample phase drifts.

Toggle: AIJOCKEY_BPM_GRID=1, AIJOCKEY_BPM_GRID_SNAPS="120,124,128,130,135,140,150,170,174"
"""
from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_BPM_GRID", "0") == "1"


def _snap_targets() -> list[int]:
    raw = os.environ.get("AIJOCKEY_BPM_GRID_SNAPS",
                           "120,124,128,130,135,140,150,170,174")
    out = []
    for tok in raw.split(","):
        try:
            out.append(int(tok.strip()))
        except Exception:
            continue
    return out or [128]


def snap_bpm(bpm: float, max_drift_pct: float = 0.06) -> float:
    """Return integer-grid BPM if within max_drift_pct of a target.

    Otherwise return input unchanged.
    """
    if not enabled() or bpm <= 0:
        return float(bpm)
    targets = _snap_targets()
    best = min(targets, key=lambda t: abs(t - bpm))
    drift = abs(best - bpm) / max(bpm, 1.0)
    if drift <= max_drift_pct:
        return float(best)
    return float(bpm)


def snap_timeline(timeline: list[dict]) -> list[dict]:
    """In-place snap of each entry's target_bpm."""
    if not enabled():
        return timeline
    for e in timeline:
        if "target_bpm" in e:
            e["target_bpm"] = snap_bpm(float(e["target_bpm"]))
    return timeline
