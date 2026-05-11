"""MERT-conditioned plan reranker (pre-render).

For each candidate Director plan, score the *expected* PQ via the MERT
reward head applied to a concatenation of audio probes from the
plan's clips (intros / drops). Picks the plan whose clip-mix would
embed closest to high-PQ outputs.

Cheaper than rendering all N plans. Uses already-cached per-clip MERT
predictions when available (mert_pred.json sidecars). Otherwise falls
back to neutral score.

Toggle: AIJOCKEY_MERT_PLAN_RERANK=1
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_MERT_PLAN_RERANK", "0") == "1"


def _clip_mert_pred(cache_dir: str | Path, clip_id: str) -> dict | None:
    p = Path(cache_dir) / f"{clip_id}.mert_pred.json"
    if not p.exists():
        lib = os.environ.get("AIJOCKEY_LIBRARY_CACHE") or "/cache"
        p = Path(lib) / f"{clip_id}.mert_pred.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def score_plan_by_clips(plan: dict, clip_ids: list[str],
                          cache_dir: str | Path,
                          axes: tuple[str, ...] = ("PQ", "CE")) -> float:
    """Aggregate MERT predicted PQ/CE across plan's clip set."""
    vals = []
    for cid in clip_ids:
        pred = _clip_mert_pred(cache_dir, cid)
        if not pred:
            continue
        for ax in axes:
            if ax in pred:
                vals.append(float(pred[ax]))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def rerank_plans(plans: list[dict], clip_pool_ids: list[str],
                   cache_dir: str | Path) -> dict | None:
    """Pick plan with highest aggregate MERT-predicted PQ/CE.

    Each plan is scored over its `clip_sequence` if present, else over
    the full pool (as a proxy for which clips it would likely select).
    """
    if not enabled() or not plans:
        return None
    best = None
    best_score = -1.0
    for p in plans:
        cids = p.get("clip_sequence") or clip_pool_ids
        s = score_plan_by_clips(p, cids, cache_dir)
        if s > best_score:
            best_score = s
            best = p
    return best
