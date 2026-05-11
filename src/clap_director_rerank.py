"""CLAP-text Director plan reranker.

Score N Director plans by CLAP-cosine of (synthesized plan_summary
text vs user_prompt). Pick plan with highest semantic alignment to
user intent. Plugs into director.run_director's multi-sample path.

Toggle: AIJOCKEY_CLAP_PLAN_RERANK=1
"""
from __future__ import annotations

import os


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_CLAP_PLAN_RERANK", "0") == "1"


def _summarize_plan(plan: dict) -> str:
    arc = plan.get("arc") or ""
    tp = plan.get("text_prompt") or ""
    tiers = plan.get("transition_tiers") or []
    tier_counts = {}
    for t in tiers:
        tier_counts[t] = tier_counts.get(t, 0) + 1
    tier_str = ", ".join(f"{n} {k}" for k, n in tier_counts.items())
    cb = plan.get("callback_budget", 0)
    sb = plan.get("surprise_budget", 0)
    return (f"DJ set arc={arc}. {tp}. Transitions: {tier_str}. "
            f"Callbacks {cb}, surprises {sb}.")


def rerank_plans(plans: list[dict], user_prompt: str) -> dict | None:
    """Returns plan with highest CLAP-cosine to user_prompt, or None."""
    if not enabled() or not plans:
        return None
    try:
        from clap_wrapper import get_text_embedding as text_embedding  # type: ignore
        import numpy as np
        prompt_emb = text_embedding(user_prompt or "")
        if prompt_emb is None:
            return None
        p_norm = float(np.linalg.norm(prompt_emb)) or 1.0
        prompt_u = prompt_emb / p_norm
        best, best_sim = None, -1.0
        for p in plans:
            summary = _summarize_plan(p)
            emb = text_embedding(summary)
            if emb is None:
                continue
            e_norm = float(np.linalg.norm(emb)) or 1.0
            sim = float((emb / e_norm) @ prompt_u)
            if sim > best_sim:
                best_sim, best = sim, p
        return best
    except Exception as e:
        print(f"[clap_plan_rerank] failed: {e}")
        return None
