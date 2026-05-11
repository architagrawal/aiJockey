"""Director self-critique loop.

After Director emits initial JSON plan, re-prompt the same LLM with
"Evaluate the plan against [Tomorrowland-arc / vocal-safety / variety
guidelines] and rewrite if it can be improved. Return revised JSON only."
N iterations, take the highest _score_plan() result.

Toggle: AIJOCKEY_DIRECTOR_CRITIQUE=1, AIJOCKEY_DIRECTOR_CRITIQUE_ITERS=2
"""
from __future__ import annotations

import json
import os


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_DIRECTOR_CRITIQUE", "0") == "1"


CRITIQUE_PROMPT = """You previously produced this DJ-set plan:

{plan_json}

Critique it against these objectives (briefly to yourself, do NOT include critique text in output):

1. Tomorrowland narrative arc — opener → climb → first peak → valley
   → bigger peak → callback → outro.
2. Tier diversity — mix of `minor`, `major`, `drop`. Avoid all-minor.
3. Vocal safety — vocal-heavy tracks must not coincide with aggressive
   transitions (chop / pitch_bend / bpm_warp).
4. Variety — callbacks earn surprise budget; don't camp on one technique.

If the plan can be improved, rewrite it. Output only the revised JSON
plan, no commentary. If the plan is already optimal, return it
unchanged.
"""


def critique_and_revise(initial_plan: dict, llm_call,
                          *, max_iters: int = 2,
                          score_fn = None) -> dict:
    """Iteratively critique + revise a Director plan.

    Args:
        initial_plan: dict from run_director.
        llm_call: callable(user_message: str) -> str (raw LLM output).
        max_iters: critique rounds.
        score_fn: optional callable(plan_dict) -> float. When provided,
            we only accept revisions that score higher than the
            current best.

    Returns: best plan dict.
    """
    if not enabled():
        return initial_plan
    best = initial_plan
    best_score = (score_fn(best) if score_fn else 0.0)
    for it in range(max_iters):
        try:
            prompt = CRITIQUE_PROMPT.format(plan_json=json.dumps(best, indent=2))
            raw = llm_call(prompt)
            # Extract first {...} JSON object from response
            from director import _extract_json_object as _extract  # type: ignore
            parsed = _extract(raw or "")
            if not parsed:
                continue
            if score_fn:
                s = score_fn(parsed)
                if s > best_score:
                    best, best_score = parsed, s
                    print(f"[critique] iter {it+1}: revised score "
                          f"{best_score:.2f}")
            else:
                best = parsed
        except Exception as e:
            print(f"[critique] iter {it+1} failed: {e}")
            continue
    return best
