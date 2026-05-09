"""
HF instruct-LM Director: user prompt (+ optional preset arc) → validated JSON plan.

Produces PlannerConfig-aligned fields plus transition_tiers (major|minor per junction).

Env:
  HF_DIRECTOR_MODEL   — Hugging Face model id (default: Smol LM instruct class)
  AIJOCKEY_USE_DIRECTOR_LLM — "0" to skip HF and use deterministic fallback JSON
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

ALLOWED_TRANSITION_TIERS = frozenset({"major", "minor"})
ALLOWED_ARCS = (
    "build", "peak", "rollercoaster", "descend", "flat_high", "flat_low", "custom"
)

SYSTEM_PROMPT = """You are a professional club DJ assistant. Output ONLY valid JSON, no markdown.
Rules:
- Use transition tier "minor" for most mix points (smooth EQ swaps, crossfades).
- Use "major" sparingly: only where a noticeable energy or structural shift is warranted.
- Field transition_tiers: array of strings, each "minor" or "major", length = max_expected_transitions.

Schema keys (all optional except follow user intent):
{
  "arc": string (build|peak|rollercoaster|descend|flat_high|flat_low),
  "text_prompt": string (natural language mix vibe, may echo user),
  "surprise_budget": integer 0-50,
  "callback_budget": integer 0-5,
  "transition_tiers": ["minor", "minor", "major", ...],
  "accent_hints": [ { "junction_index": 0, "fx_category": "hihat_rolls", "beats": 2.0 } ],
  "same_genre_tight_mix": boolean
}

junction_index counts boundaries: 0 = between track 1 and 2 after planning (first transition)."""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _fallback_director(user_prompt: str, arc_fallback: str, max_transitions: int) -> dict[str, Any]:
    arc = arc_fallback if arc_fallback in ALLOWED_ARCS else "build"
    tiers = ["minor"] * max(1, max_transitions)
    if len(tiers) > 4 and len(tiers) // 8 > 0:
        tiers[len(tiers) // 4] = "major"
    return {
        "arc": arc,
        "text_prompt": user_prompt.strip() or "club mix, cohesive energy",
        "surprise_budget": 10,
        "callback_budget": 1,
        "transition_tiers": tiers[:max_transitions],
        "accent_hints": [],
        "same_genre_tight_mix": False,
        "_fallback": True,
    }


def _sanitize_out(raw: dict[str, Any], arc_fallback: str, user_prompt: str,
                  max_transitions: int, coherence_hint: float | None) -> dict[str, Any]:
    arc = raw.get("arc") or arc_fallback
    if isinstance(arc, str):
        arc = arc.lower().strip()
    if arc not in ALLOWED_ARCS:
        arc = arc_fallback if arc_fallback in ALLOWED_ARCS else "build"

    text_prompt = raw.get("text_prompt")
    if not isinstance(text_prompt, str) or not text_prompt.strip():
        text_prompt = user_prompt.strip() or "club DJ set"

    surprise_budget = raw.get("surprise_budget")
    if not isinstance(surprise_budget, int):
        try:
            surprise_budget = int(surprise_budget) if surprise_budget is not None else 10
        except (TypeError, ValueError):
            surprise_budget = 10
    surprise_budget = max(0, min(50, surprise_budget))

    callback_budget = raw.get("callback_budget")
    if not isinstance(callback_budget, int):
        try:
            callback_budget = int(callback_budget) if callback_budget is not None else 1
        except (TypeError, ValueError):
            callback_budget = 1
    callback_budget = max(0, min(5, callback_budget))

    tiers_in = raw.get("transition_tiers")
    tiers: list[str] = []
    if isinstance(tiers_in, list):
        for x in tiers_in:
            tx = str(x).lower().strip()
            if tx in ALLOWED_TRANSITION_TIERS:
                tiers.append(tx)
            else:
                tiers.append("minor")
    while len(tiers) < max_transitions:
        tiers.append("minor")
    tiers = tiers[:max_transitions]

    accents: list[dict[str, Any]] = []
    ah = raw.get("accent_hints")
    if isinstance(ah, list):
        for item in ah:
            if not isinstance(item, dict):
                continue
            ji = item.get("junction_index")
            try:
                ji = int(ji)
            except (TypeError, ValueError):
                continue
            fx = str(item.get("fx_category", "hihat_rolls"))
            beats = float(item.get("beats", 2.0))
            accents.append({"junction_index": ji, "fx_category": fx, "beats": beats})

    sg = raw.get("same_genre_tight_mix")
    if coherence_hint is not None and coherence_hint >= 0.72:
        sg = True
    elif not isinstance(sg, bool):
        sg = False

    return {
        "arc": arc,
        "text_prompt": text_prompt,
        "surprise_budget": surprise_budget,
        "callback_budget": callback_budget,
        "transition_tiers": tiers,
        "accent_hints": accents,
        "same_genre_tight_mix": sg,
        "_fallback": bool(raw.get("_fallback")),
    }


def run_director(
    user_prompt: str,
    arc_preset: str | None,
    clip_count_estimate: int,
    coherence_hint: float | None = None,
    max_transitions_hint: int | None = None,
    approx_duration_seconds: float = 600.0,
) -> dict[str, Any]:
    """
    Returns sanitized director dict compatible with PlannerConfig + apply_llm_transition_tiers.
    max_transitions = max(clip_count_estimate - 1, min(estimated_timeline_slots, ...))
    """
    arc_fb = arc_preset if arc_preset in ALLOWED_ARCS else "build"
    if max_transitions_hint is not None:
        mt = max(1, min(64, max_transitions_hint))
    else:
        mt = estimate_max_transitions_for_pool(clip_count_estimate, approx_duration_seconds)

    use_llm = os.environ.get("AIJOCKEY_USE_DIRECTOR_LLM", "1").lower() not in (
        "0", "false", "no",
    )
    if use_llm:
        model_id = os.environ.get(
            "HF_DIRECTOR_MODEL",
            "HuggingFaceTB/SmolLM2-360M-Instruct",
        )
        llm_prompt = (
            f"User DJ request:\n{user_prompt}\n\n"
            f"Suggested arc preset: {arc_fb}\n"
            f"Rough clip pool size ~{clip_count_estimate}, "
            f"produce transition_tiers of length exactly {mt}.\n"
            f"{SYSTEM_PROMPT}"
        )
        try:
            out_text = _call_hf_instruct(llm_prompt, model_id)
            parsed = _extract_json_object(out_text or "")
            if parsed:
                return _sanitize_out(parsed, arc_fb, user_prompt, mt, coherence_hint)
        except Exception as e:
            print(f"[director] LLM failed ({e}), fallback")

    fb = _fallback_director(user_prompt, arc_fb, mt)
    return _sanitize_out(fb, arc_fb, user_prompt, mt, coherence_hint)


_LLM_CACHE: tuple[Any, Any, str] | None = None


def _call_hf_instruct(user_message: str, model_id: str) -> str:
    global _LLM_CACHE
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if _LLM_CACHE and _LLM_CACHE[2] == model_id:
        tok, model = _LLM_CACHE[0], _LLM_CACHE[1]
    else:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            model = model.cpu()
        _LLM_CACHE = (tok, model, model_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    prompt = ""
    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "System:\n" + SYSTEM_PROMPT + "\nUser:\n" + user_message + "\nAssistant:\n"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    new_tokens = gen[0, inputs["input_ids"].shape[1] :]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def estimate_max_transitions_for_pool(n_clips: int, duration_sec: float) -> int:
    """Rough upper bound on transition count for tier array sizing."""
    if n_clips < 2:
        return max(8, min(48, int(duration_sec / 45)))
    guess = max(n_clips, int(duration_sec / 60) + n_clips)
    return max(8, min(64, guess))
