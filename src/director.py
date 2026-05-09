"""
HF instruct-LM Director: user prompt (+ optional preset arc) → validated JSON plan.

Produces PlannerConfig-aligned fields plus transition_tiers (major|minor per junction).

Env:
  HF_DIRECTOR_MODEL   — Hugging Face model id (default: Qwen/Qwen2.5-7B-Instruct)
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

SYSTEM_PROMPT = """You are a professional club DJ. Output ONLY valid JSON, no markdown, no commentary.

Tier policy:
- "minor": smooth EQ swap or crossfade, used at most mix points.
- "major": a structurally significant moment — peak drop, energy shift, genre flip, big build resolution. Use sparingly.

Distribution rule: if there are N transitions, target roughly 1-2 majors when arc is "peak"/"rollercoaster", 0 majors when arc is "flat_low" or "descend", 1 when "build". Never all-minor for energetic arcs; never all-major.

Allowed arcs: build, peak, rollercoaster, descend, flat_high, flat_low.

Schema (return EXACT keys, no extras):
{
  "arc": "build|peak|rollercoaster|descend|flat_high|flat_low",
  "text_prompt": "<short vibe sentence>",
  "surprise_budget": <int 0-10>,
  "callback_budget": <int 0-3>,
  "transition_tiers": ["minor","major",...],   // length = max_expected_transitions
  "accent_hints": [ {"junction_index": 0, "fx_category": "hihat_rolls|risers|impacts|sweeps|snare_rolls", "beats": 2.0} ],
  "same_genre_tight_mix": false
}

junction_index = 0 means between clip 1 and clip 2.

Examples:

User: "festival peak time, big drops, anthemic" with 5 transitions
Output:
{"arc":"peak","text_prompt":"festival peak time euphoric drops","surprise_budget":3,"callback_budget":1,"transition_tiers":["minor","major","minor","major","minor"],"accent_hints":[{"junction_index":1,"fx_category":"risers","beats":4.0},{"junction_index":3,"fx_category":"impacts","beats":1.0}],"same_genre_tight_mix":false}

User: "after-hours noir, smoky melancholy" with 4 transitions
Output:
{"arc":"flat_low","text_prompt":"after-hours smoky lo-fi","surprise_budget":1,"callback_budget":0,"transition_tiers":["minor","minor","minor","minor"],"accent_hints":[],"same_genre_tight_mix":true}

User: "wild journey, peaks and drops" with 6 transitions
Output:
{"arc":"rollercoaster","text_prompt":"wild peaks and valleys","surprise_budget":4,"callback_budget":2,"transition_tiers":["minor","major","minor","major","minor","major"],"accent_hints":[{"junction_index":1,"fx_category":"risers","beats":4.0},{"junction_index":3,"fx_category":"snare_rolls","beats":2.0}],"same_genre_tight_mix":false}
"""


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
    audio_clip_paths: list[str] | None = None,
) -> dict[str, Any]:
    """
    Returns sanitized director dict compatible with PlannerConfig + apply_llm_transition_tiers.
    max_transitions = max(clip_count_estimate - 1, min(estimated_timeline_slots, ...))

    If audio_clip_paths provided AND HF_DIRECTOR_MODEL contains 'Audio',
    a multimodal audio-aware Director (e.g. Qwen2-Audio) is used. The model
    actually hears each clip's first window before producing the JSON plan.
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
            "Qwen/Qwen2.5-7B-Instruct",
        )
        is_audio_model = "audio" in model_id.lower() and audio_clip_paths
        llm_prompt = (
            f"User DJ request:\n{user_prompt}\n\n"
            f"Suggested arc preset: {arc_fb}\n"
            f"Clip pool size: {clip_count_estimate}, "
            f"produce transition_tiers of length exactly {mt}.\n"
        )
        try:
            if is_audio_model:
                out_text = _call_qwen2audio(llm_prompt, audio_clip_paths, model_id)
            else:
                out_text = _call_hf_instruct(llm_prompt + "\n" + SYSTEM_PROMPT, model_id)
            parsed = _extract_json_object(out_text or "")
            if parsed:
                return _sanitize_out(parsed, arc_fb, user_prompt, mt, coherence_hint)
        except Exception as e:
            print(f"[director] LLM failed ({e}), fallback")

    fb = _fallback_director(user_prompt, arc_fb, mt)
    return _sanitize_out(fb, arc_fb, user_prompt, mt, coherence_hint)


_AUDIO_LLM_CACHE: tuple[Any, Any, str] | None = None


def _call_qwen2audio(user_message: str, audio_paths: list[str],
                     model_id: str,
                     window_seconds: float = 30.0,
                     max_clips: int = 6) -> str:
    """Multimodal Director: model hears the first ~30s of each clip + reads
    the user message + system prompt, then emits JSON."""
    global _AUDIO_LLM_CACHE
    import torch
    import librosa

    if _AUDIO_LLM_CACHE and _AUDIO_LLM_CACHE[2] == model_id:
        proc, model = _AUDIO_LLM_CACHE[0], _AUDIO_LLM_CACHE[1]
    else:
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            model = model.cpu()
        _AUDIO_LLM_CACHE = (proc, model, model_id)

    audios: list = []
    target_sr = 16000
    for p in audio_paths[:max_clips]:
        try:
            y, _sr = librosa.load(p, sr=target_sr, mono=True, duration=window_seconds)
            audios.append(y)
        except Exception as e:
            print(f"[director-audio] skip {p}: {e}")

    if not audios:
        # No audio loaded — fall back to text path
        return _call_hf_instruct(user_message + "\n" + SYSTEM_PROMPT,
                                 "Qwen/Qwen2.5-7B-Instruct")

    # Build conversation with one <audio> placeholder per clip
    user_content: list[dict] = []
    for i, _ in enumerate(audios):
        user_content.append({"type": "audio", "audio_url": f"clip_{i}"})
    user_content.append({"type": "text", "text": user_message})
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    text = proc.apply_chat_template(conversation, add_generation_prompt=True,
                                    tokenize=False)
    inputs = proc(text=text, audios=audios, return_tensors="pt", padding=True,
                  sampling_rate=target_sr)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id,
        )
    new_tokens = gen[0, inputs["input_ids"].shape[1]:]
    return proc.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,           # ignored when do_sample=False; explicit for clarity
            repetition_penalty=1.05,
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
