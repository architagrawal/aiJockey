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

ALLOWED_TRANSITION_TIERS_FULL = frozenset({"minor", "major", "drop", "cut", "loop"})
PHASE1_ALLOWED_TIERS = frozenset({"minor", "major", "drop"})


def _allowed_tiers() -> frozenset[str]:
    """Phase 1 (default) restricts to minor/major/drop. cut+loop need
    material-aware gating (instrumental-only, phrase-aligned) — Phase 2.
    Set AIJOCKEY_PHASE=2 to enable full vocab.
    """
    if os.environ.get("AIJOCKEY_PHASE", "1") == "1":
        return PHASE1_ALLOWED_TIERS
    return ALLOWED_TRANSITION_TIERS_FULL


# Back-compat alias used elsewhere in the module + tests.
ALLOWED_TRANSITION_TIERS = ALLOWED_TRANSITION_TIERS_FULL
ALLOWED_ARCS_FULL = (
    "build", "peak", "rollercoaster", "descend", "flat_high", "flat_low", "custom"
)
PHASE1_ALLOWED_ARCS = ("build", "peak", "flat_low")


def _allowed_arcs() -> tuple[str, ...]:
    if os.environ.get("AIJOCKEY_PHASE", "1") == "1":
        return PHASE1_ALLOWED_ARCS
    return ALLOWED_ARCS_FULL


# Back-compat alias used in run_director / _sanitize_out / fallback paths
ALLOWED_ARCS = ALLOWED_ARCS_FULL

SYSTEM_PROMPT_FULL = """You are a Tomorrowland-grade club DJ planning a live set. Output ONLY valid JSON, no markdown, no commentary.

You design for DRAMA, not just smooth blends. Real performance has builds, drops, dead-silence moments, hard cuts, and loop stutters — pick tiers accordingly.

Five transition tiers:
- "minor": smooth EQ swap / crossfade. Use at MOST mix points.
- "major": structurally significant — filter close, drum-only break, echo-out tail, or a held-silence beat. Picks a different DSP per junction.
- "drop":  use this when the NEXT clip's section is a drop/peak. Engineers a riser-style buildup INTO the drop. Pick this for the climax moment.
- "cut":   hard cut on the 1. Theatrical, abrupt. Use rarely (1-2 per set max).
- "loop":  DJ stutter (loop_tighten) or hook callback (loop_callback). Use 0-1 times per set.

Distribution policy by arc:
- "peak"        : ~30% major, 1 drop, 0-1 cut. No more than half minor.
- "rollercoaster": alternate. ~30% major, ~15% drop, 0-1 cut, occasional loop.
- "build"       : start minor, end with 1 drop tier into the climax.
- "descend"     : mostly minor, 1 major early, fade to minor.
- "flat_high"   : ~25% major, 1 drop, no cuts.
- "flat_low"    : all minor, possibly 1 loop.
NEVER all-minor for energetic arcs. NEVER more than half major (that's chaos).

Accent hints — overlay short FX at specific junctions:
- "risers"      : 4-8 beat sweep BEFORE a drop or major moment
- "impacts"     : 1-beat boom AT a drop landing or hard cut
- "snare_rolls" : 2-4 beat snare buildup before a drop
- "sweeps"      : 4-bar filter sweep during a major filter_fade
- "hihat_rolls" : 1-2 beat tension lift during eq_swap on energy lifts

Allowed arcs: build, peak, rollercoaster, descend, flat_high, flat_low.

Schema (EXACT keys, no extras):
{
  "arc": "build|peak|rollercoaster|descend|flat_high|flat_low",
  "text_prompt": "<short vibe sentence>",
  "surprise_budget": <int 0-10>,
  "callback_budget": <int 0-3>,
  "transition_tiers": ["minor","major","drop","cut","loop",...],
  "accent_hints": [ {"junction_index": 0, "fx_category": "risers|impacts|snare_rolls|sweeps|hihat_rolls", "beats": 4.0} ],
  "same_genre_tight_mix": false
}

junction_index = 0 means between clip 1 and clip 2.

Examples:

User: "festival peak time, big drops, anthemic" with 5 transitions
Output:
{"arc":"peak","text_prompt":"festival peak euphoric drops","surprise_budget":3,"callback_budget":1,"transition_tiers":["minor","drop","major","drop","major"],"accent_hints":[{"junction_index":1,"fx_category":"risers","beats":8.0},{"junction_index":1,"fx_category":"impacts","beats":1.0},{"junction_index":3,"fx_category":"snare_rolls","beats":4.0},{"junction_index":3,"fx_category":"impacts","beats":1.0}],"same_genre_tight_mix":false}

User: "after-hours noir, smoky melancholy" with 4 transitions
Output:
{"arc":"flat_low","text_prompt":"after-hours smoky lo-fi","surprise_budget":1,"callback_budget":0,"transition_tiers":["minor","minor","minor","minor"],"accent_hints":[],"same_genre_tight_mix":true}

User: "wild journey, peaks and drops" with 6 transitions
Output:
{"arc":"rollercoaster","text_prompt":"wild peaks and valleys","surprise_budget":4,"callback_budget":2,"transition_tiers":["minor","drop","major","loop","drop","cut"],"accent_hints":[{"junction_index":1,"fx_category":"risers","beats":4.0},{"junction_index":1,"fx_category":"impacts","beats":1.0},{"junction_index":4,"fx_category":"snare_rolls","beats":4.0},{"junction_index":4,"fx_category":"impacts","beats":1.0}],"same_genre_tight_mix":false}

User: "build set into peak hour" with 7 transitions
Output:
{"arc":"build","text_prompt":"warmup into peak time","surprise_budget":2,"callback_budget":1,"transition_tiers":["minor","minor","major","minor","drop","major","cut"],"accent_hints":[{"junction_index":4,"fx_category":"risers","beats":8.0},{"junction_index":4,"fx_category":"impacts","beats":1.0}],"same_genre_tight_mix":false}
"""


SYSTEM_PROMPT_PHASE1 = """You are a club DJ planning a tightly-mixed offline set with the user's exact clip pool. Output ONLY valid JSON, no markdown, no commentary.

You will be shown the POOL INVENTORY (what clips are available, their source, genre, BPM, section, energy). Each clip is tagged USER (user-uploaded — these are STARS, must be heard) or LIB (library augmentation — supporting cast, use to bridge incompat keys, fill warmup/outro, or extend variety).

USER-VS-LIB POLICY:
- Every USER clip MUST appear in your plan at least once. They paid the upload tax.
- USER clips drive the narrative; LIB clips support it.
- Use LIB to: bridge BPM/genre gaps between USER clips, warmup/cooldown if USER pool too short, fill if USER pool can't sustain the duration.
- If pool is all USER: pure user mix. If pool is all LIB: just play coherent LIB picks.

Your job is to design a SET NARRATIVE that uses the pool intelligently. Every choice must serve the narrative.

WORKFLOW (think this way before emitting JSON):
  1. Read pool inventory. Identify clusters (techno block, chill block, etc).
  2. Pick a SET NARRATIVE — a 1-sentence story for the set.
     Examples: "warmup ambient → build to tech-house peak → cooldown lofi"
              "all-night techno hypnosis at 124 bpm"
              "disco surprise sandwiched in a deep-house build"
  3. Pick arc + tiers + accents that SERVE the narrative.
  4. Annotate each junction with INTENT: why are you using this transition here?

TRANSITION TIERS (Phase 1 — no cut, no loop):
- "minor": smooth EQ swap / crossfade over 8-16 bars. Workhorse, low drama.
- "major": structurally significant — filter_fade, drum_break, or echo_out. Use to mark NEW SECTION of the set narrative.
- "drop":  build_riser_drop. Engineer a climax. ONLY when both sides are drop-compatible (drop/hook/peak). NEVER over breakdown/intro/outro — that's an energy crater. If your pool has no drop-section clips, pick zero drop tiers.

JUNCTION INTENT (tag each junction with WHY):
- "breath"          : low-energy bridge between dense moments
- "build_tension"   : ramp up before something bigger (often before "drop_payoff")
- "drop_payoff"     : climax landing — pair with drop tier
- "genre_jump"      : intentional jump to new cluster — pair with major tier
- "callback"        : deliberate return to earlier vibe
- "smooth_continue" : keep groove rolling — pair with minor tier
- "cooldown"        : energy descend toward set end

CONSTRAINTS:
- arc: only "build", "peak", "flat_low".
- tier distribution by arc:
    build:    70% minor, 25% major, ≤1 drop near end (only if pool supports)
    peak:     50% minor, 35% major, 1-2 drops (only if pool supports)
    flat_low: all minor. No major. No drop.
- accents: cap 2 per junction. Use on major/drop tier only.
  Categories: risers, impacts, snare_rolls, sweeps, hihat_rolls, sub_drops
- Prefer NO accent over a forced one.
- If pool is too disparate (low coherence, no clusters), say so in `narrative_notes` and use mostly minor tiers — admit limitation rather than fake drama.

OUTPUT SCHEMA (EXACT keys, no extras):
{
  "arc": "build|peak|flat_low",
  "set_narrative": "<one-sentence story for the whole set>",
  "narrative_notes": "<optional: caveats about pool coherence, what you're working around>",
  "text_prompt": "<short vibe sentence echoing user request>",
  "surprise_budget": <int 0-10>,
  "callback_budget": <int 0-3>,
  "transition_tiers": ["minor","major","drop",...],
  "transition_intents": ["breath","build_tension","drop_payoff","genre_jump","callback","smooth_continue","cooldown",...],
  "accent_hints": [ {"junction_index": 0, "fx_category": "risers|impacts|snare_rolls|sweeps|hihat_rolls|sub_drops", "beats": 4.0} ],
  "same_genre_tight_mix": false
}

junction_index = 0 means between clip 1 and clip 2. transition_intents has same length as transition_tiers.

EXAMPLES (showing reasoning style):

Pool: 8 tech-house clips at 122-126 BPM (drop sections present), 2 ambient at 90 BPM
User: "warmup into peak"
Output:
{"arc":"build","set_narrative":"open with ambient breath, build through tech-house grooves, peak with a drop","narrative_notes":"pool has tight tech-house cluster + 2 ambient; using ambient as opener only","text_prompt":"warmup into peak time","surprise_budget":2,"callback_budget":0,"transition_tiers":["minor","minor","major","minor","major","drop"],"transition_intents":["breath","smooth_continue","genre_jump","smooth_continue","build_tension","drop_payoff"],"accent_hints":[{"junction_index":4,"fx_category":"risers","beats":8.0},{"junction_index":5,"fx_category":"impacts","beats":1.0}],"same_genre_tight_mix":true}

Pool: 12 disparate clips (cinematic, lofi, dnb, disco, future_bass, ambient) — coherence 0.4
User: "festival peak"
Output:
{"arc":"build","set_narrative":"navigate disparate pool as a curated journey rather than forcing peak","narrative_notes":"pool too disparate for festival-peak feel; mixing as warmup-style journey, no drops","text_prompt":"genre journey","surprise_budget":4,"callback_budget":0,"transition_tiers":["minor","major","minor","major","minor","minor"],"transition_intents":["breath","genre_jump","smooth_continue","genre_jump","smooth_continue","cooldown"],"accent_hints":[],"same_genre_tight_mix":false}

Pool: 5 lofi clips at 80-90 BPM
User: "after-hours"
Output:
{"arc":"flat_low","set_narrative":"sustained lo-fi ambient hypnosis","narrative_notes":"pool is tight lofi cluster — no major/drop needed, all minor blends","text_prompt":"after-hours lofi","surprise_budget":1,"callback_budget":0,"transition_tiers":["minor","minor","minor","minor"],"transition_intents":["breath","smooth_continue","smooth_continue","cooldown"],"accent_hints":[],"same_genre_tight_mix":true}
"""


def _system_prompt() -> str:
    """Phase-aware system prompt selector."""
    if os.environ.get("AIJOCKEY_PHASE", "1") == "1":
        return SYSTEM_PROMPT_PHASE1
    return SYSTEM_PROMPT_FULL


# Back-compat alias for code paths that still reference SYSTEM_PROMPT directly.
SYSTEM_PROMPT = SYSTEM_PROMPT_FULL


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
    allowed_arcs = _allowed_arcs()
    arc = arc_fallback if arc_fallback in allowed_arcs else "build"
    tiers = ["minor"] * max(1, max_transitions)
    if len(tiers) > 4 and len(tiers) // 8 > 0:
        tiers[len(tiers) // 4] = "major"
    intents = []
    for i, t in enumerate(tiers):
        if i == 0:
            intents.append("breath")
        elif i == len(tiers) - 1:
            intents.append("cooldown")
        elif t == "major":
            intents.append("genre_jump")
        else:
            intents.append("smooth_continue")
    return {
        "arc": arc,
        "text_prompt": user_prompt.strip() or "club mix, cohesive energy",
        "set_narrative": "deterministic fallback: minor blends with one mid-set major",
        "narrative_notes": "Director LLM unavailable; using rule-based plan",
        "transition_intents": intents,
        "surprise_budget": 10,
        "callback_budget": 1,
        "transition_tiers": tiers[:max_transitions],
        "accent_hints": [],
        "same_genre_tight_mix": False,
        "_fallback": True,
    }


def _sanitize_out(raw: dict[str, Any], arc_fallback: str, user_prompt: str,
                  max_transitions: int, coherence_hint: float | None) -> dict[str, Any]:
    allowed_arcs = _allowed_arcs()
    arc = raw.get("arc") or arc_fallback
    if isinstance(arc, str):
        arc = arc.lower().strip()
    if arc not in allowed_arcs:
        arc = arc_fallback if arc_fallback in allowed_arcs else allowed_arcs[0]

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

    allowed = _allowed_tiers()
    tiers_in = raw.get("transition_tiers")
    tiers: list[str] = []
    if isinstance(tiers_in, list):
        for x in tiers_in:
            tx = str(x).lower().strip()
            if tx in allowed:
                tiers.append(tx)
            elif tx in ALLOWED_TRANSITION_TIERS_FULL:
                # tier valid but disabled in current phase -> downgrade to major
                tiers.append("major")
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

    # Phase 1: cap accents at 2 per junction to avoid pile-on. This matches
    # the constitutional `accent_budget` rule and gives Director-level defense.
    max_per_junction = 2 if os.environ.get("AIJOCKEY_PHASE", "1") == "1" else 4
    by_junction: dict[int, list[dict[str, Any]]] = {}
    for a in accents:
        by_junction.setdefault(a["junction_index"], []).append(a)
    capped: list[dict[str, Any]] = []
    for ji in sorted(by_junction):
        capped.extend(by_junction[ji][:max_per_junction])
    accents = capped

    sg = raw.get("same_genre_tight_mix")
    if coherence_hint is not None and coherence_hint >= 0.72:
        sg = True
    elif not isinstance(sg, bool):
        sg = False

    # Narrative + per-junction intents (Phase A polish: "no character" fix).
    set_narrative = raw.get("set_narrative")
    if not isinstance(set_narrative, str) or not set_narrative.strip():
        set_narrative = f"sequence {len(tiers)} clips guided by '{text_prompt}'"
    narrative_notes = raw.get("narrative_notes") or ""
    if not isinstance(narrative_notes, str):
        narrative_notes = ""

    ALLOWED_INTENTS = {
        "breath", "build_tension", "drop_payoff", "genre_jump",
        "callback", "smooth_continue", "cooldown",
    }
    intents_in = raw.get("transition_intents") or []
    intents: list[str] = []
    if isinstance(intents_in, list):
        for x in intents_in:
            tx = str(x).lower().strip()
            intents.append(tx if tx in ALLOWED_INTENTS else "smooth_continue")
    while len(intents) < len(tiers):
        # Default intent infers from tier
        t = tiers[len(intents)]
        intents.append({"drop": "drop_payoff", "major": "genre_jump",
                        "minor": "smooth_continue"}.get(t, "smooth_continue"))
    intents = intents[:len(tiers)]

    return {
        "arc": arc,
        "text_prompt": text_prompt,
        "set_narrative": set_narrative.strip()[:240],
        "narrative_notes": narrative_notes.strip()[:240],
        "surprise_budget": surprise_budget,
        "callback_budget": callback_budget,
        "transition_tiers": tiers,
        "transition_intents": intents,
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
    clips_meta: dict | None = None,
) -> dict[str, Any]:
    """
    Returns sanitized director dict compatible with PlannerConfig + apply_llm_transition_tiers.
    max_transitions = max(clip_count_estimate - 1, min(estimated_timeline_slots, ...))

    If audio_clip_paths provided AND HF_DIRECTOR_MODEL contains 'Audio',
    a multimodal audio-aware Director (e.g. Qwen2-Audio) is used. The model
    actually hears each clip's first window before producing the JSON plan.
    """
    arcs = _allowed_arcs()
    arc_fb = arc_preset if arc_preset in arcs else arcs[0]
    if max_transitions_hint is not None:
        mt = max(1, min(64, max_transitions_hint))
    else:
        mt = estimate_max_transitions_for_pool(clip_count_estimate, approx_duration_seconds)

    use_llm = os.environ.get("AIJOCKEY_USE_DIRECTOR_LLM", "1").lower() not in (
        "0", "false", "no",
    )
    if use_llm:
        # Default Director model: audio-capable if audio paths provided AND
        # the env variable is unset, so /generate hears user clips out of
        # the box. Set HF_DIRECTOR_MODEL explicitly to override (e.g. to a
        # text-only Qwen3-Instruct when running on a CPU dev box).
        # Text default: Qwen3-8B-Instruct — better JSON adherence and
        # narrative reasoning than Qwen2.5-7B at similar VRAM cost.
        if audio_clip_paths:
            default_model = "Qwen/Qwen2-Audio-7B-Instruct"
        else:
            default_model = "Qwen/Qwen3-8B-Instruct"
        model_id = os.environ.get("HF_DIRECTOR_MODEL", default_model)
        is_audio_model = "audio" in model_id.lower() and audio_clip_paths
        # Pool intelligence: inject a clip-by-clip inventory so Director
        # can reason about WHAT the user uploaded, not just COUNT.
        pool_block = ""
        if clips_meta:
            try:
                from pool_intelligence import summary_table, diagnose
                pool_block = summary_table(clips_meta) + "\n"
                diag = diagnose(clips_meta)
                pool_block += (f"Pool diagnostic: verdict={diag['verdict']}, "
                               f"coherence={diag['coherence']}, "
                               f"genres={diag['n_genres']}, "
                               f"bpm_spread={diag['bpm_spread_pct']}%\n")
                pool_block += f"Narrative advice: {diag['narrative_advice']}\n\n"
            except Exception as e:
                pool_block = f"(pool inventory unavailable: {e})\n"
        # Style-RAG: prepend retrieved real DJ examples as taste anchors.
        rag_block = ""
        try:
            from style_rag import few_shot_block_for_director
            rag_block = few_shot_block_for_director(clips_meta=clips_meta)
        except Exception:
            rag_block = ""
        llm_prompt = (
            f"User DJ request:\n{user_prompt}\n\n"
            f"Suggested arc preset: {arc_fb}\n"
            f"Number of transitions to plan: {mt}\n\n"
            + pool_block
            + (rag_block if rag_block else "")
        )
        try:
            if is_audio_model:
                out_text = _call_qwen2audio(llm_prompt, audio_clip_paths, model_id)
            else:
                out_text = _call_hf_instruct(llm_prompt, model_id)
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
        # Prefer bf16 on bf16-capable GPUs (MI300X, A100, H100): same speed
        # as fp16 with wider exponent range — eliminates fp16 NaN risk in
        # long-context generation. Fall through to fp16 on older silicon.
        if torch.cuda.is_available():
            try:
                bf16_ok = torch.cuda.is_bf16_supported()
            except Exception:
                bf16_ok = False
            dtype = torch.bfloat16 if bf16_ok else torch.float16
        else:
            dtype = torch.float32
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
        return _call_hf_instruct(user_message, "Qwen/Qwen3-8B-Instruct")

    # Build conversation with one <audio> placeholder per clip
    user_content: list[dict] = []
    for i, _ in enumerate(audios):
        user_content.append({"type": "audio", "audio_url": f"clip_{i}"})
    user_content.append({"type": "text", "text": user_message})
    conversation = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_content},
    ]
    text = proc.apply_chat_template(conversation, add_generation_prompt=True,
                                    tokenize=False)
    inputs = proc(text=text, audios=audios, return_tensors="pt", padding=True,
                  sampling_rate=target_sr)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}
    with torch.inference_mode():
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
        if torch.cuda.is_available():
            try:
                bf16_ok = torch.cuda.is_bf16_supported()
            except Exception:
                bf16_ok = False
            _dtype = torch.bfloat16 if bf16_ok else torch.float16
        else:
            _dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            model = model.cpu()
        _LLM_CACHE = (tok, model, model_id)
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_message},
    ]
    prompt = ""
    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "System:\n" + SYSTEM_PROMPT + "\nUser:\n" + user_message + "\nAssistant:\n"
    # Pool inventory + system prompt + chat-template tags can exceed 2k
    # tokens. Truncating mid-template strips <|im_start|>assistant and
    # makes the LM continue user text instead of answering. Qwen2.5
    # supports 32k+ context — give the prompt headroom.
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=8192)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.inference_mode():
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
    """Realistic transition count given the planner picks segments around
    20-30s post-overlap. duration/20 is the upper estimate; capped at 16
    so Director does not exhaust LLM context with the pool inventory
    injected. Floor at 8 so short user prompts still get enough variety.
    """
    if duration_sec <= 0:
        return 8
    target = max(8, int(duration_sec / 20))
    return min(16, target)
