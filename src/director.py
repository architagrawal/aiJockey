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

TRANSITION TIERS (Phase 1 — no cut, no loop, full vocabulary available):
- "minor": smooth handoff. Vocab: crossfade, eq_swap, bass_swap, highs_swap,
           short_crossfade, long_crossfade, instrumental_swap. Workhorse.
- "major": structurally significant. Vocab: filter_fade, highpass_sweep_in,
           drum_break, kickless_swap, drum_replace, stem_swap, mashup,
           echo_out, reverb_wash, pitch_bend, spectral_hold, bpm_warp,
           harmonic_overlay. Use to mark NEW SECTION of the set narrative.
- "drop":  engineer a climax. Vocab: silence_drop, build_riser_drop (when
           pool has drop-compatible material), acapella_drop. ONLY when
           both sides are drop-compatible (drop/hook/peak/chorus). NEVER
           over breakdown/intro/outro — that's an energy crater. If pool
           has no drop-section clips, pick zero drop tiers.

VOCAL-SAFETY RULE (ENFORCED at execute layer):
   When EITHER junction side has vocal_activity > 0.30, AGGRESSIVE
   techniques are auto-rejected and downgraded to crossfade. Aggressive
   set: chop, tape_stop, drum_replace, kickless_swap, spinback,
   forward_spin, build_riser_drop, snare_buildup, scratch_fill,
   loop_tighten, loop_roll, beat_juggle, pitch_bend, bpm_warp,
   spectral_hold. These shred vocals — use them only on instrumental
   sections (label='inst' / 'break' / 'bridge' / 'solo').

   You ARE allowed to suggest aggressive techniques on instrumental
   sections — the pool has plenty of instrumental material, lean into it
   when the section vocab supports it. Don't be conservative when the
   material says go big.

VOCAL-FRIENDLY TECHNIQUES (safe even on chorus / verse / vocals-active):
   crossfade, short_crossfade, long_crossfade, eq_swap, bass_swap,
   highs_swap, frequency_blend, filter_fade, highpass_sweep_in,
   band_filter_sweep, stem_swap, mashup, instrumental_swap, echo_out,
   reverb_wash, harmonic_overlay, riser_overlay, impact_overlay.

   These can run over vocal-active sections without warping the vocal.

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
  Categories: risers, impacts, snare_rolls, sweeps, hihat_rolls, sub_drops,
              airhorns (festival meme), vinyl (vinyl-stop / spinback meme)
- Prefer NO accent over a forced one.
- If pool is too disparate (low coherence, no clusters), say so in `narrative_notes` and use MOSTLY minor tiers (>=60%) — but ALWAYS allow at least 1-2 `major` picks per 10 junctions to mark intentional genre jumps. ALL-MINOR is FORBIDDEN — it produces a flat, boring mix the user notices immediately. Even a "journey" set needs occasional sectional markers.
- HARD RULE on incompatible bridges: NEVER use `drop` or `major` tier for a junction whose two clips differ in genre cluster AND have BPM gap > 8 BPM. That combination produces audible energy mismatch + phase cancellation (verified: probe severity ≥ 0.85 on every such forced junction). When forced into such a bridge, use `minor` with intent `breath` or `cooldown` — let the listener absorb the cluster shift, don't paper it over with a drop.
- TIER DIVERSITY MINIMUM: out of N junctions, at most ceil(N * 0.75) may be `minor`. Remaining must be `major` (or `drop` when pool truly supports it). Pick the major picks at the most BPM/key/energy-compatible junctions. Even on disparate pools, find the 2-3 junctions where a major fits and mark them — don't blanket-minor.
- VOCAL-AWARE TIER PLACEMENT: prefer placing `major`/`drop` tiers at junctions where BOTH adjacent clips have an INSTRUMENTAL section available (label='inst' / 'break' / 'bridge' / 'solo' / 'breakdown'). The execute layer's vocal_guard would clamp aggressive techniques on vocal-active junctions anyway — wasting the tier slot. Pick vocal-light junctions for major/drop.

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
  "accent_hints": [ {"junction_index": 0, "fx_category": "risers|impacts|snare_rolls|sweeps|hihat_rolls|sub_drops|airhorns|vinyl", "beats": 4.0} ],
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

Pool: 4 vocal pop (Despacito/Taki/Stars/Waka) + 4 instrumental electronic (cinematic, electrodoodle)
User: "vocal-forward journey, varied transitions welcome"
Reasoning: pool mixes vocal pop with instrumental cinematic. Use instrumental
clips as build-up + drop-payoff slots (their drop sections won't fight vocals).
Multi-peak structure: open mellow → first peak (drop into vocal hook) → valley
→ bigger second peak (instrumental drop) → callback to opener vibe.
Output:
{"arc":"build","set_narrative":"layered vocal-pop journey with two peak moments anchored by instrumental drops","narrative_notes":"pool supports build-drop-recovery-bigger-drop. Reserve drop tier for inst→inst junctions where vocal_guard won't clamp.","text_prompt":"vocal-forward festival journey","surprise_budget":3,"callback_budget":2,"transition_tiers":["minor","major","minor","drop","major","minor","drop","major","minor","minor","major"],"transition_intents":["breath","genre_jump","smooth_continue","drop_payoff","build_tension","callback","drop_payoff","genre_jump","callback","smooth_continue","cooldown"],"accent_hints":[{"junction_index":2,"fx_category":"snare_rolls","beats":4.0},{"junction_index":3,"fx_category":"impacts","beats":1.0},{"junction_index":5,"fx_category":"risers","beats":8.0},{"junction_index":6,"fx_category":"impacts","beats":1.0}],"same_genre_tight_mix":false}

NARRATIVE-DRIVEN STRUCTURE (Tomorrowland-grade per docs/dj_research.md):
A real DJ set isn't a flat parade of crossfades. It has SHAPE:
  • Opener: low energy, room-reading. (1-2 minor + breath/smooth_continue)
  • First climb: tension building (1-2 major with build_tension intent)
  • First peak: drop or major-into-hook (1 drop OR major with drop_payoff)
  • Recovery valley: breakdown / breath section (1-2 minor with breath)
  • Bigger climb: more aggressive techniques (loop_tighten, drum_break)
  • Main peak: drop tier with riser+impact accents (the festival moment)
  • Callbacks during recovery: revisit earlier clip (callback intent)
  • Outro: cooldown back to minor (gentle exit)
ENERGY ARC IS NOT LINEAR. Use valleys deliberately to make peaks feel bigger.

WHEN TO PICK DROP TIER (the climax moment):
  • Pool has at least 1 clip with section type 'drop' / 'chorus' / 'hook'
  • Junction lands at 50-80% through the set (peak zone)
  • Both adjacent clips are instrumental-friendly (vocal_guard won't clamp)
  • Pair with riser accent BEFORE + impact accent ON the drop
PICK 0 DROPS only if pool is genuinely flat (lofi, ambient, all-minor genre).
For build/peak arcs with electronic material, 1-2 drops is the norm.

WHEN TO USE SURPRISE BUDGET:
  surprise_budget=N means N intentional surprises (genre_jump on incompatible
  pair, callback to long-ago clip, drop tier on unexpected junction). DON'T
  emit surprise_budget=0 unless the pool is truly homogeneous — wasted
  opportunity to add character. For 10+ junction sets, surprise_budget 2-4
  is the sweet spot.
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

    # ENFORCE TIER DIVERSITY (post-LLM safety net):
    # Director's LLM is biased toward all-minor for "disparate" pools — even
    # though the prompt says <=75% minor + always 1-2 majors. Enforce here.
    # Disable with AIJOCKEY_DIRECTOR_TIER_ENFORCE=0.
    if os.environ.get("AIJOCKEY_DIRECTOR_TIER_ENFORCE", "1") != "0" and tiers:
        import math as _math
        n = len(tiers)
        max_minor_pct = float(os.environ.get("AIJOCKEY_DIRECTOR_MAX_MINOR_PCT", "0.75"))
        max_minor = int(_math.ceil(n * max_minor_pct))
        cur_minor = sum(1 for t in tiers if t == "minor")
        if cur_minor > max_minor and n >= 4:
            # Need to demote (cur_minor - max_minor) entries from minor → major.
            # Prefer interior junctions (skip first + last, which are
            # naturally fade_in / cooldown). Pick evenly-spaced positions
            # so majors are spread across the set, not clumped.
            n_to_promote = cur_minor - max_minor
            interior_minor_idx = [i for i, t in enumerate(tiers)
                                  if t == "minor" and 0 < i < n - 1]
            if interior_minor_idx:
                # Even spacing within available indices
                step = max(1, len(interior_minor_idx) // n_to_promote)
                pick = interior_minor_idx[::step][:n_to_promote]
                for idx in pick:
                    tiers[idx] = "major"
                print(f"[director] tier-diversity enforcement: promoted "
                      f"{len(pick)} minors → major at indices {pick} "
                      f"(was {cur_minor}/{n} minor, max_allowed {max_minor})")

        # FORCE_DROP retired — user feedback: don't FORCE effects, encourage
        # naturally via prompt examples. Disabled by default; flip
        # AIJOCKEY_DIRECTOR_FORCE_DROP=1 to re-enable for stress tests.
        if (os.environ.get("AIJOCKEY_DIRECTOR_FORCE_DROP", "0") == "1"
                and arc in ("build", "peak") and n >= 6
                and not any(t == "drop" for t in tiers)):
            zone_lo = max(1, int(0.55 * n)); zone_hi = min(n - 1, int(0.80 * n))
            for j in range(zone_hi, zone_lo - 1, -1):
                if tiers[j] in ("major", "minor"):
                    tiers[j] = "drop"; break

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

    # AUTO_ACCENT retired — user feedback: don't FORCE injection. Director
    # picks accents organically when prompt examples model the behavior.
    # Default off; flip AIJOCKEY_DIRECTOR_AUTO_ACCENT=1 for stress test.
    if (os.environ.get("AIJOCKEY_DIRECTOR_AUTO_ACCENT", "0") == "1"
            and not accents and tiers):
        for j_idx, t in enumerate(tiers):
            if t == "drop":
                accents.append({"junction_index": j_idx, "fx_category": "risers", "beats": 8.0})
                accents.append({"junction_index": j_idx, "fx_category": "impacts", "beats": 1.0})
            elif t == "major":
                accents.append({"junction_index": j_idx, "fx_category": "snare_rolls", "beats": 4.0})

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
    # Re-sync intents to tiers in case tier enforcement promoted entries:
    # if a junction is now `major` but its intent says `smooth_continue`,
    # upgrade intent to `genre_jump` so planner sees coherent signal.
    for i, (t, it) in enumerate(zip(tiers, intents)):
        if t == "major" and it in ("smooth_continue", "breath", "cooldown"):
            intents[i] = "genre_jump"
        elif t == "drop" and it != "drop_payoff":
            intents[i] = "drop_payoff"

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


def _score_plan_with_clap(plan: dict, arc_fb: str,
                             cache_dir: str | None = None) -> float:
    """Heuristic score + cross-clip CLAP coherence bonus when enabled."""
    base = _score_plan(plan, arc_fb)
    try:
        from clap_coherence import enabled as _cc_en, coherence_score
        if _cc_en() and cache_dir:
            cids = plan.get("clip_sequence") or []
            if cids:
                sim = coherence_score(cids, cache_dir)
                # sim in [-1, 1] → bonus up to ±0.6
                base += 0.6 * float(sim)
    except Exception:
        pass
    return base


def _score_plan(plan: dict, arc_fb: str) -> float:
    """Pre-render heuristic to compare candidate Director plans.

    Without rendering each (expensive), score on structural signals the
    planner+execute layers already exploit: tier variety, drop presence,
    callback budget, arc-preset compliance, surprise budget. Higher is
    better. Cheap, deterministic.
    """
    tiers = plan.get('transition_tiers') or []
    n = len(tiers) or 1
    uniq = len(set(tiers))
    has_major = any(t == 'major' for t in tiers)
    has_drop = any(t == 'drop' for t in tiers)
    pct_minor = sum(1 for t in tiers if t == 'minor') / n
    arc_match = 1.0 if plan.get('arc') == arc_fb else 0.0
    cb = int(plan.get('callback_budget') or 0)
    sb = int(plan.get('surprise_budget') or 0)
    return (
        1.5 * (uniq / 4.0)            # 4 tier classes max in phase 1
        + 1.0 * float(has_major)
        + 0.7 * float(has_drop)
        + 0.6 * max(0.0, 1.0 - pct_minor)
        + 0.5 * arc_match
        + 0.4 * min(1.0, cb / 3.0)
        + 0.3 * min(1.0, sb / 6.0)
    )


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
        # text-only Qwen-Instruct when running on a CPU dev box, or to a
        # Qwen3 family ID once you've validated availability).
        if audio_clip_paths:
            default_model = "Qwen/Qwen2-Audio-7B-Instruct"
        else:
            # Qwen3-8B-Instruct does not exist on HF (verified 401). Real
            # Qwen3 text-instruct models: Qwen3-4B-Instruct-2507 (small)
            # or Qwen3-235B-A22B-Instruct-2507 (MoE, needs QLoRA on
            # 192GB VRAM). Default = Qwen2.5-7B-Instruct (proven, fast).
            # Set HF_DIRECTOR_MODEL=Qwen/Qwen3-4B-Instruct-2507 to test
            # newer family.
            default_model = "Qwen/Qwen2.5-7B-Instruct"
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
        # Multi-Director sampling: draw N plans at temperature>0, score with
        # _score_plan, return best. N=1 keeps legacy greedy-decode behavior.
        try:
            n_samples = max(1, int(os.environ.get('AIJOCKEY_DIRECTOR_N_SAMPLES', '1')))
        except Exception:
            n_samples = 1
        try:
            temperature = float(os.environ.get('AIJOCKEY_DIRECTOR_TEMPERATURE', '0.7'))
        except Exception:
            temperature = 0.7
        try:
            best_plan = None
            best_score = float('-inf')
            collected_plans: list[dict] = []
            for s_i in range(n_samples):
                do_sample = n_samples > 1 or temperature > 0 and s_i > 0
                # Confidence-aware early-exit: if best plan so far scores
                # well above HIGH_CONF_THRESH, stop sampling. Saves GPU.
                _conf_thr = float(os.environ.get(
                    "AIJOCKEY_DIRECTOR_HIGH_CONF_THRESH", "4.0"))
                if (best_plan is not None and best_score >= _conf_thr
                        and s_i >= 1 and os.environ.get(
                            "AIJOCKEY_DIRECTOR_CONF_EARLY_EXIT", "1") == "1"):
                    print(f"[director] early-exit at sample {s_i} "
                          f"(score={best_score:.2f} >= {_conf_thr})")
                    break
                if is_audio_model:
                    # Qwen2-Audio path: keep greedy on first draw, sample on extras.
                    out_text = _call_qwen2audio(llm_prompt, audio_clip_paths, model_id)
                else:
                    out_text = _call_hf_instruct(
                        llm_prompt, model_id,
                        do_sample=(n_samples > 1),
                        temperature=temperature,
                    )
                parsed = _extract_json_object(out_text or "")
                if not parsed:
                    continue
                cand_plan = _sanitize_out(parsed, arc_fb, user_prompt, mt, coherence_hint)
                collected_plans.append(cand_plan)
                sc = _score_plan(cand_plan, arc_fb)
                if sc > best_score:
                    best_score = sc
                    best_plan = cand_plan
                if n_samples > 1:
                    print(f"[director] sample {s_i+1}/{n_samples} score={sc:.2f}")
            # CLAP-text plan rerank — overrides best_plan if its choice
            # aligns better with user_prompt semantics.
            try:
                from clap_director_rerank import (
                    enabled as _clap_en,
                    rerank_plans as _clap_rerank,
                )
                if _clap_en() and collected_plans:
                    picked = _clap_rerank(collected_plans, user_prompt)
                    if picked is not None:
                        best_plan = picked
                        print("[director] CLAP plan rerank picked alternative")
            except Exception:
                pass
            # MERT-conditioned plan rerank — picks plan whose clip set
            # has highest pre-computed MERT-predicted PQ/CE.
            try:
                from mert_plan_rerank import (
                    enabled as _mp_en,
                    rerank_plans as _mp_rerank,
                )
                if _mp_en() and collected_plans and clips_meta:
                    cache_dir = os.environ.get(
                        "AIJOCKEY_LIBRARY_CACHE", "/cache")
                    pool_ids = list(clips_meta.keys())
                    picked = _mp_rerank(collected_plans, pool_ids, cache_dir)
                    if picked is not None:
                        best_plan = picked
                        print("[director] MERT plan rerank picked alternative")
            except Exception:
                pass
            # Self-critique loop — Director evaluates and revises own plan.
            try:
                from director_critique import (
                    enabled as _crit_en,
                    critique_and_revise as _crit_rev,
                )
                if _crit_en() and best_plan is not None:
                    def _crit_llm(msg: str) -> str:
                        if is_audio_model:
                            return _call_qwen2audio(msg, audio_clip_paths,
                                                      model_id)
                        return _call_hf_instruct(msg, model_id,
                                                  do_sample=True,
                                                  temperature=0.5)
                    revised = _crit_rev(
                        best_plan, _crit_llm,
                        max_iters=int(os.environ.get(
                            "AIJOCKEY_DIRECTOR_CRITIQUE_ITERS", "2")),
                        score_fn=lambda p: _score_plan(p, arc_fb),
                    )
                    if revised is not None:
                        best_plan = revised
                        # Re-sanitize after revision
                        best_plan = _sanitize_out(
                            best_plan, arc_fb, user_prompt, mt, coherence_hint)
                        print("[director] self-critique revised plan")
            except Exception as _ce:
                print(f"[director] critique skip: {_ce}")
            if best_plan is not None:
                if n_samples > 1:
                    best_plan = dict(best_plan)
                    best_plan['_sampling_score'] = round(best_score, 3)
                    best_plan['_n_samples'] = n_samples
                return best_plan
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
        # No audio loaded — fall back to text path. Match text Director
        # default (Qwen3-8B-Instruct doesn't exist on HF; commit f68b62b).
        return _call_hf_instruct(user_message, "Qwen/Qwen2.5-7B-Instruct")

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


def _call_hf_instruct(user_message: str, model_id: str,
                       *, do_sample: bool = False,
                       temperature: float = 0.7,
                       top_p: float = 0.9) -> str:
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
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
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
