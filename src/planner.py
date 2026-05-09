"""
Set planner — beam-search subset selection + non-sequential ordering.

Inputs: pool of analyzed clips (cache/*.json + cache/*.npz).
Output: timeline.json — ordered list of TimelineEntry with technique per junction.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
from copy import deepcopy
from dataclasses import dataclass, field, asdict
import numpy as np

from camelot import camelot_distance
from style_rag import StyleRAG


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimelineEntry:
    clip_id: str
    segment: dict                     # {start, end, type, energy}
    target_bpm: float
    target_key: str                   # Camelot
    transition_in: dict               # {name, bars, ...}
    play_at: float = 0.0


@dataclass
class PlannerConfig:
    target_duration: float = 1800.0           # 30 min default
    # Energy arc = the planner's PLAN. AI decides shape based on intent.
    # Presets:
    #   'build':     low → peak → cooldown (default warm-up set)
    #   'peak':      start high, sustain, slight cooldown
    #   'rollercoaster': oscillating peaks/valleys
    #   'descend':   cooldown set
    # Override via energy_arc (list of floats) directly.
    energy_arc: list[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.7, 0.9, 1.0, 0.95, 0.85, 0.6, 0.4, 0.3])
    arc_shape: str = 'build'                  # 'build' | 'peak' | 'rollercoaster' | 'descend' | 'custom'
    surprise_budget: int = 10                 # generous: long sets need surprises
    callback_budget: int = 2
    beam_width: int = 16
    max_clips: int = 200                      # large cap for 30-min mixes
    min_clips: int = 1                        # min total entries (allows callbacks)
    min_unique_clips: int = 5                 # min distinct clips that must appear
    allow_clip_reuse: bool = True             # reuse clips when pool exhausted
    clip_reuse_cooldown: int = 5              # min entries between reuses of same clip
    min_segment_seconds: float = 30.0         # skip sections shorter than this
    weights: dict = field(default_factory=lambda: dict(
        key=0.25, tempo=0.20, energy=0.20, timbre=0.15,
        variety=0.10, surprise=0.10,
    ))
    style_rag_dir: str | None = None
    style_rag_top_k: int = 5
    style_rag_bias_weight: float = 0.15
    text_prompt: str | None = None            # natural-language mix description;
                                              # CLAP-cosine-biases clip selection
    text_prompt_weight: float = 1.5           # strong bias — overrides genre/timbre defaults
    classifier_ckpt: str | None = None        # technique classifier ckpt
    compat_head_ckpt: str | None = None       # CLAP compat head ckpt
                                              # (Tier 1.5 — projects CLAP into
                                              #  DJ-compatibility space)
    restricted: bool = True                   # demo-safe technique whitelist
                                              # + multi-genre BPM/key filter


@dataclass
class State:
    sequence: list[TimelineEntry]
    cumulative_duration: float = 0.0
    used_clip_ids: set[str] = field(default_factory=set)
    surprises_used: int = 0
    callbacks_used: int = 0
    score: float = 0.0
    # For reuse: track recent clip uses (cooldown) and per-clip segment usage
    recent_clip_ids: list[str] = field(default_factory=list)
    used_segments: dict = field(default_factory=dict)  # clip_id -> set of segment indices


# ---------------------------------------------------------------------------
# Cache loader
# ---------------------------------------------------------------------------

def load_clips(cache_dir: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for jp in sorted(Path(cache_dir).glob('*.json')):
        with open(jp) as f:
            d = json.load(f)
        npz_path = Path(cache_dir) / f"{jp.stem}.npz"
        if npz_path.exists():
            npz = np.load(str(npz_path))
            d['clap'] = npz['clap']
            d['energy_arr'] = npz['energy']
        else:
            d['clap'] = np.zeros(512, dtype=np.float32)
            d['energy_arr'] = np.zeros(0, dtype=np.float32)
        # Normalize section energies per-clip to 0..1 (raw librosa RMS is
        # often ~0.05-0.3; arc presets are 0..1 — without rescaling, arc
        # comparisons saturate). Each clip's loudest section becomes 1.0.
        sections = d.get('sections', [])
        if sections:
            energies = [s.get('energy', 0.0) for s in sections]
            peak = max(energies) if energies else 0.0
            if peak > 0:
                for s in sections:
                    raw = s.get('energy', 0.0)
                    s['energy_raw'] = raw            # keep original
                    s['energy'] = raw / peak         # 0..1 normalized
        out[jp.stem] = d
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _normalize_clap(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def transition_score(prev_clip: dict, prev_seg: dict, prev_target_bpm: float,
                     prev_target_key: str, cand_clip: dict, cand_seg: dict,
                     weights: dict,
                     style_rag: StyleRAG | None = None,
                     rag_top_k: int = 5,
                     rag_bias_weight: float = 0.15,
                     classifier_ckpt: str | None = None) -> tuple[float, dict, bool]:
    """Returns (score, technique_dict, is_surprise)."""
    cand_bpm = cand_clip.get('tempo', prev_target_bpm)
    cand_bpm = cand_bpm if cand_bpm > 0 else prev_target_bpm
    stretch = abs(prev_target_bpm - cand_bpm) / max(cand_bpm, 1.0)
    tempo_score = max(0.0, 1.0 - stretch / 0.12)

    key_dist = camelot_distance(prev_target_key, cand_clip.get('key', '?'))
    key_score = max(0.0, 1.0 - key_dist / 4.0)

    out_e = float(prev_seg.get('energy', 0.5))
    in_e = float(cand_seg.get('energy', 0.5))
    energy_score = 1.0 - min(1.0, abs(out_e - in_e) * 2.0)

    a_raw = np.asarray(prev_clip.get('clap', np.zeros(512)), dtype=np.float32)
    b_raw = np.asarray(cand_clip.get('clap', np.zeros(512)), dtype=np.float32)
    # Use 'compat' projection if pre-computed (set by plan() when ckpt provided),
    # else fall back to raw normalized CLAP cosine.
    a = _normalize_clap(prev_clip.get('compat', a_raw))
    b = _normalize_clap(cand_clip.get('compat', b_raw))
    timbre_score = float(a @ b) if a.size == b.size and a.size > 0 else 0.0
    variety_score = 1.0 - timbre_score if timbre_score > 0.95 else 1.0

    score = (weights['key'] * key_score
             + weights['tempo'] * tempo_score
             + weights['energy'] * energy_score
             + weights['timbre'] * timbre_score
             + weights['variety'] * variety_score)

    # Technique selection — priority order
    if key_score < 0.3 and tempo_score < 0.4:
        tech = {'name': 'echo_out', 'bars': 8, 'delay_beats': 0.5, 'feedback': 0.55}
    elif 0.3 <= key_score < 0.6 and key_dist <= 3:
        tech = {'name': 'pitch_bend', 'bars': 8, 'semitones': 1.0}
    elif key_score < 0.4:
        tech = {'name': 'drum_break', 'bars': 8}
    elif out_e > 0.85 and in_e > 0.85 and timbre_score < 0.5:
        tech = {'name': 'spinback', 'spinback_beats': 4}
    elif in_e > 0.85 and out_e > 0.6:
        tech = {'name': 'loop_tighten', 'start_bars': 4}
    elif in_e > 0.85 and out_e < 0.5:
        tech = {'name': 'silence_drop', 'silence_beats': 2}
    elif timbre_score > 0.7 and in_e > 0.5 and out_e > 0.5:
        tech = {'name': 'mashup', 'bars': 16}
    elif in_e < out_e - 0.2:
        tech = {'name': 'filter_fade', 'bars': 16}
    elif energy_score > 0.7 and tempo_score > 0.6 and in_e > 0.7:
        tech = {'name': 'eq_swap', 'bars': 32}
    elif energy_score > 0.5:
        tech = {'name': 'crossfade', 'bars': 16}
    else:
        tech = {'name': 'eq_swap', 'bars': 16}

    # Trained classifier overrides decision tree if available.
    if classifier_ckpt:
        try:
            from training.integrate import pick_technique
            tech = pick_technique(prev_clip, prev_seg, cand_clip, cand_seg,
                                  ckpt_path=classifier_ckpt,
                                  default_bars=tech.get('bars', 16))
        except Exception as e:
            print(f"warn: classifier pick failed ({e}), using rule tree")

    # Style-RAG bias: query reference patterns, bonus for techniques used in
    # similar transition contexts. Optional.
    if style_rag is not None and len(style_rag) > 0:
        retrieved = style_rag.query(
            out_clap=a, in_clap=b,
            out_energy=out_e, in_energy=in_e,
            top_k=rag_top_k,
        )
        bias = style_rag.technique_bias(retrieved)
        if tech['name'] in bias:
            score += rag_bias_weight * bias[tech['name']]
        # If RAG strongly suggests a different technique, switch
        if bias:
            top_tech, top_freq = max(bias.items(), key=lambda kv: kv[1])
            if top_freq >= 0.6 and top_tech != tech['name']:
                # Strong consensus from refs — adopt their choice (keep bars)
                tech = {'name': top_tech, 'bars': tech.get('bars', 16)}
                score += rag_bias_weight * top_freq

    is_surprise = score < 0.4
    return score, tech, is_surprise


ARC_PRESETS = {
    'build':         [0.3, 0.5, 0.7, 0.9, 1.0, 0.95, 0.85, 0.6, 0.4, 0.3],
    'peak':          [0.85, 0.95, 1.0, 1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6],
    'rollercoaster': [0.4, 0.85, 0.55, 0.95, 0.5, 1.0, 0.6, 0.85, 0.4, 0.3],
    'descend':       [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3],
    'flat_high':     [0.85, 0.9, 0.9, 0.95, 0.95, 0.9, 0.9, 0.85, 0.85, 0.8],
    'flat_low':      [0.35, 0.4, 0.4, 0.45, 0.45, 0.4, 0.4, 0.35, 0.35, 0.3],
}


def resolve_arc(shape: str, custom: list[float] | None = None) -> list[float]:
    if custom and shape == 'custom':
        return list(custom)
    return ARC_PRESETS.get(shape, ARC_PRESETS['build'])


def _arc_slope_at(arc: list[float], progress: float) -> float:
    """
    Slope of target energy arc at given normalized progress (0..1).
    Returns delta to next arc point. Used to match section energy gradient
    against where we expect the mix's energy to be heading.
    """
    if not arc or len(arc) < 2:
        return 0.0
    n = len(arc)
    idx = min(int(progress * (n - 1)), n - 2)
    return float(arc[idx + 1] - arc[idx])


def _apply_restricted_filter(tech: dict) -> dict:
    """Replace artifact-prone techniques with safe fallbacks for demo output."""
    try:
        from restricted_mode import filter_technique
        return filter_technique(tech, restricted=True)
    except Exception:
        return tech


# ---------------------------------------------------------------------------
# Segment selection
# ---------------------------------------------------------------------------

def pick_segment(clip: dict, prefer: str | None = None,
                 target_energy: float | None = None,
                 exclude_indices: set | None = None,
                 min_seconds: float = 30.0) -> tuple[dict, int]:
    """
    Returns (segment_dict, segment_index). Filters sections shorter than
    min_seconds. Falls back to longest available if all too short.
    """
    sections = clip.get('sections', [])
    if not sections:
        return ({'start': 0.0, 'end': clip.get('duration', 30.0),
                 'type': 'unknown', 'energy': 0.5}, -1)
    exclude = exclude_indices or set()
    # Filter sections by min duration
    long_enough = [(i, s) for i, s in enumerate(sections)
                   if (s['end'] - s['start']) >= min_seconds]
    available = [(i, s) for i, s in long_enough if i not in exclude]
    if not available:
        # No long enough remaining — fall back to longest section regardless
        if long_enough:
            available = long_enough
        else:
            available = list(enumerate(sections))  # last resort, allow short
    if prefer:
        for i, s in available:
            if s.get('type') == prefer:
                return (dict(s), i)
    if target_energy is not None:
        i, s = min(available, key=lambda kv: abs(kv[1].get('energy', 0.5) - target_energy))
        return (dict(s), i)
    body = [(i, s) for i, s in available if s.get('type') in ('drop', 'verse', 'breakdown')]
    pool = body or available
    i, s = max(pool, key=lambda kv: kv[1]['end'] - kv[1]['start'])
    return (dict(s), i)


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

def plan(clips: dict[str, dict], config: PlannerConfig) -> list[dict]:
    """Returns timeline as list of dicts (asdict of TimelineEntry)."""
    if not clips:
        return []

    # Resolve arc shape (preset name -> values). 'custom' uses config.energy_arc as-is.
    if config.arc_shape != 'custom':
        config.energy_arc = resolve_arc(config.arc_shape)
    print(f"arc shape: {config.arc_shape}, "
          f"start={config.energy_arc[0]:.2f}, peak={max(config.energy_arc):.2f}, "
          f"end={config.energy_arc[-1]:.2f}")

    # Optional Style-RAG index
    rag = None
    if config.style_rag_dir:
        rag = StyleRAG(config.style_rag_dir)
        if len(rag) > 0:
            print(f"Style-RAG: loaded {len(rag)} reference patterns")
        else:
            print(f"Style-RAG: no patterns found in {config.style_rag_dir}, skipping bias")
            rag = None

    # Optional text prompt — embed once, used to bias clip selection
    text_emb_norm = None
    if config.text_prompt:
        try:
            import numpy as _np
            from clap_wrapper import get_text_embedding
            t_emb = get_text_embedding(config.text_prompt)[0].astype(_np.float32)
            n = float(_np.linalg.norm(t_emb))
            text_emb_norm = t_emb / n if n > 0 else t_emb
            print(f"text prompt: '{config.text_prompt[:60]}' "
                  f"(weight={config.text_prompt_weight})")
        except Exception as e:
            print(f"warn: text prompt embedding failed ({e}), ignoring")

    # Optional CLAP compat head (Tier 1.5) — pre-project all clip CLAPs once
    if config.compat_head_ckpt:
        try:
            import numpy as _np
            from training.clap_finetune import load_compat_head, project_batch
            head = load_compat_head(config.compat_head_ckpt)
            ids = list(clips.keys())
            claps = _np.stack([clips[c]['clap'] for c in ids]).astype(_np.float32)
            projected = project_batch(head, claps)
            for cid, p in zip(ids, projected):
                clips[cid]['compat'] = p
            print(f"CLAP compat head: projected {len(ids)} clips into "
                  f"{projected.shape[1]}-dim DJ-compat space")
        except Exception as e:
            print(f"warn: compat head load/project failed ({e}), using raw CLAP")

    # Opener selection: pick section + clip matching arc[0] (the planner's
    # decision about where the mix STARTS). Low arc[0] -> warmup-style opener.
    # High arc[0] -> banger-from-the-jump opener. Arc shape is configurable;
    # not assumed to start low.
    target_e_open = config.energy_arc[0] if config.energy_arc else 0.3
    starts: list[State] = []
    for cid, clip in clips.items():
        seg, seg_idx = pick_segment(
            clip, target_energy=target_e_open,
            min_seconds=config.min_segment_seconds,
        )
        target_bpm = clip.get('tempo', 128.0) or 128.0
        entry = TimelineEntry(
            clip_id=cid, segment=seg, target_bpm=target_bpm,
            target_key=clip.get('key', '?'),
            transition_in={'name': 'fade_in', 'bars': 4},
        )
        used_segs = {cid: {seg_idx}} if seg_idx >= 0 else {}
        # Opener score: energy-match-to-arc-start + prompt cosine
        seg_e = float(seg.get('energy', 0.5))
        energy_match = 1.0 - min(1.0, abs(seg_e - target_e_open) * 2.0)
        opener_score = 0.30 * energy_match
        if text_emb_norm is not None:
            cand_clap = np.asarray(clip.get('clap', np.zeros(512)), dtype=np.float32)
            n_clap = float(np.linalg.norm(cand_clap))
            if n_clap > 0:
                opener_score += config.text_prompt_weight * float(
                    (cand_clap / n_clap) @ text_emb_norm)
        starts.append(State(
            sequence=[entry],
            cumulative_duration=seg['end'] - seg['start'],
            used_clip_ids={cid},
            recent_clip_ids=[cid],
            used_segments=used_segs,
            score=opener_score,
        ))

    beam = sorted(starts, key=lambda s: -s.score)[:config.beam_width]
    finished: list[State] = []

    while beam:
        next_beam: list[State] = []
        for st in beam:
            n_unique = len(st.used_clip_ids)
            duration_met = st.cumulative_duration >= config.target_duration
            cap_hit = len(st.sequence) >= config.max_clips
            if duration_met or cap_hit:
                # Only finish if min_unique_clips satisfied (or max_clips cap hit)
                if (n_unique >= config.min_unique_clips
                        and len(st.sequence) >= config.min_clips):
                    finished.append(st)
                    continue
                # Duration met but not enough unique clips — keep extending
                # (loop will try to add more clips)
                if cap_hit:
                    if len(st.sequence) >= config.min_clips:
                        finished.append(st)
                    continue
                # else: fall through to expansion below
            progress = st.cumulative_duration / config.target_duration
            arc_idx = min(int(progress * len(config.energy_arc)),
                          len(config.energy_arc) - 1)
            target_e = config.energy_arc[arc_idx]
            last = st.sequence[-1]
            last_clip = clips[last.clip_id]

            # Build candidate pool. Prefer unused clips. Allow reused with cooldown
            # if all unused exhausted OR explicitly enabled.
            unused = [cid for cid in clips if cid not in st.used_clip_ids]
            cooldown = set(st.recent_clip_ids[-config.clip_reuse_cooldown:])
            reuse = ([cid for cid in clips
                      if cid in st.used_clip_ids and cid not in cooldown]
                     if config.allow_clip_reuse else [])
            candidate_ids = unused or reuse  # prefer fresh clips

            scored_candidates: list[tuple[float, dict, bool, str, dict, int]] = []
            for cid in candidate_ids:
                cand = clips[cid]
                seg, seg_idx = pick_segment(
                    cand, target_energy=target_e,
                    exclude_indices=st.used_segments.get(cid, set()),
                    min_seconds=config.min_segment_seconds,
                )
                score, tech, is_surprise = transition_score(
                    last_clip, last.segment, last.target_bpm, last.target_key,
                    cand, seg, config.weights,
                    style_rag=rag,
                    rag_top_k=config.style_rag_top_k,
                    rag_bias_weight=config.style_rag_bias_weight,
                    classifier_ckpt=config.classifier_ckpt,
                )
                # Text-prompt bias (Option A): cosine of candidate's CLAP vs prompt
                if text_emb_norm is not None:
                    cand_clap = np.asarray(cand.get('clap', np.zeros(512)),
                                           dtype=np.float32)
                    n = float(np.linalg.norm(cand_clap))
                    if n > 0:
                        cand_norm = cand_clap / n
                        prompt_match = float(cand_norm @ text_emb_norm)
                        score += config.text_prompt_weight * prompt_match
                # Phrase-aware energy slope match (Option C):
                target_slope = _arc_slope_at(config.energy_arc, progress)
                section_slope = (seg.get('energy', 0.5)
                                 - last.segment.get('energy', 0.5))
                slope_diff = abs(target_slope - section_slope)
                slope_match = max(0.0, 1.0 - slope_diff)
                score += 0.10 * slope_match
                if is_surprise and st.surprises_used >= config.surprise_budget:
                    continue
                scored_candidates.append((score, tech, is_surprise, cid, seg, seg_idx))

            # If we ran out of candidates entirely (small pool), the state
            # cannot extend further. Mark it finished if it meets minimums.
            if not scored_candidates and not candidate_ids:
                if (len(st.used_clip_ids) >= config.min_unique_clips
                        and len(st.sequence) >= config.min_clips):
                    finished.append(st)
                continue

            # If nothing passed surprise filter, force-add best candidate anyway
            # (better to make progress than die at 1 clip).
            if not scored_candidates and candidate_ids:
                cid = candidate_ids[0]
                cand = clips[cid]
                seg, seg_idx = pick_segment(
                    cand, target_energy=target_e,
                    exclude_indices=st.used_segments.get(cid, set()),
                    min_seconds=config.min_segment_seconds,
                )
                score, tech, _ = transition_score(
                    last_clip, last.segment, last.target_bpm, last.target_key,
                    cand, seg, config.weights,
                    style_rag=rag,
                    rag_top_k=config.style_rag_top_k,
                    rag_bias_weight=config.style_rag_bias_weight,
                )
                scored_candidates.append((score, tech, True, cid, seg, seg_idx))

            for score, tech, is_surprise, cid, seg, seg_idx in scored_candidates:
                new_used_segs = {k: set(v) for k, v in st.used_segments.items()}
                if seg_idx >= 0:
                    new_used_segs.setdefault(cid, set()).add(seg_idx)
                new_entry = TimelineEntry(
                    clip_id=cid, segment=seg,
                    target_bpm=last.target_bpm,
                    target_key=last.target_key,
                    transition_in=tech,
                )
                next_beam.append(State(
                    sequence=st.sequence + [new_entry],
                    cumulative_duration=st.cumulative_duration + (seg['end'] - seg['start']),
                    used_clip_ids=st.used_clip_ids | {cid},
                    recent_clip_ids=(st.recent_clip_ids + [cid])[-10:],
                    used_segments=new_used_segs,
                    surprises_used=st.surprises_used + (1 if is_surprise else 0),
                    callbacks_used=st.callbacks_used,
                    score=st.score + score,
                ))
        beam = sorted(next_beam, key=lambda s: -s.score)[:config.beam_width]

    if not finished:
        # Fallback: use best partial. Prefer states that satisfy min_unique_clips,
        # then highest score. Avoid silently returning a 1-clip state when a
        # multi-clip partial state was reachable.
        candidates_pool = list(beam) + list(starts)
        if candidates_pool:
            candidates_pool.sort(
                key=lambda s: (
                    len(s.used_clip_ids) >= config.min_unique_clips,
                    len(s.used_clip_ids),
                    s.score,
                ),
                reverse=True,
            )
            finished = candidates_pool[:1]
    # Pick best by score, with a floor on unique-clip count if any state qualifies
    qualified = [s for s in finished
                 if len(s.used_clip_ids) >= config.min_unique_clips]
    pool = qualified or finished
    best = max(pool, key=lambda s: s.score)
    print(f"plan: {len(best.sequence)} entries, "
          f"{best.cumulative_duration:.1f}s "
          f"(target {config.target_duration:.0f}s), "
          f"surprises_used={best.surprises_used}")

    # Insert callback (Loop Callback technique) — repeat strongest hook later in set
    if config.callback_budget > 0 and len(best.sequence) > 4:
        seq = deepcopy(best.sequence)
        candidates: list[tuple[float, str, dict]] = []
        for e in seq[: len(seq) // 2]:
            for h in clips[e.clip_id].get('hooks', []):
                candidates.append((h.get('strength', 0.0), e.clip_id, h))
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            _, cid, hook = candidates[0]
            insert_at = max(1, len(seq) * 3 // 4)
            anchor = seq[min(insert_at, len(seq) - 1)]
            callback_entry = TimelineEntry(
                clip_id=cid,
                segment={'start': hook['start'], 'end': hook['end'],
                         'type': 'callback', 'energy': 0.7},
                target_bpm=anchor.target_bpm,
                target_key=anchor.target_key,
                transition_in={'name': 'loop_callback',
                               'bars': hook.get('bars', 8),
                               'repetitions': 2},
            )
            seq.insert(insert_at, callback_entry)
            best.sequence = seq

    # Schedule play_at times (cumulative)
    t = 0.0
    for e in best.sequence:
        e.play_at = t
        t += e.segment['end'] - e.segment['start']

    return [asdict(e) for e in best.sequence]


def save_timeline(timeline: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'timeline': timeline}, f, indent=2)


# ---------------------------------------------------------------------------
# Path F: N-best generation + heuristic + CLAP rerank
# ---------------------------------------------------------------------------

def _section_vocal_activity(clip_id: str, section: dict, cache_dir: str | Path) -> float:
    """Cheap vocal-active probe via cached vocals stem RMS over [start, end].

    Returns vocal_rms / (vocal_rms + inst_rms) in [0, 1].
    1.0 = pure vocal, 0.0 = pure instrumental. Uses Demucs stems on disk.
    Cached per (clip, section_idx) at module level.
    """
    import soundfile as sf
    cache = Path(cache_dir)
    vox_path = cache / 'stems' / clip_id / 'vocals.wav'
    if not vox_path.exists():
        return 0.5
    s, e = float(section.get('start', 0)), float(section.get('end', 0))
    if e <= s:
        return 0.5
    try:
        info = sf.info(str(vox_path))
        sr = info.samplerate
        start_f = max(0, int(s * sr))
        stop_f = min(info.frames, int(e * sr))
        vox, _ = sf.read(str(vox_path), start=start_f, stop=stop_f, always_2d=False)
    except Exception:
        return 0.5
    if vox.ndim > 1:
        vox = vox.mean(axis=-1)
    vox_rms = float(np.sqrt(np.mean(vox ** 2)) + 1e-8)
    # estimate inst RMS via drums+bass+other if present
    inst_rms = 1e-8
    for stem in ('drums', 'bass', 'other'):
        sp = cache / 'stems' / clip_id / f'{stem}.wav'
        if not sp.exists():
            continue
        try:
            arr, _ = sf.read(str(sp), start=start_f, stop=stop_f, always_2d=False)
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            inst_rms += float(np.sqrt(np.mean(arr ** 2)))
        except Exception:
            pass
    return vox_rms / (vox_rms + inst_rms)


def _timeline_quality(timeline: list[dict], clips: dict[str, dict],
                      cache_dir: str | Path,
                      target_duration: float) -> tuple[float, dict]:
    """Heuristic quality score for a timeline. Higher = better.

    Components (signed):
      + duration_match — closer to target = better
      + clap_coherence — avg CLAP cosine between consecutive clips
      + unique_clip_count — more variety
      - vocal_collisions — penalty when overlap-style transitions land on
        sections that are both vocal-active
      - bpm_strain — penalty for time-stretch >5%
    """
    if not timeline:
        return -1e9, {}
    total_dur = sum(e['segment']['end'] - e['segment']['start'] for e in timeline)
    duration_match = max(0.0, 1.0 - abs(total_dur - target_duration) / max(target_duration, 1.0))

    # CLAP coherence
    clap_pairs = []
    for a, b in zip(timeline[:-1], timeline[1:]):
        ca = clips[a['clip_id']].get('clap')
        cb = clips[b['clip_id']].get('clap')
        if ca is None or cb is None:
            continue
        ca_n = ca / (np.linalg.norm(ca) + 1e-8)
        cb_n = cb / (np.linalg.norm(cb) + 1e-8)
        clap_pairs.append(float(ca_n @ cb_n))
    clap_coherence = float(np.mean(clap_pairs)) if clap_pairs else 0.0

    # Vocal collision penalty: for each transition, check if both sides vocal-active
    overlap_techs = {'crossfade', 'eq_swap', 'filter_fade', 'echo_out', 'mashup'}
    vocal_collisions = 0
    for a, b in zip(timeline[:-1], timeline[1:]):
        tech = (b.get('transition_in') or {}).get('name', '')
        if tech not in overlap_techs:
            continue
        a_vox = _section_vocal_activity(a['clip_id'], a['segment'], cache_dir)
        b_vox = _section_vocal_activity(b['clip_id'], b['segment'], cache_dir)
        if a_vox > 0.25 and b_vox > 0.25:
            vocal_collisions += 1

    # BPM strain penalty
    bpm_strain = 0.0
    for a, b in zip(timeline[:-1], timeline[1:]):
        ta = clips[a['clip_id']].get('tempo', 0) or 0
        tb = clips[b['clip_id']].get('tempo', 0) or 0
        if ta > 0 and tb > 0:
            diff = abs(tb - ta) / max(ta, tb)
            bpm_strain += max(0.0, diff - 0.05)

    unique_clips = len({e['clip_id'] for e in timeline})

    score = (
        2.0 * duration_match
        + 1.5 * clap_coherence
        + 0.05 * unique_clips
        - 1.5 * vocal_collisions
        - 1.0 * bpm_strain
    )
    breakdown = dict(
        duration_match=duration_match,
        clap_coherence=clap_coherence,
        unique_clips=unique_clips,
        vocal_collisions=vocal_collisions,
        bpm_strain=bpm_strain,
        total=score,
    )
    return score, breakdown


def plan_n_best(clips: dict[str, dict], config: PlannerConfig,
                cache_dir: str | Path,
                n_candidates: int = 5,
                verbose: bool = True) -> tuple[list[dict], dict]:
    """Generate N candidate timelines by varying planner config; rerank by
    heuristic quality (CLAP coherence + vocal-collision penalty + duration
    match). Returns (best_timeline, scoring_metadata).
    """
    variants = []
    base_surprise = config.surprise_budget
    base_beam = config.beam_width
    # Variations spanning conservative -> exploratory
    grid = [
        dict(surprise_budget=0,                  beam_width=max(1, base_beam // 2)),
        dict(surprise_budget=base_surprise // 2, beam_width=base_beam),
        dict(surprise_budget=base_surprise,      beam_width=base_beam),
        dict(surprise_budget=base_surprise * 2,  beam_width=base_beam),
        dict(surprise_budget=base_surprise * 3,  beam_width=base_beam * 2),
    ][:n_candidates]
    candidates: list[tuple[float, list[dict], dict]] = []
    for i, override in enumerate(grid):
        cfg = deepcopy(config)
        for k, v in override.items():
            setattr(cfg, k, v)
        try:
            tl = plan(clips, cfg)
        except Exception as e:
            if verbose:
                print(f"[N-best #{i}] plan failed: {e}")
            continue
        score, br = _timeline_quality(tl, clips, cache_dir, cfg.target_duration)
        if verbose:
            print(f"[N-best #{i}] sb={override['surprise_budget']} bw={override['beam_width']} "
                  f"score={score:.3f} entries={len(tl)} "
                  f"clap={br.get('clap_coherence',0):.3f} "
                  f"vox_coll={br.get('vocal_collisions',0)} "
                  f"bpm_strain={br.get('bpm_strain',0):.3f}")
        candidates.append((score, tl, br))
    if not candidates:
        # Fallback to single planner run
        tl = plan(clips, config)
        return tl, {'note': 'no candidates ranked, used base config'}
    candidates.sort(key=lambda x: -x[0])
    best_score, best_tl, best_br = candidates[0]
    return best_tl, {
        'best_score': best_score,
        'best_breakdown': best_br,
        'all_scores': [c[0] for c in candidates],
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='cache')
    ap.add_argument('--out', default='output/timeline.json')
    ap.add_argument('--duration', type=float, default=1800.0)
    ap.add_argument('--surprises', type=int, default=1)
    ap.add_argument('--callbacks', type=int, default=1)
    ap.add_argument('--max_clips', type=int, default=20)
    ap.add_argument('--min_unique_clips', type=int, default=5,
                    help='minimum distinct clips that must appear in mix')
    ap.add_argument('--style_rag', default=None,
                    help='reference dir for Style-RAG bias (optional)')
    ap.add_argument('--classifier', default=None,
                    help='path to trained technique classifier .pt (optional)')
    ap.add_argument('--compat_head', default=None,
                    help='path to CLAP compat head .pt (Tier 1.5, optional)')
    ap.add_argument('--prompt', default=None,
                    help='natural-language mix description, e.g. '
                         '"uplifting trance peak-time set"')
    ap.add_argument('--arc', default='build',
                    choices=list(ARC_PRESETS.keys()) + ['custom'],
                    help='energy arc shape: build|peak|rollercoaster|descend|flat_high|flat_low')
    args = ap.parse_args()
    clips = load_clips(args.cache)
    if not clips:
        print(f"no analyzed clips in {args.cache}. Run analyze first.")
        raise SystemExit(1)
    cfg = PlannerConfig(
        target_duration=args.duration,
        surprise_budget=args.surprises,
        callback_budget=args.callbacks,
        max_clips=args.max_clips,
        min_unique_clips=args.min_unique_clips,
        style_rag_dir=args.style_rag,
        classifier_ckpt=args.classifier,
        compat_head_ckpt=args.compat_head,
        text_prompt=args.prompt,
        arc_shape=args.arc,
    )
    tl = plan(clips, cfg)
    save_timeline(tl, args.out)
    print(f"wrote {args.out} ({len(tl)} entries)")
