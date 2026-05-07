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
    target_duration: float = 600.0
    energy_arc: list[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.7, 0.9, 1.0, 0.85, 0.6, 0.3])
    surprise_budget: int = 1
    callback_budget: int = 1
    beam_width: int = 12
    max_clips: int = 20
    min_clips: int = 3
    weights: dict = field(default_factory=lambda: dict(
        key=0.25, tempo=0.20, energy=0.20, timbre=0.15,
        variety=0.10, surprise=0.10,
    ))
    style_rag_dir: str | None = None     # if set, use Style-RAG bias
    style_rag_top_k: int = 5
    style_rag_bias_weight: float = 0.15  # max bonus added to score from RAG


@dataclass
class State:
    sequence: list[TimelineEntry]
    cumulative_duration: float = 0.0
    used_clip_ids: set[str] = field(default_factory=set)
    surprises_used: int = 0
    callbacks_used: int = 0
    score: float = 0.0


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
                     rag_bias_weight: float = 0.15) -> tuple[float, dict, bool]:
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

    a = _normalize_clap(np.asarray(prev_clip.get('clap', np.zeros(512)), dtype=np.float32))
    b = _normalize_clap(np.asarray(cand_clip.get('clap', np.zeros(512)), dtype=np.float32))
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


# ---------------------------------------------------------------------------
# Segment selection
# ---------------------------------------------------------------------------

def pick_segment(clip: dict, prefer: str | None = None,
                 target_energy: float | None = None) -> dict:
    sections = clip.get('sections', [])
    if not sections:
        return {'start': 0.0, 'end': clip.get('duration', 30.0),
                'type': 'unknown', 'energy': 0.5}
    if prefer:
        for s in sections:
            if s.get('type') == prefer:
                return dict(s)
    if target_energy is not None:
        return dict(min(sections,
                        key=lambda s: abs(s.get('energy', 0.5) - target_energy)))
    body = [s for s in sections if s.get('type') in ('drop', 'verse', 'breakdown')]
    return dict(max(body or sections, key=lambda s: s['end'] - s['start']))


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------

def plan(clips: dict[str, dict], config: PlannerConfig) -> list[dict]:
    """Returns timeline as list of dicts (asdict of TimelineEntry)."""
    if not clips:
        return []

    # Optional Style-RAG index
    rag = None
    if config.style_rag_dir:
        rag = StyleRAG(config.style_rag_dir)
        if len(rag) > 0:
            print(f"Style-RAG: loaded {len(rag)} reference patterns")
        else:
            print(f"Style-RAG: no patterns found in {config.style_rag_dir}, skipping bias")
            rag = None

    # Initial states — try each clip as opener
    starts: list[State] = []
    for cid, clip in clips.items():
        seg = pick_segment(clip, prefer='intro') if any(
            s.get('type') == 'intro' for s in clip.get('sections', [])
        ) else pick_segment(clip)
        target_bpm = clip.get('tempo', 128.0) or 128.0
        entry = TimelineEntry(
            clip_id=cid, segment=seg, target_bpm=target_bpm,
            target_key=clip.get('key', '?'),
            transition_in={'name': 'fade_in', 'bars': 4},
        )
        starts.append(State(
            sequence=[entry],
            cumulative_duration=seg['end'] - seg['start'],
            used_clip_ids={cid},
            score=0.0,
        ))

    beam = sorted(starts, key=lambda s: -s.score)[:config.beam_width]
    finished: list[State] = []

    while beam:
        next_beam: list[State] = []
        for st in beam:
            if (st.cumulative_duration >= config.target_duration
                    or len(st.sequence) >= config.max_clips):
                if len(st.sequence) >= config.min_clips:
                    finished.append(st)
                continue
            progress = st.cumulative_duration / config.target_duration
            arc_idx = min(int(progress * len(config.energy_arc)),
                          len(config.energy_arc) - 1)
            target_e = config.energy_arc[arc_idx]
            last = st.sequence[-1]
            last_clip = clips[last.clip_id]
            for cid, cand in clips.items():
                if cid in st.used_clip_ids:
                    continue
                seg = pick_segment(cand, target_energy=target_e)
                score, tech, is_surprise = transition_score(
                    last_clip, last.segment, last.target_bpm, last.target_key,
                    cand, seg, config.weights,
                    style_rag=rag,
                    rag_top_k=config.style_rag_top_k,
                    rag_bias_weight=config.style_rag_bias_weight,
                )
                if is_surprise and st.surprises_used >= config.surprise_budget:
                    continue
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
                    surprises_used=st.surprises_used + (1 if is_surprise else 0),
                    callbacks_used=st.callbacks_used,
                    score=st.score + score,
                ))
        beam = sorted(next_beam, key=lambda s: -s.score)[:config.beam_width]

    if not finished:
        finished = sorted(starts, key=lambda s: -s.score)[:1]
        if beam:
            finished = sorted(beam, key=lambda s: -s.score)[:1]
    best = max(finished, key=lambda s: s.score / max(1, len(s.sequence)))

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


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='cache')
    ap.add_argument('--out', default='output/timeline.json')
    ap.add_argument('--duration', type=float, default=600.0)
    ap.add_argument('--surprises', type=int, default=1)
    ap.add_argument('--callbacks', type=int, default=1)
    ap.add_argument('--max_clips', type=int, default=20)
    ap.add_argument('--style_rag', default=None,
                    help='reference dir for Style-RAG bias (optional)')
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
        style_rag_dir=args.style_rag,
    )
    tl = plan(clips, cfg)
    save_timeline(tl, args.out)
    print(f"wrote {args.out} ({len(tl)} entries)")
