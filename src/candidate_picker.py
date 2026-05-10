"""Candidate-scored transition picker (Tier-1 follow-up H).

Replaces ad-hoc junction selection with a 3-5-candidate-per-clip scored
picker. Empirically grounded on All-In-One's labeled segments (verse,
chorus, break, bridge, inst, solo, intro, outro) instead of vibes.

Multi-factor score per candidate:
    1. energy_curve_match    — segment energy vs arc target
    2. transition_type_fit   — certain types prefer certain section labels
    3. vocal_presence        — low vocal activity preferred at junction
    4. key_compat            — Camelot distance to target key
    5. bpm_strain            — distance from target BPM (stretch cost)
    6. duration_fit          — long enough to render, not so long it dominates

Each weight tunable via env knobs (defaults conservative).

Why this addresses phase cancellation:
    DTW handles SAMPLE-jitter phase cancellation. Candidate-picker
    addresses BAR-GRID-mismatch phase cancellation by ensuring junctions
    land at structural reset points (downbeat 1 of bar 1 of section
    start) where both sides are at musical "zero" — minimal harmonic
    overlap by design.

Inspired by kckDeepak/AI-DJ-Mixing-System (Camelot-driven dynamic
junction choice).

API:

    candidates = build_candidates(clip_meta, sections)
    best = pick_best_junction(
        prev_meta, candidates,
        target_bpm=128, target_key='8A', target_energy=0.7,
        transition_type='filter_fade',
    )
    if best is None:
        # All-In-One labels absent or no candidates met threshold; caller
        # falls back to existing planner logic.
        ...

Env:
    AIJOCKEY_CANDIDATE_PICKER         0|1   default 0 (opt-in)
    AIJOCKEY_PICKER_W_ENERGY          float default 1.0
    AIJOCKEY_PICKER_W_TYPE_FIT        float default 1.5
    AIJOCKEY_PICKER_W_VOCAL           float default 1.0
    AIJOCKEY_PICKER_W_KEY             float default 1.2
    AIJOCKEY_PICKER_W_BPM             float default 0.8
    AIJOCKEY_PICKER_W_DURATION        float default 0.5
    AIJOCKEY_PICKER_MIN_SCORE         float default 0.0  (negative = reject)

Lazy: no model loads, pure scoring math. Sub-millisecond per candidate.
"""
from __future__ import annotations

import os
from typing import Any

# Camelot distance — same import the rest of the pipeline uses.
try:
    from camelot import camelot_distance
except ImportError:
    def camelot_distance(a: str, b: str) -> int:
        return 0


# ---------------------------------------------------------------------------
# Section-label → transition-type fit scores
# ---------------------------------------------------------------------------

# How well does a section label suit a given transition technique?
# Range [0, 1]. 1 = ideal fit (drop tier loves pre_drop / chorus_end).
# 0 = bad fit (drop tier on intro is an energy crater).
_TYPE_FIT: dict[str, dict[str, float]] = {
    # Drop / climax transitions love drop-section / chorus / pre-drop
    'build_riser_drop': {
        'chorus': 1.0, 'drop': 1.0, 'solo': 0.85, 'inst': 0.7,
        'verse': 0.4, 'bridge': 0.4, 'break': 0.3, 'outro': 0.1, 'intro': 0.1,
    },
    'silence_drop': {
        'chorus': 0.95, 'drop': 1.0, 'pre_drop': 1.0, 'solo': 0.7,
        'verse': 0.5, 'bridge': 0.5, 'break': 0.3, 'outro': 0.1, 'intro': 0.1,
    },
    # Major-tier transitions land best on structural transitions
    'filter_fade': {
        'chorus': 0.85, 'verse': 0.85, 'bridge': 0.95, 'break': 0.8,
        'inst': 0.8, 'solo': 0.6, 'drop': 0.5, 'outro': 0.6, 'intro': 0.5,
    },
    'drum_break': {
        'break': 1.0, 'bridge': 0.9, 'inst': 0.7, 'verse': 0.6,
        'chorus': 0.4, 'drop': 0.3, 'solo': 0.5, 'outro': 0.3, 'intro': 0.4,
    },
    'echo_out': {
        'outro': 1.0, 'break': 0.8, 'bridge': 0.7, 'verse': 0.6,
        'chorus': 0.6, 'inst': 0.6, 'solo': 0.6, 'drop': 0.4, 'intro': 0.2,
    },
    # Minor-tier (smooth crossfade) is mostly section-agnostic but slight
    # preference for sustained material over breakdown
    'crossfade': {
        'verse': 0.85, 'chorus': 0.8, 'inst': 0.8, 'bridge': 0.7,
        'solo': 0.7, 'break': 0.6, 'drop': 0.7, 'outro': 0.5, 'intro': 0.5,
    },
    'eq_swap': {
        'verse': 0.85, 'chorus': 0.85, 'inst': 0.85, 'bridge': 0.7,
        'solo': 0.7, 'break': 0.6, 'drop': 0.6, 'outro': 0.5, 'intro': 0.5,
    },
}

_DEFAULT_TYPE_FIT = 0.5    # unknown transition type → neutral


def _type_fit_score(transition_type: str, section_label: str) -> float:
    """0..1 score for how well a section type suits a transition technique."""
    if not transition_type:
        return _DEFAULT_TYPE_FIT
    table = _TYPE_FIT.get(transition_type.lower())
    if table is None:
        return _DEFAULT_TYPE_FIT
    return table.get(section_label.lower(), _DEFAULT_TYPE_FIT)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enabled() -> bool:
    return os.environ.get('AIJOCKEY_CANDIDATE_PICKER', '0') == '1'


def build_candidates(clip_meta: dict, sections: list[dict] | None = None,
                      min_seconds: float = 8.0,
                      max_seconds: float = 60.0,
                      preferred_min_seconds: float = 20.0,
                      hard_min_seconds: float = 8.0,
                      exclude_indices: set | None = None,
                      downbeats: list[float] | None = None) -> list[dict]:
    """Build candidate junctions from a clip's labeled sections.

    Output dict keys: start, end, label, canonical_label, energy,
    vocal_activity, duration, has_vocals, duration_penalty, orig_index.

    Tiered duration (#6): >=preferred_min penalty=0; hard_min..preferred
    penalty=linear 0..1; <hard_min rejected.

    `exclude_indices` (#4): drop sections at original indices — for
    callback / reuse paths to force a DIFFERENT section on revisit.

    `downbeats` (#5): if provided, snap start/end to nearest downbeat
    within ±2 bars (BPM-derived). Phrase-aligned junctions reduce
    bar-grid mismatch phase cancellation.
    """
    if sections is None:
        sections = clip_meta.get('sections') or []
    exclude = exclude_indices or set()
    try:
        from picker_synonyms import canonical_section as _canon
    except Exception:
        def _canon(x):
            return (x or '').lower()

    bpm = float(clip_meta.get('tempo', 120.0)) or 120.0
    bar_dur = 4.0 * 60.0 / bpm
    snap_drift_max = 2.0 * bar_dur

    def _snap(t: float) -> float:
        if not downbeats:
            return t
        nearest = min(downbeats, key=lambda d: abs(d - t))
        return float(nearest) if abs(nearest - t) <= snap_drift_max else t

    candidates: list[dict] = []
    for i, s in enumerate(sections):
        if i in exclude:
            continue
        start = float(s.get('start', 0.0))
        end = float(s.get('end', 0.0))
        if downbeats:
            start = _snap(start)
            end = _snap(end)
        dur = end - start
        if dur < hard_min_seconds:
            continue
        if dur > max_seconds:
            end = start + max_seconds
            dur = max_seconds
        if dur < preferred_min_seconds:
            denom = max(1e-6, preferred_min_seconds - hard_min_seconds)
            dur_penalty = (preferred_min_seconds - dur) / denom
        else:
            dur_penalty = 0.0
        label = str(s.get('label', s.get('type', '?'))).lower()
        canonical = _canon(label)
        energy = float(s.get('energy', _label_energy(canonical or label)))
        va_raw = s.get('vocal_activity')
        va = float(va_raw) if isinstance(va_raw, (int, float)) else None
        if va is not None:
            has_vocals = va > 0.20
        else:
            has_vocals = canonical in ('verse', 'chorus', 'bridge', 'outro')
        candidates.append({
            'start': start,
            'end': end,
            'label': label,
            'canonical_label': canonical,
            'energy': energy,
            'vocal_activity': va,
            'duration': dur,
            'has_vocals': has_vocals,
            'duration_penalty': dur_penalty,
            'orig_index': i,
        })
    return candidates


def filter_by_transition_requires(candidates: list[dict],
                                   transition_type: str,
                                   side: str = 'prev') -> list[dict]:
    """Drop candidates that don't satisfy a transition's requires spec.

    side: 'prev' (outgoing) | 'cur' (incoming).
    Returns input unchanged on import failure / no constraints.
    """
    if not transition_type:
        return list(candidates)
    try:
        from picker_synonyms import requires_for, section_satisfies
    except Exception:
        return list(candidates)
    spec = (requires_for(transition_type) or {}).get(f'{side}_requires')
    if not spec:
        return list(candidates)
    return [c for c in candidates if section_satisfies(c, spec)]


def filter_by_catalog_compat(candidates: list[dict],
                              transition_type: str) -> list[dict]:
    """Pre-pick check: reject candidates with section in transition's
    catalog `incompatible_with` list (#14)."""
    if not transition_type:
        return list(candidates)
    try:
        from transition_catalog import get as _cat_get
    except Exception:
        return list(candidates)
    entry = _cat_get(transition_type)
    if not entry:
        return list(candidates)
    incompat = [s.lower() for s in (entry.get('incompatible_with') or [])]
    if not incompat:
        return list(candidates)
    return [c for c in candidates
            if c.get('canonical_label', '').lower() not in incompat
            and c.get('label', '').lower() not in incompat]


_LABEL_ENERGY_DEFAULTS = {
    'intro': 0.30, 'outro': 0.25, 'break': 0.35, 'bridge': 0.50,
    'inst': 0.55, 'verse': 0.60, 'solo': 0.75, 'chorus': 0.85,
    'drop': 0.95, 'pre_drop': 0.85,
}


def _label_energy(label: str) -> float:
    return _LABEL_ENERGY_DEFAULTS.get(label.lower(), 0.5)


_AGGRESSIVE_TRANSITIONS = frozenset({
    'chop', 'tape_stop', 'drum_replace', 'kickless_swap',
    'spinback', 'forward_spin', 'build_riser_drop', 'snare_buildup',
    'scratch_fill', 'loop_tighten', 'loop_roll', 'beat_juggle',
    'pitch_bend', 'bpm_warp', 'spectral_hold', 'silence_drop',
    'breakdown_to_drop', 'drum_break',
})


def score_candidate(cand: dict, prev_meta: dict, *,
                    target_bpm: float | None = None,
                    target_key: str | None = None,
                    target_energy: float | None = None,
                    transition_type: str = '',
                    next_transition_type: str = '',
                    weights: dict | None = None) -> tuple[float, dict]:
    """Multi-factor score in roughly [-3, +3] range. Higher = better.

    Returns (score, breakdown) where breakdown is per-factor contributions
    so the picker can log why a candidate won / lost.

    next_transition_type (#7): when the FOLLOWING transition is in the
    aggressive set, penalize high-VA candidates pre-emptively. Avoids
    the picker choosing a vocal chorus only for vocal_guard to downgrade
    the next transition to crossfade later.
    """
    w = _weights(weights)

    # 1. Energy curve match — penalize big mismatch from target_energy
    if target_energy is not None:
        e_diff = abs(cand['energy'] - float(target_energy))
        # 0 mismatch → +1 contribution; 1.0 mismatch → -1 contribution
        e_contrib = (0.5 - e_diff) * 2.0
    else:
        e_contrib = 0.0

    # 2. Transition-type fit (uses canonical_label when available)
    label_for_fit = cand.get('canonical_label') or cand.get('label', '')
    fit = _type_fit_score(transition_type, label_for_fit)
    # 0.5 fit (neutral) → 0 contribution; 1.0 fit → +1; 0.0 fit → -1
    fit_contrib = (fit - 0.5) * 2.0

    # 3. Vocal presence — penalize vocals at junctions (collisions)
    va_actual = cand.get('vocal_activity')
    if isinstance(va_actual, (int, float)):
        # 0 va → +0.5; 0.5 va → 0; 1.0 va → -0.5
        vocal_contrib = 0.5 - float(va_actual)
    else:
        vocal_contrib = -0.5 if cand['has_vocals'] else 0.5
    # #7 — extra penalty when NEXT transition is aggressive (vocal_guard
    # would otherwise downgrade it). Pre-empt at picker layer.
    if next_transition_type and next_transition_type.lower() in _AGGRESSIVE_TRANSITIONS:
        if isinstance(va_actual, (int, float)) and va_actual > 0.30:
            vocal_contrib -= 0.8 * (float(va_actual) - 0.30) / 0.70
        elif cand.get('has_vocals'):
            vocal_contrib -= 0.5

    # 4. Camelot key compat (penalize larger distance)
    if target_key:
        clip_key = prev_meta.get('key', '?')
        try:
            kdist = camelot_distance(clip_key, target_key)
        except Exception:
            kdist = 0
        # 0 dist → +1, 6+ dist → -1
        key_contrib = max(-1.0, 1.0 - kdist / 3.0)
    else:
        key_contrib = 0.0

    # 5. BPM strain (stretch ratio cost)
    if target_bpm and prev_meta.get('tempo'):
        try:
            ratio = float(target_bpm) / float(prev_meta['tempo'])
            strain = abs(ratio - 1.0)
            # 0 strain → +1; 0.2+ strain → -1
            bpm_contrib = max(-1.0, 1.0 - strain * 5.0)
        except Exception:
            bpm_contrib = 0.0
    else:
        bpm_contrib = 0.0

    # 6. Duration fit — slight preference for ~16-32s sections
    dur = cand['duration']
    if 12.0 <= dur <= 40.0:
        dur_contrib = 0.5
    elif dur < 8.0:
        dur_contrib = -1.0
    else:
        dur_contrib = 0.0    # tolerable but not preferred

    # Tiered duration penalty (#6) — graded reduction within preferred
    # band. Stacks with hard 8/40/12-40 logic above.
    dp = float(cand.get('duration_penalty', 0.0))
    if dp > 0:
        dur_contrib -= dp     # subtract penalty (0..1) from contribution

    breakdown = {
        'energy':   round(w['energy']   * e_contrib, 3),
        'type_fit': round(w['type_fit'] * fit_contrib, 3),
        'vocal':    round(w['vocal']    * vocal_contrib, 3),
        'key':      round(w['key']      * key_contrib, 3),
        'bpm':      round(w['bpm']      * bpm_contrib, 3),
        'duration': round(w['duration'] * dur_contrib, 3),
    }
    score = sum(breakdown.values())
    return score, breakdown


def pick_best_junction(prev_meta: dict, candidates: list[dict], *,
                       target_bpm: float | None = None,
                       target_key: str | None = None,
                       target_energy: float | None = None,
                       transition_type: str = '',
                       next_transition_type: str = '',
                       weights: dict | None = None,
                       min_score: float | None = None,
                       ) -> dict | None:
    """Score all candidates, return the highest-scoring (with breakdown
    + ranks attached). Returns None when no candidates OR best score
    below min_score threshold.

    Output dict (when not None):
        {**candidate_fields, 'score': float, 'breakdown': dict, 'rank': 1, 'all_scores': [...]}
    """
    if min_score is None:
        min_score = float(os.environ.get('AIJOCKEY_PICKER_MIN_SCORE', '0.0'))
    if not candidates:
        return None

    scored = []
    for c in candidates:
        s, b = score_candidate(
            c, prev_meta,
            target_bpm=target_bpm,
            target_key=target_key,
            target_energy=target_energy,
            transition_type=transition_type,
            next_transition_type=next_transition_type,
            weights=weights,
        )
        scored.append((s, b, c))

    scored.sort(key=lambda x: -x[0])
    best_score, best_break, best_cand = scored[0]
    if best_score < min_score:
        return None

    out = dict(best_cand)
    out['score'] = round(best_score, 3)
    out['breakdown'] = best_break
    out['rank'] = 1
    out['all_scores'] = [
        {'label': c['label'], 'start': c['start'],
         'score': round(s, 3), 'breakdown': b}
        for s, b, c in scored
    ]
    return out


def _weights(override: dict | None = None) -> dict[str, float]:
    base = {
        'energy':   float(os.environ.get('AIJOCKEY_PICKER_W_ENERGY',   '1.0')),
        'type_fit': float(os.environ.get('AIJOCKEY_PICKER_W_TYPE_FIT', '1.5')),
        'vocal':    float(os.environ.get('AIJOCKEY_PICKER_W_VOCAL',    '1.0')),
        'key':      float(os.environ.get('AIJOCKEY_PICKER_W_KEY',      '1.2')),
        'bpm':      float(os.environ.get('AIJOCKEY_PICKER_W_BPM',      '0.8')),
        'duration': float(os.environ.get('AIJOCKEY_PICKER_W_DURATION', '0.5')),
    }
    if override:
        base.update(override)
    return base


__all__ = [
    'enabled',
    'build_candidates',
    'score_candidate',
    'pick_best_junction',
]
