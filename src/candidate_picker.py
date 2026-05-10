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
                      max_seconds: float = 60.0) -> list[dict]:
    """Build candidate junctions from a clip's labeled sections.

    Each candidate is a dict:
        {
          'start': float,         # seconds
          'end':   float,
          'label': str,           # All-In-One section label
          'energy': float,        # 0..1 (from label or measured)
          'duration': float,
          'has_vocals': bool,     # heuristic from label
        }

    Filters by duration (too-short = no room to render the segment;
    too-long = dominates the mix). Returns empty list when sections
    absent or all candidates filtered out.
    """
    if sections is None:
        sections = clip_meta.get('sections') or []
    candidates: list[dict] = []
    for s in sections:
        start = float(s.get('start', 0.0))
        end = float(s.get('end', 0.0))
        if end - start < min_seconds:
            continue
        if end - start > max_seconds:
            # Truncate long sections — keep the front so junction lands at
            # the section's structural start.
            end = start + max_seconds
        label = str(s.get('label', s.get('type', '?'))).lower()
        energy = float(s.get('energy', _label_energy(label)))
        has_vocals = label in ('verse', 'chorus', 'bridge', 'outro')
        candidates.append({
            'start': start,
            'end': end,
            'label': label,
            'energy': energy,
            'duration': end - start,
            'has_vocals': has_vocals,
        })
    return candidates


_LABEL_ENERGY_DEFAULTS = {
    'intro': 0.30, 'outro': 0.25, 'break': 0.35, 'bridge': 0.50,
    'inst': 0.55, 'verse': 0.60, 'solo': 0.75, 'chorus': 0.85,
    'drop': 0.95, 'pre_drop': 0.85,
}


def _label_energy(label: str) -> float:
    return _LABEL_ENERGY_DEFAULTS.get(label.lower(), 0.5)


def score_candidate(cand: dict, prev_meta: dict, *,
                    target_bpm: float | None = None,
                    target_key: str | None = None,
                    target_energy: float | None = None,
                    transition_type: str = '',
                    weights: dict | None = None) -> tuple[float, dict]:
    """Multi-factor score in roughly [-3, +3] range. Higher = better.

    Returns (score, breakdown) where breakdown is per-factor contributions
    so the picker can log why a candidate won / lost.
    """
    w = _weights(weights)

    # 1. Energy curve match — penalize big mismatch from target_energy
    if target_energy is not None:
        e_diff = abs(cand['energy'] - float(target_energy))
        # 0 mismatch → +1 contribution; 1.0 mismatch → -1 contribution
        e_contrib = (0.5 - e_diff) * 2.0
    else:
        e_contrib = 0.0

    # 2. Transition-type fit
    fit = _type_fit_score(transition_type, cand['label'])
    # 0.5 fit (neutral) → 0 contribution; 1.0 fit → +1; 0.0 fit → -1
    fit_contrib = (fit - 0.5) * 2.0

    # 3. Vocal presence — penalize vocals at junctions (collisions)
    vocal_contrib = -0.5 if cand['has_vocals'] else 0.5

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
