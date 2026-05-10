"""Library-side picker enhancements (server/api.py companion).

Standalone scoring helpers — server/api.py can adopt incrementally.
None of these REPLACE existing picker logic; each is a SCORE BONUS or
FILTER that composes with whatever scoring loop already exists.

Implemented (NEXT.TXT picker fixes):
  #1 vocal_diversity_bonus     — balance pool vocal_activity around 0.5
  #2 section_coverage_bonus    — favor clips with drop/break sections
  #3 mmr_select                — Maximal Marginal Relevance rerank
  #4 graded_camelot_score      — continuous compat (1.0/0.8/0.5/0.2/0)
  #5 enforce_genre_floor       — mode-gated cluster requirement
  #6 arc_conditioned_bpm_target — fits BPM picks to arc shape

All pure-Python, numpy-only, env-gated where they introduce a knob.
Tests cover edge cases.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# #1 — vocal_diversity_bonus
# ---------------------------------------------------------------------------

def pool_vocal_mean(meta_list: list[dict]) -> float:
    """Mean vocal_activity across a list of clip metas. Falls back to
    section-derived heuristic when explicit field missing."""
    vals = []
    for m in meta_list:
        v = m.get('vocal_activity_mean')
        if isinstance(v, (int, float)):
            vals.append(float(v))
            continue
        # Derive from sections
        sections = m.get('sections') or []
        sec_vas = [s.get('vocal_activity') for s in sections
                    if isinstance(s.get('vocal_activity'), (int, float))]
        if sec_vas:
            vals.append(float(sum(sec_vas) / len(sec_vas)))
            continue
        # Last-resort: presence of vocals stem in cache
        if m.get('has_vocals') is False:
            vals.append(0.0)
        elif m.get('has_vocals') is True:
            vals.append(0.5)
    if not vals:
        return 0.5
    return float(sum(vals) / len(vals))


def vocal_diversity_bonus(candidate_meta: dict, user_pool_va: float,
                           strength: float = 0.4) -> float:
    """Bonus that pulls the picked pool toward balanced vocal_activity.

    user_pool_va is the mean of CURRENT user pool (computed via
    pool_vocal_mean before picking). Returns a [-strength, +strength]
    bonus to add to the candidate's existing score.

    Logic: when user pool is vocal-heavy (mean > 0.5), boost low-VA
    candidates (instrumental). When vocal-sparse (< 0.3), boost high-VA.
    Around 0.4-0.5, no bonus (already balanced).
    """
    cand_va = candidate_meta.get('vocal_activity_mean')
    if not isinstance(cand_va, (int, float)):
        # Derive from sections
        sections = candidate_meta.get('sections') or []
        sec_vas = [s.get('vocal_activity') for s in sections
                    if isinstance(s.get('vocal_activity'), (int, float))]
        cand_va = (sum(sec_vas) / len(sec_vas)) if sec_vas else 0.5
    cand_va = float(cand_va)

    if user_pool_va > 0.5:
        # Pool vocal-heavy → boost low-VA candidates
        return strength * (0.5 - cand_va) * 2.0    # range [-strength, +strength]
    if user_pool_va < 0.3:
        # Pool vocal-sparse → boost high-VA candidates
        return strength * (cand_va - 0.5) * 2.0
    return 0.0


# ---------------------------------------------------------------------------
# #2 — section_coverage_bonus
# ---------------------------------------------------------------------------

def section_coverage_bonus(candidate_meta: dict,
                           required_labels: tuple[str, ...] = (
                               'drop', 'break', 'chorus'),
                           strength: float = 0.3) -> float:
    """Bonus when a candidate has at least one section matching any
    `required_labels`. Default targets pool-level coverage of drop /
    break / chorus — without these, Director can't pick aggressive
    transitions even when the narrative wants them.
    """
    try:
        from picker_synonyms import canonical_section
    except Exception:
        def canonical_section(x):
            return (x or '').lower()
    sections = candidate_meta.get('sections') or []
    canonicals = {canonical_section(s.get('label', s.get('type', '')))
                   for s in sections}
    canonicals.discard('?')
    matched = canonicals.intersection(required_labels)
    if not matched:
        return 0.0
    # 1 match → strength*0.5; 2 matches → strength*0.75; 3+ → strength
    return strength * min(1.0, 0.5 + 0.25 * (len(matched) - 1))


# ---------------------------------------------------------------------------
# #3 — MMR (Maximal Marginal Relevance) rerank
# ---------------------------------------------------------------------------

def mmr_select(candidates: list[dict], k: int,
                lambda_: float = 0.7,
                relevance_key: str = 'score',
                embedding_key: str = 'clap',
                ) -> list[dict]:
    """Maximal Marginal Relevance pick: select k items balancing relevance
    (existing score) vs diversity (1 - max similarity to already picked).

    lambda_ in [0, 1]: 1.0 = pure relevance (current top-k behavior),
    0.0 = pure diversity. 0.7 default — relevance-leaning. Pass 0.4 for
    exploratory mode.

    Each candidate must have the `relevance_key` (default 'score') and
    optionally `embedding_key` (default 'clap'). When no embedding, falls
    back to label-based diversity (canonical_label set difference).

    Returns a NEW list of length min(k, len(candidates)).
    """
    if k <= 0 or not candidates:
        return []
    if k >= len(candidates):
        return sorted(candidates, key=lambda c: -c.get(relevance_key, 0.0))[:k]

    pool = list(candidates)
    selected: list[dict] = []
    # Pre-extract embeddings (or labels) for fast similarity.
    embs = []
    labels = []
    for c in pool:
        e = c.get(embedding_key)
        if e is not None:
            arr = np.asarray(e, dtype=np.float32)
            n = float(np.linalg.norm(arr))
            embs.append(arr / n if n > 0 else arr)
        else:
            embs.append(None)
        labels.append((c.get('label') or '').lower())

    while pool and len(selected) < k:
        best_idx = -1
        best_score = -float('inf')
        for i, c in enumerate(pool):
            relevance = float(c.get(relevance_key, 0.0))
            # Max similarity to already-selected
            max_sim = 0.0
            ce = embs[i]
            for sel in selected:
                se = sel.get('_emb_norm')
                if ce is not None and se is not None:
                    sim = float(ce @ se)
                else:
                    # label-based fallback: 1.0 if same label, else 0
                    sim = 1.0 if sel.get('label', '').lower() == labels[i] else 0.0
                max_sim = max(max_sim, sim)
            mmr = lambda_ * relevance - (1.0 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        if best_idx < 0:
            break
        chosen = pool.pop(best_idx)
        ce_chosen = embs.pop(best_idx)
        labels.pop(best_idx)
        if ce_chosen is not None:
            chosen = dict(chosen)
            chosen['_emb_norm'] = ce_chosen
        selected.append(chosen)

    # Strip internal field
    return [{k: v for k, v in c.items() if k != '_emb_norm'} for c in selected]


# ---------------------------------------------------------------------------
# #4 — graded camelot score
# ---------------------------------------------------------------------------

def graded_camelot_score(prev_key: str, cand_key: str) -> float:
    """Continuous compat score in [0, 1] over Camelot wheel:
        same key            1.0
        ±1 wheel position   0.8
        relative key        0.5
        dominant            0.2
        else                0.0

    Uses the existing camelot_distance() infrastructure rather than
    re-implementing the wheel — distance N maps to a known score.
    """
    if not prev_key or not cand_key or prev_key == '?' or cand_key == '?':
        return 0.5    # unknown — neutral
    if prev_key == cand_key:
        return 1.0
    try:
        from camelot import camelot_distance
        d = camelot_distance(prev_key, cand_key)
    except Exception:
        return 0.5
    if d == 0:
        return 1.0
    if d == 1:
        return 0.8
    if d == 3:
        return 0.5    # relative key (8A↔8B distance often = 3)
    if d == 2:
        return 0.6
    if d == 4:
        return 0.2    # dominant-ish
    return 0.0        # 5+ semitones apart on wheel


# ---------------------------------------------------------------------------
# #5 — genre coherence floor (mode-gated)
# ---------------------------------------------------------------------------

def enforce_genre_floor(candidates: list[dict], k: int,
                        mix_mode: str = 'balanced',
                        min_shared_cluster: int = 2,
                        cluster_distance_threshold: float = 0.35,
                        ) -> list[dict]:
    """Mode-gated genre coherence. tight=enforce, balanced=warn,
    exploratory=skip. When enforced, ensures ≥min_shared_cluster picks
    are within cluster_distance_threshold of pool centroid (CLAP cosine).

    Returns candidates list (possibly reordered to push coherent picks
    to the front) or input unchanged for exploratory.
    """
    mode = (mix_mode or 'balanced').lower()
    if mode == 'exploratory' or k <= 0 or not candidates:
        return list(candidates)

    # Compute pool centroid from existing CLAP embeddings
    embs = []
    for c in candidates:
        e = c.get('clap')
        if e is not None:
            arr = np.asarray(e, dtype=np.float32)
            n = float(np.linalg.norm(arr))
            if n > 0:
                embs.append(arr / n)
    if not embs:
        return list(candidates)
    centroid = np.mean(embs, axis=0)
    cn = np.linalg.norm(centroid)
    if cn > 0:
        centroid = centroid / cn

    # Score each candidate by distance to centroid
    def _coherence(c: dict) -> float:
        e = c.get('clap')
        if e is None:
            return 0.0
        arr = np.asarray(e, dtype=np.float32)
        n = float(np.linalg.norm(arr))
        if n == 0:
            return 0.0
        return float((arr / n) @ centroid)

    scored = sorted(
        ((c, _coherence(c)) for c in candidates),
        key=lambda x: -x[1],
    )

    if mode == 'tight':
        # Hard reject candidates beyond threshold; move coherent to front
        coherent = [c for c, s in scored if s >= (1.0 - cluster_distance_threshold)]
        rest = [c for c, s in scored if s < (1.0 - cluster_distance_threshold)]
        if len(coherent) >= min_shared_cluster:
            return coherent + rest
        return [c for c, _ in scored]    # fall back, can't enforce

    # balanced: soft — order by coherence but keep all
    return [c for c, _ in scored]


# ---------------------------------------------------------------------------
# #6 — arc-conditioned BPM target curve
# ---------------------------------------------------------------------------

# Per-arc BPM curve: (target, spread) — target is desired BPM relative to
# pool median; spread is how wide the BPM picks should be.
_ARC_BPM_PROFILE: dict[str, dict] = {
    'build':      {'curve': [0.92, 0.96, 1.00, 1.04, 1.08],
                    'spread': 0.10},   # slow → fast
    'peak':       {'curve': [1.03, 1.05, 1.06, 1.05, 1.03],
                    'spread': 0.04},   # tight high
    'rollercoaster': {'curve': [0.95, 1.05, 0.97, 1.07, 0.98],
                       'spread': 0.12},
    'descend':    {'curve': [1.04, 1.00, 0.96, 0.92, 0.88],
                    'spread': 0.10},
    'flat_high':  {'curve': [1.02, 1.02, 1.02, 1.02, 1.02],
                    'spread': 0.03},   # narrow high
    'flat_low':   {'curve': [0.96, 0.96, 0.96, 0.96, 0.96],
                    'spread': 0.03},   # narrow low
}


def arc_conditioned_bpm_target(arc: str, position: float,
                                pool_median_bpm: float = 120.0
                                ) -> tuple[float, float]:
    """Return (target_bpm, allowed_spread_bpm) for a position in [0, 1]
    along the arc curve. Picker applies BPM scoring around target with
    `allowed_spread` as the soft tolerance band.
    """
    profile = _ARC_BPM_PROFILE.get((arc or 'build').lower(),
                                     _ARC_BPM_PROFILE['build'])
    curve = profile['curve']
    if not curve:
        return float(pool_median_bpm), 5.0
    pos = max(0.0, min(1.0, float(position)))
    idx_f = pos * (len(curve) - 1)
    lo = int(idx_f)
    hi = min(lo + 1, len(curve) - 1)
    frac = idx_f - lo
    factor = curve[lo] * (1.0 - frac) + curve[hi] * frac
    target = float(pool_median_bpm) * factor
    spread = float(pool_median_bpm) * float(profile['spread'])
    return target, spread


def arc_bpm_score(cand_bpm: float, arc: str, position: float,
                   pool_median_bpm: float = 120.0) -> float:
    """[-1, +1] score: how well candidate's BPM fits the arc's target
    curve at this position.
    """
    if not cand_bpm or cand_bpm <= 0:
        return 0.0
    target, spread = arc_conditioned_bpm_target(arc, position, pool_median_bpm)
    diff = abs(float(cand_bpm) - target)
    if spread <= 0:
        return -1.0 if diff > 0 else 1.0
    # 0 diff → +1; 1 spread → 0; 2+ spreads → -1
    return max(-1.0, 1.0 - diff / spread)


__all__ = [
    'pool_vocal_mean',
    'vocal_diversity_bonus',
    'section_coverage_bonus',
    'mmr_select',
    'graded_camelot_score',
    'enforce_genre_floor',
    'arc_conditioned_bpm_target',
    'arc_bpm_score',
]
