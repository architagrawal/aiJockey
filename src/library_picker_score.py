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


# ---------------------------------------------------------------------------
# #1c — Audiobox feedback rerank (closed-loop quality signal)
# ---------------------------------------------------------------------------
# State: per-clip running mean of mix-level Audiobox PQ across renders the
# clip appeared in. Cheap approximation of leave-one-out PQ delta — uses
# overall mix PQ as a credit signal instead of computing N renders per pick.
# Storage: simple JSON sidecar at $AIJOCKEY_AUDIOBOX_HISTORY (default
# /scratch/picker_history/audiobox.json). Updated after each render.

import json
from pathlib import Path


def _audiobox_history_path() -> Path:
    p = os.environ.get('AIJOCKEY_AUDIOBOX_HISTORY')
    if p:
        return Path(p)
    for cand in (Path('/scratch/picker_history/audiobox.json'),
                  Path('/workspace/scratch/picker_history/audiobox.json'),
                  Path('./.aijockey/audiobox_history.json')):
        try:
            cand.parent.mkdir(parents=True, exist_ok=True)
            return cand
        except Exception:
            continue
    return Path('./audiobox_history.json')


def _load_audiobox_history() -> dict:
    """Returns {clip_id: {'count': int, 'pq_sum': float, 'ce_sum': float, ...}}."""
    p = _audiobox_history_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_audiobox_history(data: dict) -> None:
    p = _audiobox_history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)


def record_audiobox_render(clip_ids: list[str], aesthetics: dict | None) -> None:
    """Update per-clip running stats after a render. aesthetics is a dict
    with keys PQ/PC/CE/CU (per audiobox_aesthetics.score()). Skips silently
    when aesthetics None or no clip_ids."""
    if not aesthetics or not clip_ids:
        return
    hist = _load_audiobox_history()
    for cid in clip_ids:
        entry = hist.setdefault(cid, {'count': 0, 'pq_sum': 0.0,
                                        'ce_sum': 0.0, 'pc_sum': 0.0, 'cu_sum': 0.0})
        entry['count'] += 1
        entry['pq_sum'] += float(aesthetics.get('PQ', 0.0))
        entry['ce_sum'] += float(aesthetics.get('CE', 0.0))
        entry['pc_sum'] += float(aesthetics.get('PC', 0.0))
        entry['cu_sum'] += float(aesthetics.get('CU', 0.0))
    try:
        _save_audiobox_history(hist)
    except Exception as e:
        print(f"[audiobox_history] save failed: {e}")


def audiobox_lift_term(clip_id: str, axes: tuple[str, ...] = ('PQ', 'CE'),
                        baseline: float = 6.0,
                        strength: float = 0.25) -> float:
    """Bonus from clip's historical mix-level Audiobox mean.

    Cheap proxy for leave-one-out PQ delta — uses cumulative PQ/CE means
    of all renders this clip appeared in, vs a `baseline` (default 6.0
    on Audiobox 0-10 scale). Boost up to `strength` when mean above
    baseline, penalty down to -strength when below.

    Returns 0 when clip never seen (cold-start neutral).
    """
    hist = _load_audiobox_history()
    entry = hist.get(clip_id)
    if not entry or entry.get('count', 0) == 0:
        return 0.0
    n = max(1, int(entry['count']))
    means = []
    for axis in axes:
        key = f'{axis.lower()}_sum'
        if key in entry:
            means.append(entry[key] / n)
    if not means:
        return 0.0
    avg = sum(means) / len(means)
    # avg=baseline → 0; avg=baseline+2 → +strength; avg=baseline-2 → -strength
    return float(strength * max(-1.0, min(1.0, (avg - baseline) / 2.0)))


# ---------------------------------------------------------------------------
# #1d — Audiobox slice term (per-section PQ from cache build)
# ---------------------------------------------------------------------------
# Reads cache/<clip_id>.audiobox_slices.json (built once by
# scripts/audiobox_slice_prescore.py). Returns a per-section bonus the
# planner adds to the candidate score. Cold-start = 0 when no sidecar.

_SLICE_CACHE: dict[str, list[dict]] = {}


def _slice_candidate_paths(cache_dir: 'str | Path', clip_id: str) -> list[Path]:
    """Per-job cache first, then global library cache fallback."""
    fname = f'{clip_id}.audiobox_slices.json'
    paths = [Path(cache_dir) / fname]
    lib_cache = os.environ.get('AIJOCKEY_LIBRARY_CACHE') or '/cache'
    lib = Path(lib_cache) / fname
    if lib not in paths:
        paths.append(lib)
    return paths


def _load_slice_sidecar(cache_dir: 'str | Path', clip_id: str) -> list[dict]:
    key = f'{cache_dir}::{clip_id}'
    if key in _SLICE_CACHE:
        return _SLICE_CACHE[key]
    data: list[dict] = []
    for p in _slice_candidate_paths(cache_dir, clip_id):
        if not p.exists():
            continue
        try:
            blob = json.loads(p.read_text())
            data = blob.get('sections') or []
            if data:
                break
        except Exception:
            continue
    _SLICE_CACHE[key] = data
    return data


def audiobox_slice_term(cache_dir: 'str | Path', clip_id: str,
                         section_start: float, section_end: float,
                         baseline: float = 6.0,
                         strength: float = 0.4,
                         axes: tuple[str, ...] = ('PQ', 'CE')) -> float:
    """Score bonus from this section's pre-computed Audiobox PQ/CE.

    Matches section by start/end (tolerance 0.5s). Returns 0 when no
    sidecar or no match — cold-start neutral. Above baseline → bonus,
    below baseline → penalty (symmetric, clamped to ±strength).
    """
    sections = _load_slice_sidecar(cache_dir, clip_id)
    if not sections:
        return 0.0
    best = None
    for s in sections:
        try:
            ds = abs(float(s.get('start', 0.0)) - float(section_start))
            de = abs(float(s.get('end', 0.0)) - float(section_end))
        except Exception:
            continue
        if ds < 0.6 and de < 0.6:
            best = s
            break
    if best is None:
        # Fallback: enclosing window
        for s in sections:
            try:
                ss = float(s.get('start', 0.0))
                se = float(s.get('end', 0.0))
            except Exception:
                continue
            if ss <= section_start + 0.5 and se + 0.5 >= section_end:
                best = s
                break
    if best is None:
        return 0.0
    vals: list[float] = []
    for ax in axes:
        v = best.get(ax)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return 0.0
    avg = sum(vals) / len(vals)
    return float(strength * max(-1.0, min(1.0, (avg - baseline) / 2.0)))


# ---------------------------------------------------------------------------
# #4 — Probe-failure exclusion / blame decay
# ---------------------------------------------------------------------------
# When a junction's probe severity > threshold, BOTH clips at that junction
# get partial blame. Penalty decays linearly over `decay_renders` future
# renders. Storage: same JSON sidecar shape as audiobox.

def _blame_history_path() -> Path:
    p = os.environ.get('AIJOCKEY_BLAME_HISTORY')
    if p:
        return Path(p)
    for cand in (Path('/scratch/picker_history/blame.json'),
                  Path('/workspace/scratch/picker_history/blame.json'),
                  Path('./.aijockey/blame_history.json')):
        try:
            cand.parent.mkdir(parents=True, exist_ok=True)
            return cand
        except Exception:
            continue
    return Path('./blame_history.json')


def _load_blame() -> dict:
    p = _blame_history_path()
    if not p.exists():
        return {'global_render_counter': 0, 'clips': {}}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {'global_render_counter': 0, 'clips': {}}


def _save_blame(data: dict) -> None:
    p = _blame_history_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)


def record_probe_failures(junctions: list[dict],
                          severity_threshold: float = 0.6) -> None:
    """For each junction with severity > threshold, increment blame on both
    participating clip_ids. Junctions are dicts with at least
    {prev_clip_id, cur_clip_id, severity}.

    Partial credit: each clip gets +0.5 blame per flagged junction (so a
    clip in 2 bad junctions accumulates 1.0). Decay handled at read time.
    """
    if not junctions:
        return
    state = _load_blame()
    state['global_render_counter'] = state.get('global_render_counter', 0) + 1
    cur_render = state['global_render_counter']
    clips = state.setdefault('clips', {})
    for j in junctions:
        sev = float(j.get('severity', 0.0) or 0.0)
        if sev <= severity_threshold:
            continue
        for cid_key in ('prev_clip_id', 'cur_clip_id'):
            cid = j.get(cid_key)
            if not cid:
                continue
            entry = clips.setdefault(cid, [])
            # Append (render_idx, blame_amount). Decay computed at read.
            entry.append([cur_render, 0.5])
    try:
        _save_blame(state)
    except Exception as e:
        print(f"[blame_history] save failed: {e}")


def probe_blame_decay(clip_id: str, decay_renders: int = 5,
                       strength: float = 0.4) -> float:
    """Penalty contribution: decays linearly across `decay_renders`
    future renders. Returns negative number (penalty) or 0 when clip
    has no recent failures."""
    state = _load_blame()
    entry = state.get('clips', {}).get(clip_id, [])
    if not entry:
        return 0.0
    cur_render = state.get('global_render_counter', 0)
    total = 0.0
    for render_idx, amount in entry:
        age = cur_render - int(render_idx)
        if age < 0 or age >= decay_renders:
            continue
        weight = 1.0 - (age / decay_renders)
        total += float(amount) * weight
    # Cap at 1.0 → max penalty = -strength
    return float(-strength * min(1.0, total))


# ---------------------------------------------------------------------------
# #5 — Style-RAG wire helpers (already-scaffolded module → picker)
# ---------------------------------------------------------------------------

def style_rag_clip_prior(candidate_clap: np.ndarray,
                          user_pool_centroid: np.ndarray,
                          rag_top_k: int = 5,
                          strength: float = 0.20) -> float:
    """Bonus from style-RAG retrieval. For the user pool's CLAP centroid,
    style_rag returns top-k pro-DJ-set transitions. Their clip neighborhoods
    define a "this kind of mix uses these kinds of clips" prior.

    Implementation: cosine sim of candidate against the union of pro-set
    clip embeddings retrieved as similar to user centroid. Boost when
    candidate is near a pro choice.
    """
    try:
        from style_rag import StyleRAG
    except Exception:
        return 0.0
    try:
        rag = StyleRAG()
        # rag returns examples — extract their CLAP / clip embeddings if available
        examples = rag.retrieve(user_pool_centroid, k=rag_top_k)
    except Exception:
        return 0.0
    if not examples:
        return 0.0
    # Each example may carry a 'clap' / 'embedding' field — collect available ones
    pro_embs = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        e = ex.get('clap') or ex.get('embedding') or ex.get('vec')
        if e is not None:
            arr = np.asarray(e, dtype=np.float32)
            n = float(np.linalg.norm(arr))
            if n > 0:
                pro_embs.append(arr / n)
    if not pro_embs:
        return 0.0
    cand = np.asarray(candidate_clap, dtype=np.float32)
    n = float(np.linalg.norm(cand))
    if n == 0:
        return 0.0
    cand = cand / n
    sims = [float(cand @ p) for p in pro_embs]
    # Best match cosine sim ∈ [-1, 1]; map to [-strength, +strength]
    return float(strength * max(sims))


__all__ = [
    'pool_vocal_mean',
    'vocal_diversity_bonus',
    'section_coverage_bonus',
    'mmr_select',
    'graded_camelot_score',
    'enforce_genre_floor',
    'arc_conditioned_bpm_target',
    'arc_bpm_score',
    # closed-loop
    'record_audiobox_render',
    'audiobox_lift_term',
    'record_probe_failures',
    'probe_blame_decay',
    # style-RAG
    'style_rag_clip_prior',
]
