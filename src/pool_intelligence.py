"""Pool intelligence — analyze a user-uploaded clip pool, expose its
shape to the Director, optionally sub-curate when too disparate.

Reference: research §1 (genre alignment) + §4 (energy curve) + plan
§14.2 P5 (pool-aware retrieval).

The Director was previously given only `clip_count + prompt`. With a
heterogeneous user pool that produces incoherent mixes. This module
adds:

  - Per-clip tags: genre band, BPM band, dominant section, energy
  - Pool clusters via CLAP centroid + BPM proximity
  - Coherence score: 0..1 = how close clips sit in CLAP space
  - Subset picker: when pool too wide, pick the largest coherent
    cluster + bridges to outliers
  - Pool summary string fed into Director system prompt as a TABLE
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Genre / BPM / section taxonomy
# ---------------------------------------------------------------------------

GENRE_KEYWORDS = {
    'techno':      ('techno', 'tech-house', 'tech_house', 'minimal'),
    'house':       ('house', 'deep_house', 'deep-house', 'tech-house', 'progressive'),
    'trance':      ('trance', 'progressive_trance', 'uplifting'),
    'dnb':         ('dnb', 'drum_and_bass', 'liquid', 'jungle'),
    'dubstep':     ('dubstep', 'brostep', 'riddim'),
    'edm':         ('edm', 'mainstage', 'festival'),
    'future_bass': ('future_bass', 'future-bass', 'futurebass'),
    'chillstep':   ('chillstep', 'chill_step', 'chill-step'),
    'lofi':        ('lofi', 'lo_fi', 'lo-fi', 'lofi-hiphop'),
    'ambient':     ('ambient', 'drone', 'atmosphere'),
    'synthwave':   ('synthwave', 'retrowave', 'outrun'),
    'disco':       ('disco', 'nu_disco', 'nu-disco'),
    'cinematic':   ('cinematic', 'soundtrack', 'orchestral', 'epic'),
    'hip_hop':     ('hiphop', 'hip_hop', 'hip-hop', 'rap', 'trap'),
    'pop':         ('pop', 'edm_pop', 'electropop'),
    'bollywood':   ('bollywood', 'punjabi', 'desi', 'hindi'),
    'classical':   ('classical', 'sitar', 'tabla', 'sarod', 'bansuri', 'hindustani'),
    'rock':        ('rock', 'metal', 'punk', 'indie_rock'),
    'jazz':        ('jazz', 'fusion', 'bebop'),
    'chill':       ('chill', 'downtempo'),
}


def _genre_from_string(s: str) -> str:
    s = (s or '').lower()
    for g, kws in GENRE_KEYWORDS.items():
        for k in kws:
            if k in s:
                return g
    return 'unknown'


def bpm_band(bpm: float) -> str:
    if bpm <= 0:
        return 'unknown'
    if bpm < 90:
        return 'slow_<90'
    if bpm < 110:
        return 'mid_90-110'
    if bpm < 125:
        return 'house_110-125'
    if bpm < 135:
        return 'techno_125-135'
    if bpm < 150:
        return 'fast_135-150'
    return 'very_fast_150+'


def _dominant_section(meta: dict) -> str:
    sections = meta.get('sections') or []
    if not sections:
        return 'unknown'
    types = [s.get('type', '') for s in sections]
    return Counter(types).most_common(1)[0][0] or 'unknown'


def _energy_mean(meta: dict) -> float:
    sections = meta.get('sections') or []
    if not sections:
        return 0.5
    energies = [float(s.get('energy', 0.5)) for s in sections]
    return float(np.mean(energies)) if energies else 0.5


# ---------------------------------------------------------------------------
# Per-clip tag extraction
# ---------------------------------------------------------------------------

def tag_clip(meta: dict) -> dict:
    """Return {clip_id, source, genre, bpm, bpm_band, key, section_top, energy, duration, has_vocals}.

    `source` reads from meta['source'] (= 'user' | 'library'); defaults
    to 'unknown' so legacy cache JSONs without the field still tag.
    """
    cid = meta.get('clip_id') or ''
    name_genre = _genre_from_string(cid)
    bpm = float(meta.get('tempo', 0.0)) or 0.0

    # Vocals heuristic: vocal_activity averaged across sections > 0.3 = has vocals
    has_vocals = False
    sections = meta.get('sections') or []
    if sections:
        va = [float(s.get('vocal_activity', 0.0)) for s in sections]
        if va and float(np.mean(va)) > 0.3:
            has_vocals = True

    return {
        'clip_id':      cid,
        'source':       (meta.get('source') or 'unknown').lower(),
        'genre':        name_genre,
        'bpm':          round(bpm, 1),
        'bpm_band':     bpm_band(bpm),
        'key':          meta.get('key', '?'),
        'section_top':  _dominant_section(meta),
        'energy':       round(_energy_mean(meta), 2),
        'duration':     round(float(meta.get('duration', 0.0)), 1),
        'has_vocals':   has_vocals,
    }


# ---------------------------------------------------------------------------
# Coherence + clustering
# ---------------------------------------------------------------------------

def _clap_centroid(clips: dict[str, dict]) -> np.ndarray | None:
    arrs: list[np.ndarray] = []
    for cm in clips.values():
        v = cm.get('clap_embedding')
        if v is None:
            continue
        a = np.asarray(v, dtype=np.float32).reshape(-1)
        if a.size:
            arrs.append(a)
    if not arrs:
        return None
    return np.mean(np.stack(arrs), axis=0)


def coherence_score(clips: dict[str, dict]) -> float:
    """Compute pool coherence in [0, 1]. 1 = all clips at same CLAP centroid;
    0 = clips orthogonal in CLAP space.
    """
    centroid = _clap_centroid(clips)
    if centroid is None:
        return 0.0
    cnorm = np.linalg.norm(centroid) + 1e-9
    sims = []
    for cm in clips.values():
        v = cm.get('clap_embedding')
        if v is None:
            continue
        a = np.asarray(v, dtype=np.float32).reshape(-1)
        if a.size == 0:
            continue
        sim = float((a @ centroid) / (np.linalg.norm(a) * cnorm + 1e-9))
        sims.append(sim)
    if not sims:
        return 0.0
    # Coherence = mean cosine similarity, mapped to [0, 1]
    return max(0.0, min(1.0, (np.mean(sims) + 1.0) / 2.0))


def cluster_pool(clips: dict[str, dict],
                  bpm_tol_pct: float = 8.0) -> list[dict]:
    """Cluster clips by (genre, BPM band) tag combo. Returns list of
    {tag, members, count, bpm_mean, energy_mean} sorted by size desc.
    """
    by_key: dict[tuple[str, str], list[str]] = {}
    for cid, cm in clips.items():
        t = tag_clip(cm)
        key = (t['genre'], t['bpm_band'])
        by_key.setdefault(key, []).append(cid)
    clusters = []
    for (genre, band), members in by_key.items():
        bpms = [float(clips[c].get('tempo', 0)) for c in members]
        energies = [_energy_mean(clips[c]) for c in members]
        clusters.append({
            'tag':         f'{genre}/{band}',
            'genre':       genre,
            'bpm_band':    band,
            'count':       len(members),
            'members':     members,
            'bpm_mean':    round(float(np.mean(bpms)), 1) if bpms else 0.0,
            'energy_mean': round(float(np.mean(energies)), 2) if energies else 0.5,
        })
    clusters.sort(key=lambda c: -c['count'])
    return clusters


# ---------------------------------------------------------------------------
# Subset picker — when pool too wide for one mix
# ---------------------------------------------------------------------------

def pick_coherent_subset(clips: dict[str, dict],
                          min_subset_size: int = 5,
                          target_size: int = 12) -> tuple[dict[str, dict], dict]:
    """Pick largest coherent cluster (or top 2 cluster union) of clips.
    Returns (subset_clips, info_dict).

    If main cluster has < min_subset_size, falls back to keeping all clips
    with a low-coherence warning in info.
    """
    clusters = cluster_pool(clips)
    if not clusters:
        return clips, {'note': 'no clusters', 'clusters': []}

    primary = clusters[0]
    if primary['count'] < min_subset_size:
        return clips, {
            'note': f'no coherent cluster (largest={primary["count"]} clips); '
                    f'using full pool with low-coherence warning',
            'clusters': clusters[:5],
            'coherence_warning': True,
        }

    members = list(primary['members'])
    used_clusters = [primary['tag']]

    # Fold in adjacent clusters of same genre with similar BPM (within ±8%)
    for c in clusters[1:]:
        if len(members) >= target_size:
            break
        if c['genre'] == primary['genre']:
            members.extend(c['members'])
            used_clusters.append(c['tag'])
            continue
        if (abs(c['bpm_mean'] - primary['bpm_mean'])
                / max(primary['bpm_mean'], 1) < 0.08):
            members.extend(c['members'])
            used_clusters.append(c['tag'])

    members = members[:target_size]
    subset = {cid: clips[cid] for cid in members}
    return subset, {
        'note':     f'picked {len(subset)}/{len(clips)} clips from clusters {used_clusters}',
        'clusters': clusters[:5],
        'used':     used_clusters,
        'excluded': [c for c in clips if c not in subset],
    }


# ---------------------------------------------------------------------------
# Pool summary string for Director system-prompt injection
# ---------------------------------------------------------------------------

def summary_table(clips: dict[str, dict], max_rows: int = 16) -> str:
    """Render pool as a markdown-style table for Director consumption.

    Director sees:

        | id | genre | bpm | section | energy | vocals | dur |
        | ...

    Plus footer line with cluster summary + coherence.
    """
    if not clips:
        return '(empty pool)'
    rows = [tag_clip(cm) for cm in clips.values()]
    rows.sort(key=lambda r: (r['bpm_band'], r['genre'], -r['energy']))

    lines = ['Pool inventory (what you have to mix with). USER = user-uploaded; LIB = library augmentation:']
    lines.append('| # | source | genre | bpm | section | energy | vox | dur |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for i, r in enumerate(rows[:max_rows]):
        vox = 'V' if r['has_vocals'] else '-'
        src = (r.get('source') or 'unknown').upper()[:4]
        lines.append(
            f"| {i} | {src:6s} | {r['genre']:10s} | {r['bpm']:5.1f} | "
            f"{r['section_top']:10s} | {r['energy']:.2f} | {vox} | {r['duration']:.0f}s |"
        )
    if len(rows) > max_rows:
        lines.append(f'| ... | (+{len(rows)-max_rows} more) | | | | | | |')

    clusters = cluster_pool(clips)
    coh = coherence_score(clips)
    lines.append('')
    lines.append(f'Pool coherence: {coh:.2f} (1.0=tight, 0=disparate)')
    lines.append('Clusters (genre/bpm-band: count):')
    for c in clusters[:6]:
        lines.append(f'  - {c["tag"]}: {c["count"]} clips, bpm~{c["bpm_mean"]}, '
                     f'energy~{c["energy_mean"]}')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Honest pool diagnostic
# ---------------------------------------------------------------------------

def diagnose(clips: dict[str, dict]) -> dict:
    """Return diagnostic dict that planner/api can persist as metadata.

    Use this to expose POOL LIMITATIONS honestly to user.
    """
    if not clips:
        return {'verdict': 'empty', 'coherence': 0.0}
    coh = coherence_score(clips)
    clusters = cluster_pool(clips)
    bpms = [float(cm.get('tempo', 0)) for cm in clips.values()
            if cm.get('tempo')]
    bpm_min = min(bpms) if bpms else 0
    bpm_max = max(bpms) if bpms else 0
    bpm_spread = (bpm_max - bpm_min) / max(np.mean(bpms), 1) if bpms else 0
    genres = {tag_clip(cm)['genre'] for cm in clips.values()}

    if coh > 0.85 and len(clusters[0]['members']) >= 0.6 * len(clips):
        verdict = 'tight'
        narrative = 'pool is coherent — single genre/bpm cluster, can build narrative arc'
    elif coh > 0.65 and len(clusters[:3]) >= 3:
        verdict = 'mixed_navigable'
        narrative = ('pool has 2-3 distinct clusters — DJ-style multi-genre journey '
                     'feasible if narrative carefully bridged')
    else:
        verdict = 'disparate'
        narrative = ('pool too disparate for single-narrative mix — recommend '
                     'curating to a coherent subset, or expect genre-jumps to '
                     'feel abrupt')

    return {
        'verdict':           verdict,
        'narrative_advice':  narrative,
        'coherence':         round(coh, 3),
        'n_clips':           len(clips),
        'n_genres':          len(genres),
        'genres':            sorted(genres),
        'bpm_spread_pct':    round(bpm_spread * 100, 1),
        'bpm_range':         (round(bpm_min, 1), round(bpm_max, 1)),
        'top_clusters':      [{k: c[k] for k in ('tag', 'count', 'bpm_mean')}
                              for c in clusters[:5]],
    }


__all__ = [
    'tag_clip', 'bpm_band', 'coherence_score', 'cluster_pool',
    'pick_coherent_subset', 'summary_table', 'diagnose',
]
