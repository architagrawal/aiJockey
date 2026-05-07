"""
Build (anchor, positive, negative) triplets from analyzed clip cache.

No raw audio needed — uses CLAP embeddings already in cache/<id>.npz.

Pair heuristics (synthetic, no real DJ-mix annotations needed):

POSITIVE (compatible) — at least one of:
  - Camelot key distance <= 1 AND BPM diff <= 4%
  - Same key family AND BPM within 5%

NEGATIVE (incompatible) — at least one of:
  - Camelot key distance >= 4
  - BPM diff >= 10%
  - Vastly different energy section types

When real Tier 2 data lands, replace synthetic positives with actual
DJ-set transition pairs.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import random
import numpy as np
from typing import Iterator

from camelot import camelot_distance


def load_clip_cache(cache_dir: str) -> list[dict]:
    """Return list of clips with merged json + clap embedding."""
    out: list[dict] = []
    for jp in sorted(Path(cache_dir).glob('*.json')):
        with open(jp) as f:
            d = json.load(f)
        npz_path = Path(cache_dir) / f"{jp.stem}.npz"
        if not npz_path.exists():
            continue
        d['clap'] = np.load(str(npz_path))['clap'].astype(np.float32)
        d['clip_id'] = jp.stem
        out.append(d)
    return out


def is_compatible(a: dict, b: dict) -> bool:
    """Heuristic: compatible enough for a DJ to mix."""
    ta, tb = a.get('tempo', 0), b.get('tempo', 0)
    if ta <= 0 or tb <= 0:
        return False
    bpm_diff_pct = abs(ta - tb) / max(ta, tb)
    key_dist = camelot_distance(a.get('key', '?'), b.get('key', '?'))
    if bpm_diff_pct <= 0.04 and key_dist <= 1:
        return True
    if bpm_diff_pct <= 0.05 and key_dist == 1:
        return True
    return False


def is_incompatible(a: dict, b: dict) -> bool:
    """Heuristic: bad pair, definitely shouldn't mix."""
    ta, tb = a.get('tempo', 0), b.get('tempo', 0)
    if ta <= 0 or tb <= 0:
        return True
    bpm_diff_pct = abs(ta - tb) / max(ta, tb)
    key_dist = camelot_distance(a.get('key', '?'), b.get('key', '?'))
    return bpm_diff_pct >= 0.10 or key_dist >= 4


def generate_triplets(clips: list[dict],
                      n_anchors: int = 500,
                      n_neg_per_anchor: int = 8,
                      rng_seed: int = 42) -> dict:
    """
    Returns dict with arrays:
        anchor:    (N, 512)
        positive:  (N, 512)
        negatives: (N, K, 512)
        meta:      list of {anchor_id, positive_id, negative_ids}
    where N = number of triplets generated, K = n_neg_per_anchor.
    """
    rng = random.Random(rng_seed)
    if len(clips) < 3:
        raise ValueError(f"need >=3 clips, got {len(clips)}")

    # For tiny clip pools, augment with synthetic perturbations of CLAP vectors
    augment = len(clips) < 50

    anchors_a: list[np.ndarray] = []
    positives_a: list[np.ndarray] = []
    negatives_a: list[np.ndarray] = []
    meta: list[dict] = []

    attempts = 0
    max_attempts = n_anchors * 50
    while len(anchors_a) < n_anchors and attempts < max_attempts:
        attempts += 1
        a = rng.choice(clips)
        # Find compatible candidate
        compat = [c for c in clips if c is not a and is_compatible(a, c)]
        if not compat:
            if augment:
                # Synthetic positive: small perturbation of anchor's CLAP
                pos = a['clap'] + rng.gauss(0, 0.05) * np.random.randn(512).astype(np.float32) * np.linalg.norm(a['clap']) * 0.05
                pos_id = f"{a['clip_id']}__aug"
            else:
                continue
        else:
            p = rng.choice(compat)
            pos = p['clap']
            pos_id = p['clip_id']
        # Find K incompatible candidates
        incompat = [c for c in clips if c is not a and is_incompatible(a, c)]
        if len(incompat) < n_neg_per_anchor:
            if augment:
                # Augment with random other clips
                extras = [c for c in clips if c is not a and c not in incompat]
                rng.shuffle(extras)
                incompat = (incompat + extras)[:n_neg_per_anchor]
            else:
                continue
        rng.shuffle(incompat)
        negs = incompat[:n_neg_per_anchor]
        anchors_a.append(a['clap'])
        positives_a.append(pos)
        negatives_a.append(np.stack([n['clap'] for n in negs]))
        meta.append({
            'anchor_id': a['clip_id'],
            'positive_id': pos_id,
            'negative_ids': [n['clip_id'] for n in negs],
        })

    if not anchors_a:
        raise RuntimeError(
            "no triplets generated — clip pool may be too homogeneous "
            "(all incompatible) or too small. Try lowering compatibility "
            "thresholds or adding more clips.")

    return {
        'anchor': np.stack(anchors_a).astype(np.float32),
        'positive': np.stack(positives_a).astype(np.float32),
        'negatives': np.stack(negatives_a).astype(np.float32),
        'meta': meta,
    }


def save_triplets(triplets: dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        anchor=triplets['anchor'],
        positive=triplets['positive'],
        negatives=triplets['negatives'],
    )
    meta_path = Path(out_path).with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(triplets['meta'], f, indent=2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='cache')
    ap.add_argument('--out', default='datasets/clap_triplets.npz')
    ap.add_argument('--n_anchors', type=int, default=500)
    ap.add_argument('--n_neg', type=int, default=8)
    args = ap.parse_args()
    clips = load_clip_cache(args.cache)
    print(f"loaded {len(clips)} clips from {args.cache}")
    t = generate_triplets(clips, n_anchors=args.n_anchors,
                          n_neg_per_anchor=args.n_neg)
    print(f"generated triplets: anchor {t['anchor'].shape}, "
          f"positive {t['positive'].shape}, negatives {t['negatives'].shape}")
    save_triplets(t, args.out)
    print(f"saved {args.out}")
