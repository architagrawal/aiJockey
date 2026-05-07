"""
Feature extraction for transition classifier.

Input: outgoing clip + segment, incoming clip + segment.
Output: feature vector (numpy float32).

Feature schema (~1049 dim):
- prev_clap (512)        — outgoing CLAP embedding
- cand_clap (512)        — incoming CLAP embedding
- clap_cosine (1)        — cosine similarity
- clap_diff_norm (1)     — L2 norm of difference
- tempo_a, tempo_b, tempo_diff_pct (3)
- key_dist (1)           — Camelot wheel distance
- energy_a, energy_b, energy_diff (3)
- section_a_onehot (10)
- section_b_onehot (10)
"""
from __future__ import annotations
import sys
from pathlib import Path

# Allow sibling imports when run as `python src/training/features.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from camelot import camelot_distance


SECTION_TYPES = [
    'intro', 'build', 'drop', 'verse', 'breakdown',
    'chorus', 'bridge', 'outro', 'callback', 'unknown',
]
SECTION_INDEX = {t: i for i, t in enumerate(SECTION_TYPES)}

FEATURE_DIM = 512 + 512 + 1 + 1 + 3 + 1 + 3 + len(SECTION_TYPES) * 2  # = 1051


def _section_onehot(section_type: str) -> np.ndarray:
    v = np.zeros(len(SECTION_TYPES), dtype=np.float32)
    v[SECTION_INDEX.get(section_type, SECTION_INDEX['unknown'])] = 1.0
    return v


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def extract_pair_features(prev_clip: dict, prev_seg: dict,
                          cand_clip: dict, cand_seg: dict) -> np.ndarray:
    """Build feature vector for a transition (prev -> cand)."""
    prev_clap = np.asarray(prev_clip.get('clap', np.zeros(512)), dtype=np.float32)
    cand_clap = np.asarray(cand_clip.get('clap', np.zeros(512)), dtype=np.float32)
    if prev_clap.size != 512:
        prev_clap = np.zeros(512, dtype=np.float32)
    if cand_clap.size != 512:
        cand_clap = np.zeros(512, dtype=np.float32)

    a = _norm(prev_clap)
    b = _norm(cand_clap)
    clap_cosine = float(a @ b)
    clap_diff_norm = float(np.linalg.norm(prev_clap - cand_clap))

    tempo_a = float(prev_clip.get('tempo', 0.0))
    tempo_b = float(cand_clip.get('tempo', 0.0))
    tempo_diff_pct = abs(tempo_a - tempo_b) / max(tempo_a, 1.0)

    key_dist = float(camelot_distance(
        prev_clip.get('key', '?'), cand_clip.get('key', '?'),
    ))

    energy_a = float(prev_seg.get('energy', 0.5))
    energy_b = float(cand_seg.get('energy', 0.5))
    energy_diff = energy_b - energy_a

    sect_a = _section_onehot(prev_seg.get('type', 'unknown'))
    sect_b = _section_onehot(cand_seg.get('type', 'unknown'))

    return np.concatenate([
        prev_clap,                                            # 512
        cand_clap,                                            # 512
        np.array([clap_cosine, clap_diff_norm,
                  tempo_a, tempo_b, tempo_diff_pct,
                  key_dist,
                  energy_a, energy_b, energy_diff],
                 dtype=np.float32),                           # 9
        sect_a,                                               # 10
        sect_b,                                               # 10
    ]).astype(np.float32)


# Canonical transition technique vocabulary (matches transitions.py)
TECHNIQUES = [
    'cut', 'crossfade', 'eq_swap', 'filter_fade', 'silence_drop',
    'drum_break', 'mashup', 'stem_swap', 'echo_out', 'spinback',
    'pitch_bend', 'loop_tighten', 'scratch_fill', 'loop_callback', 'fade_in',
]
TECHNIQUE_INDEX = {t: i for i, t in enumerate(TECHNIQUES)}
N_TECHNIQUES = len(TECHNIQUES)
