"""
Style-RAG: retrieve transition patterns from curated reference mixes.

Reference index format: directory of reference timelines (JSON), each describing
a known-good DJ set. We extract per-transition (out_genre_emb, in_genre_emb,
energy_delta, technique) tuples, build CLAP-conditioned vector index, then
retrieve top-K matching patterns at planning time and bias technique selection.

For MVP: simple numpy cosine search, no faiss dependency.

Reference timeline format:
{
  "name": "Solomun Sunset Set 2023",
  "transitions": [
    {
      "out_clap": [512 floats],
      "in_clap":  [512 floats],
      "out_energy": 0.7,
      "in_energy": 0.85,
      "technique": "eq_swap",
      "bars": 32
    },
    ...
  ]
}

How to build refs (manual step or scripted):
1. Analyze a curated DJ set (or its constituent tracks)
2. Note each transition: which technique, energy levels, CLAP of clips
3. Save as JSON in references/<name>.json
"""
from __future__ import annotations
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class TransitionPattern:
    out_clap: np.ndarray   # (512,)
    in_clap: np.ndarray    # (512,)
    out_energy: float
    in_energy: float
    technique: str
    bars: int
    source: str = ''


class StyleRAG:
    def __init__(self, ref_dir: str = 'references'):
        self.patterns: list[TransitionPattern] = []
        self.ref_dir = Path(ref_dir)
        if self.ref_dir.exists():
            self._load()

    def _load(self) -> None:
        for jp in sorted(self.ref_dir.glob('*.json')):
            try:
                with open(jp) as f:
                    d = json.load(f)
                for t in d.get('transitions', []):
                    self.patterns.append(TransitionPattern(
                        out_clap=np.asarray(t['out_clap'], dtype=np.float32),
                        in_clap=np.asarray(t['in_clap'], dtype=np.float32),
                        out_energy=float(t.get('out_energy', 0.5)),
                        in_energy=float(t.get('in_energy', 0.5)),
                        technique=str(t.get('technique', 'crossfade')),
                        bars=int(t.get('bars', 16)),
                        source=str(jp.stem),
                    ))
            except Exception as e:
                print(f"warn: failed to load {jp}: {e}")

    def __len__(self) -> int:
        return len(self.patterns)

    def query(self, out_clap: np.ndarray, in_clap: np.ndarray,
              out_energy: float, in_energy: float,
              top_k: int = 5,
              clap_weight: float = 0.6,
              energy_weight: float = 0.4) -> list[TransitionPattern]:
        """Return top-K patterns most similar to (out, in) context."""
        if not self.patterns:
            return []
        out_q = self._norm(out_clap)
        in_q = self._norm(in_clap)
        scores: list[tuple[float, TransitionPattern]] = []
        for p in self.patterns:
            sim_out = float(out_q @ self._norm(p.out_clap))
            sim_in = float(in_q @ self._norm(p.in_clap))
            clap_sim = (sim_out + sim_in) / 2.0
            energy_dist = (abs(p.out_energy - out_energy)
                           + abs(p.in_energy - in_energy)) / 2.0
            energy_sim = max(0.0, 1.0 - energy_dist)
            score = clap_weight * clap_sim + energy_weight * energy_sim
            scores.append((score, p))
        scores.sort(key=lambda x: -x[0])
        return [p for _, p in scores[:top_k]]

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def technique_bias(self, retrieved: list[TransitionPattern]) -> dict[str, float]:
        """
        Build per-technique bonus from retrieved patterns.
        Returns {technique_name: bonus_score} where bonus = frequency / total.
        """
        if not retrieved:
            return {}
        c = Counter(p.technique for p in retrieved)
        total = sum(c.values())
        return {tech: count / total for tech, count in c.items()}


def build_pattern_from_clips(out_clip_meta: dict, out_clap: np.ndarray,
                             out_energy: float, in_clip_meta: dict,
                             in_clap: np.ndarray, in_energy: float,
                             technique: str, bars: int = 16) -> dict:
    """Helper to construct a transition pattern dict for saving as reference."""
    return {
        'out_clap': out_clap.tolist(),
        'in_clap': in_clap.tolist(),
        'out_energy': float(out_energy),
        'in_energy': float(in_energy),
        'technique': technique,
        'bars': int(bars),
        'out_clip': out_clip_meta.get('clip_id', '?'),
        'in_clip': in_clip_meta.get('clip_id', '?'),
    }
