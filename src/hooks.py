"""Hook detection via self-similarity matrix on chroma+MFCC features.

Used by planner for callback scheduling: 'repeat hook from earlier clip later'.
"""
from __future__ import annotations
import numpy as np
import librosa


def detect_hooks(mono: np.ndarray, sr: int, downbeats: list[float],
                 min_bars: int = 4, max_bars: int = 16,
                 sim_threshold: float = 0.7,
                 max_hooks: int = 5) -> list[dict]:
    """
    Find recurring N-bar segments via self-similarity on chroma+MFCC.
    Returns list of {start, end, repetition_count, strength, bars}.
    """
    if len(downbeats) < min_bars * 2:
        return []
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=13)
    feat = np.concatenate([chroma, mfcc], axis=0)
    times = librosa.frames_to_time(np.arange(feat.shape[1]), sr=sr)

    bar_feats: list[np.ndarray] = []
    for i in range(len(downbeats) - 1):
        s, e = downbeats[i], downbeats[i + 1]
        mask = (times >= s) & (times < e)
        if mask.any():
            bar_feats.append(feat[:, mask].mean(axis=1))
    if len(bar_feats) < min_bars * 2:
        return []
    bar_arr = np.asarray(bar_feats)
    norm = bar_arr / (np.linalg.norm(bar_arr, axis=1, keepdims=True) + 1e-8)
    sim = norm @ norm.T

    hooks: list[dict] = []
    used: set[int] = set()
    for L in range(min_bars, max_bars + 1, 4):
        for i in range(len(bar_arr) - L):
            if i in used:
                continue
            sims: list[tuple[int, float]] = []
            for j in range(i + L, len(bar_arr) - L):
                if j in used:
                    continue
                segment_sim = float(sim[i:i + L, j:j + L].diagonal().mean())
                if segment_sim > sim_threshold:
                    sims.append((j, segment_sim))
            if sims:
                hooks.append({
                    'start': float(downbeats[i]),
                    'end': float(downbeats[i + L]),
                    'repetition_count': len(sims) + 1,
                    'strength': float(np.mean([s for _, s in sims])),
                    'bars': L,
                })
                used.update(range(i, i + L))
                for j, _ in sims:
                    used.update(range(j, j + L))
    hooks.sort(key=lambda h: -h['strength'])
    return hooks[:max_hooks]
