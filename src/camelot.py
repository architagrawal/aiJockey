"""Camelot wheel key detection + compatibility logic."""
from __future__ import annotations
import numpy as np

# Krumhansl-Schmuckler key profiles
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Camelot wheel: (pitch_class, mode) -> code
# Pitch class: 0=C, 1=C#, 2=D, ..., 11=B
CAMELOT: dict[tuple[int, str], str] = {
    (0,  'maj'): '8B',  (7,  'maj'): '9B',  (2,  'maj'): '10B',
    (9,  'maj'): '11B', (4,  'maj'): '12B', (11, 'maj'): '1B',
    (6,  'maj'): '2B',  (1,  'maj'): '3B',  (8,  'maj'): '4B',
    (3,  'maj'): '5B',  (10, 'maj'): '6B',  (5,  'maj'): '7B',
    (9,  'min'): '8A',  (4,  'min'): '9A',  (11, 'min'): '10A',
    (6,  'min'): '11A', (1,  'min'): '12A', (8,  'min'): '1A',
    (3,  'min'): '2A',  (10, 'min'): '3A',  (5,  'min'): '4A',
    (0,  'min'): '5A',  (7,  'min'): '6A',  (2,  'min'): '7A',
}

# Reverse: camelot code -> (pitch_class, mode)
CAMELOT_REVERSE: dict[str, tuple[int, str]] = {v: k for k, v in CAMELOT.items()}


def krumhansl_key(chroma_mean: np.ndarray) -> tuple[int, str, float]:
    """Return (pitch_class, mode, confidence) using Krumhansl correlation."""
    scores_maj = [float(np.corrcoef(np.roll(MAJ, i), chroma_mean)[0, 1])
                  for i in range(12)]
    scores_min = [float(np.corrcoef(np.roll(MIN, i), chroma_mean)[0, 1])
                  for i in range(12)]
    best_maj = int(np.argmax(scores_maj))
    best_min = int(np.argmax(scores_min))
    if scores_maj[best_maj] >= scores_min[best_min]:
        return best_maj, 'maj', scores_maj[best_maj]
    return best_min, 'min', scores_min[best_min]


def to_camelot(pitch_class: int, mode: str) -> str:
    """Map (pitch_class, mode) to Camelot code, e.g. '8A'."""
    return CAMELOT.get((pitch_class, mode), '?')


def from_camelot(code: str) -> tuple[int, str] | None:
    """Inverse of to_camelot. None if unknown."""
    return CAMELOT_REVERSE.get(code)


def camelot_distance(a: str, b: str) -> int:
    """
    Distance on Camelot wheel.
    0 = same key. 1 = adjacent number same letter (perfect 5th) or relative key.
    Higher = less compatible.
    """
    if a == '?' or b == '?':
        return 6
    if a == b:
        return 0
    num_a, letter_a = int(a[:-1]), a[-1]
    num_b, letter_b = int(b[:-1]), b[-1]
    if letter_a == letter_b:
        diff = abs(num_a - num_b)
        return min(diff, 12 - diff)
    if num_a == num_b:
        return 1  # relative major/minor
    diff = abs(num_a - num_b)
    return min(diff, 12 - diff) + 1


def semitones_between(src: str, dst: str) -> float:
    """
    Pitch shift (semitones) to move src key to dst key.
    Capped at +-3 semitones to avoid heavy artifacts.
    """
    s = from_camelot(src)
    d = from_camelot(dst)
    if s is None or d is None:
        return 0.0
    diff = (d[0] - s[0]) % 12
    if diff > 6:
        diff -= 12
    return float(np.clip(diff, -3, 3))
