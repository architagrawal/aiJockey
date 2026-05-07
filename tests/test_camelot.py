"""Unit tests for camelot module. Run: pytest tests/"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from camelot import (
    krumhansl_key, to_camelot, from_camelot,
    camelot_distance, semitones_between, CAMELOT,
)


def test_camelot_roundtrip():
    for (pc, mode), code in CAMELOT.items():
        assert from_camelot(code) == (pc, mode)


def test_distance_same():
    assert camelot_distance('8A', '8A') == 0


def test_distance_relative():
    # 8A and 8B = relative minor/major, distance 1
    assert camelot_distance('8A', '8B') == 1


def test_distance_adjacent():
    # 8A and 9A = perfect 5th, distance 1
    assert camelot_distance('8A', '9A') == 1
    assert camelot_distance('8A', '7A') == 1


def test_distance_unknown():
    assert camelot_distance('?', '8A') == 6


def test_semitones_clipped():
    # Far keys should clip to ±3 semitones
    s = semitones_between('1A', '7A')
    assert -3 <= s <= 3


def test_krumhansl_c_major():
    # C major chroma should detect C major (pitch class 0, mode 'maj')
    chroma = np.zeros(12)
    chroma[[0, 4, 7]] = 1.0  # C-E-G triad
    pc, mode, conf = krumhansl_key(chroma)
    assert mode == 'maj'
    assert pc == 0
    assert conf > 0
