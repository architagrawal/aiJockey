"""Unit tests for phrase module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from phrase import snap_to_phrase, detect_phrase_length


def test_snap_basic():
    downbeats = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
                 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
    # 16-bar phrase: phrase boundaries at downbeats[0], downbeats[16]
    # = 0.0 and 32.0
    assert snap_to_phrase(15.0, downbeats, bars_per_phrase=16) == 0.0
    assert snap_to_phrase(20.0, downbeats, bars_per_phrase=16) == 32.0


def test_snap_empty():
    assert snap_to_phrase(5.0, []) == 5.0


def test_detect_default_short():
    # Too few bars -> returns 16
    assert detect_phrase_length([0.0, 2.0], [0.5, 0.5]) == 16
