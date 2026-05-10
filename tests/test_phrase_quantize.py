"""Phrase quantization snap correctness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phrase import snap_to_phrase


def test_snap_to_phrase_returns_existing_boundary():
    # 32 downbeats every 0.5s = 16s of bars at 120 BPM
    downbeats = [i * 0.5 for i in range(32)]
    # Snap to 8-bar phrase: phrase boundaries every 4s (8 bars * 0.5s)
    snapped = snap_to_phrase(3.7, downbeats, bars_per_phrase=8)
    assert snapped == 4.0


def test_snap_to_phrase_empty_downbeats_passthrough():
    assert snap_to_phrase(5.0, [], bars_per_phrase=8) == 5.0


def test_snap_to_phrase_picks_nearest():
    downbeats = [0.0, 4.0, 8.0, 12.0, 16.0]
    # Snap to 16-bar = phrase every 16 downbeats which we don't have, so
    # uses downbeats[::16] = [0.0]; nearest to 10.0 is 0.0
    snapped = snap_to_phrase(10.0, downbeats, bars_per_phrase=16)
    assert snapped == 0.0
    # 4-bar phrasing: phrase boundaries are downbeats[::4] = [0.0, 16.0]
    snapped = snap_to_phrase(10.0, downbeats, bars_per_phrase=4)
    assert snapped in (0.0, 16.0)
