"""Smoke tests for procedural FX. Verify shapes, finite values."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
from synth_fx import (
    riser_uplift, downsweep, snare_roll, sub_drop,
    impact, vinyl_stop, airhorn, hihat_roll,
    SYNTHESIZERS, synthesize,
)


SR = 44100


def _check(arr: np.ndarray, expected_n: int | None = None):
    assert arr.ndim == 2 and arr.shape[0] == 2
    assert arr.shape[1] > 0
    assert np.isfinite(arr).all()
    assert np.abs(arr).max() <= 1.01  # allow tiny clip headroom
    if expected_n is not None:
        assert abs(arr.shape[1] - expected_n) <= SR // 10  # within 100ms


def test_riser():
    out = riser_uplift(beats=4, bpm=128)
    _check(out, expected_n=int(4 * 60 / 128 * SR))


def test_downsweep():
    out = downsweep(beats=2, bpm=128)
    _check(out)


def test_snare_roll():
    out = snare_roll(beats=4, bpm=128)
    _check(out)


def test_sub_drop():
    out = sub_drop(beats=2, bpm=128)
    _check(out)


def test_impact():
    out = impact(decay_sec=1.0)
    _check(out, expected_n=SR)


def test_vinyl_stop():
    out = vinyl_stop(duration_sec=0.5)
    _check(out, expected_n=int(0.5 * SR))


def test_airhorn():
    out = airhorn(beats=1, bpm=128)
    _check(out)


def test_hihat_roll():
    out = hihat_roll(beats=2, bpm=128)
    _check(out)


@pytest.mark.parametrize('fx_type', list(SYNTHESIZERS.keys()))
def test_synthesize_dispatch(fx_type):
    out = synthesize(fx_type, bpm=128, beats=2)
    _check(out)


def test_unknown_type():
    assert synthesize('totally_unknown_fx', 128, 1) is None
