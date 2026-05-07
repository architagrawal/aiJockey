"""Unit tests for transition primitives. Numerical only — no audio listening."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from transitions import (
    equal_power_xfade, lp_filter, hp_filter,
    cut_transition, crossfade_transition, eq_swap_transition,
    filter_fade_transition, silence_drop_transition,
    spinback_transition, loop_tighten_transition, loop_callback,
    riser_bridge,
)

SR = 44100


def make_stereo(seconds: float = 1.0, freq: float = 440.0) -> np.ndarray:
    n = int(seconds * SR)
    t = np.linspace(0, seconds, n, endpoint=False)
    sig = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    return np.stack([sig, sig])


def test_equal_power_xfade_length():
    a = make_stereo(1.0, 440)
    b = make_stereo(1.0, 880)
    n = SR // 2
    out = equal_power_xfade(a, b, n)
    assert out.shape[1] == a.shape[1] + b.shape[1] - n


def test_equal_power_xfade_zero_overlap():
    a = make_stereo(0.5)
    b = make_stereo(0.5)
    out = equal_power_xfade(a, b, 0)
    assert out.shape[1] == a.shape[1] + b.shape[1]


def test_cut():
    a = make_stereo(0.5)
    b = make_stereo(0.5)
    out = cut_transition(a, b)
    assert out.shape[1] == a.shape[1] + b.shape[1]


def test_crossfade():
    a = make_stereo(2.0)
    b = make_stereo(2.0)
    out = crossfade_transition(a, b, SR, bars=2, beat_dur=0.1)  # 0.8s xfade
    assert out.shape[1] > 0
    assert not np.isnan(out).any()


def test_eq_swap():
    a = make_stereo(2.0, 100)
    b = make_stereo(2.0, 800)
    out = eq_swap_transition(a, b, SR, bars=2, beat_dur=0.1)
    assert out.shape[1] > 0
    assert not np.isnan(out).any()


def test_filter_fade():
    a = make_stereo(2.0, 100)
    b = make_stereo(2.0, 800)
    out = filter_fade_transition(a, b, SR, bars=2, beat_dur=0.1)
    assert out.shape[1] > 0


def test_silence_drop():
    a = make_stereo(0.5)
    b = make_stereo(0.5)
    out = silence_drop_transition(a, b, SR, silence_beats=2, beat_dur=0.5)
    silence_n = int(2 * 0.5 * SR)
    assert out.shape[1] == a.shape[1] + silence_n + b.shape[1]


def test_spinback():
    a = make_stereo(2.0)
    b = make_stereo(0.5)
    out = spinback_transition(a, b, SR, spinback_beats=2, beat_dur=0.5)
    assert out.shape[1] > 0


def test_loop_tighten():
    a = make_stereo(2.0)
    b = make_stereo(0.5)
    out = loop_tighten_transition(a, b, SR, beat_dur=0.5, start_bars=1)
    assert out.shape[1] > 0


def test_loop_callback():
    hook = make_stereo(0.25)
    out = loop_callback(hook, repetitions=3)
    assert out.shape[1] == hook.shape[1] * 3


def test_riser_bridge():
    out = riser_bridge(0.5, SR)
    assert out.shape == (2, int(0.5 * SR))
