"""Smoke tests for training/features.py and classifier shape."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'training'))

import numpy as np
import torch


def _fake_clip(key='8A', tempo=128.0):
    return {
        'clap': np.random.randn(512).astype(np.float32),
        'key': key,
        'tempo': tempo,
    }


def _fake_seg(seg_type='drop', energy=0.8):
    return {'start': 0.0, 'end': 30.0, 'type': seg_type, 'energy': energy}


def test_feature_dim():
    from features import extract_pair_features, FEATURE_DIM
    f = extract_pair_features(_fake_clip(), _fake_seg(), _fake_clip(), _fake_seg())
    assert f.shape == (FEATURE_DIM,)
    assert f.dtype == np.float32
    assert np.isfinite(f).all()


def test_classifier_forward():
    from classifier import TechniqueClassifier
    from features import FEATURE_DIM, N_TECHNIQUES
    model = TechniqueClassifier()
    x = torch.randn(4, FEATURE_DIM)
    out = model(x)
    assert out.shape == (4, N_TECHNIQUES)
    assert torch.isfinite(out).all()


def test_techniques_match_planner():
    """Ensure TECHNIQUES list mirrors what transitions.py implements."""
    from features import TECHNIQUES
    # Key transitions all present
    for must_have in ('cut', 'crossfade', 'eq_swap', 'silence_drop',
                      'spinback', 'mashup', 'echo_out', 'loop_tighten'):
        assert must_have in TECHNIQUES
