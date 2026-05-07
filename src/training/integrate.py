"""
Glue: load trained classifier, expose as `pick_technique()` function the
planner can call instead of the hand-coded decision tree.

Wire-up: planner.transition_score() can optionally take a model_ckpt path.
If provided, technique selection comes from the classifier; scoring weights
still drive ordering between candidates.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional
import numpy as np

from features import extract_pair_features, TECHNIQUES, FEATURE_DIM


_MODEL = None
_DEVICE = None


def load_model(ckpt_path: str, device: str = 'auto'):
    """Lazy-load classifier. Returns torch model in eval mode."""
    global _MODEL, _DEVICE
    if _MODEL is not None and ckpt_path == getattr(_MODEL, '_ckpt', None):
        return _MODEL
    import torch
    from classifier import TechniqueClassifier
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _DEVICE = device
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TechniqueClassifier(state['feature_dim'], state['n_classes']).to(device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    model._ckpt = ckpt_path  # type: ignore[attr-defined]
    _MODEL = model
    print(f"loaded technique classifier from {ckpt_path} "
          f"(epoch={state.get('epoch','?')}, val_acc={state.get('val_acc','?'):.3f})")
    return _MODEL


def pick_technique(prev_clip: dict, prev_seg: dict,
                   cand_clip: dict, cand_seg: dict,
                   ckpt_path: str,
                   default_bars: int = 16) -> dict:
    """
    Use trained classifier to pick best technique for this transition.
    Returns technique dict like {'name': ..., 'bars': ...}.
    """
    import torch
    feats = extract_pair_features(prev_clip, prev_seg, cand_clip, cand_seg)
    model = load_model(ckpt_path)
    x = torch.from_numpy(feats).float().unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    name = TECHNIQUES[idx]
    return _technique_with_defaults(name, default_bars)


def _technique_with_defaults(name: str, default_bars: int) -> dict:
    """Provide sensible default params per technique."""
    base = {'name': name, 'bars': default_bars}
    if name == 'silence_drop':
        base['silence_beats'] = 2
    elif name == 'spinback':
        base['spinback_beats'] = 4
    elif name == 'loop_tighten':
        base['start_bars'] = 4
    elif name == 'pitch_bend':
        base['semitones'] = 1.0
    elif name == 'echo_out':
        base['delay_beats'] = 0.5
        base['feedback'] = 0.55
    elif name == 'scratch_fill':
        base['n_jogs'] = 4
    elif name == 'loop_callback':
        base['repetitions'] = 2
    return base
