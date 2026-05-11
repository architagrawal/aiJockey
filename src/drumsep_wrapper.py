"""DrumSep — drum-stem sub-separation (kick/snare/toms/hat/cymbals).

Inagoy/drumsep (MIT). Trained on StemGMD. Takes the `drums` stem
(from Demucs/RoFormer) and splits into 4 sub-stems. Useful for:
    - drum_replace transition: target only kick or only snare
    - sidechain duck trigger: clean kick envelope
    - de-essing: isolate hi-hats for surgical EQ

Env:
    AIJOCKEY_DRUMSEP_ENABLE   0|1   default 0
    AIJOCKEY_DRUMSEP_CKPT     path  default 'inagoy/drumsep'
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_PIPE = None
_LOAD_FAILED = False


def enabled() -> bool:
    if os.environ.get("AIJOCKEY_DRUMSEP_ENABLE", "0") != "1":
        return False
    return not _LOAD_FAILED


def _load(device: str | None = None):
    global _PIPE, _LOAD_FAILED
    if _PIPE is not None:
        return _PIPE
    if _LOAD_FAILED:
        return None
    with _LOCK:
        if _PIPE is not None:
            return _PIPE
        try:
            import torch
            # DrumSep uses Demucs's apply_model style. Loaded as a custom
            # HF repo with a model.yaml + weights.
            from demucs.api import Separator  # type: ignore
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt = os.environ.get("AIJOCKEY_DRUMSEP_CKPT", "inagoy/drumsep")
            sep = Separator(model=ckpt, device=device)
            _PIPE = {"sep": sep, "device": device}
            print(f"[drumsep] loaded {ckpt} on {device}")
            return _PIPE
        except Exception as e:
            print(f"[drumsep] load failed ({e.__class__.__name__}: {e})")
            _LOAD_FAILED = True
            return None


def separate(drums_audio_path: str | Path,
              device: str = "cuda") -> dict | None:
    """Returns {"kick","snare","toms","hat","cymbals": np.ndarray} or None.

    Input should be Demucs's `drums` stem WAV. Output keys may collapse
    based on the trained checkpoint (4 vs 5 sub-stems).
    """
    if not enabled():
        return None
    pipe = _load(device=device)
    if pipe is None:
        return None
    try:
        sep = pipe["sep"]
        _, separated = sep.separate_audio_file(str(drums_audio_path))
        out: dict = {}
        for name, tensor in separated.items():
            out[name.lower()] = tensor.detach().cpu().numpy().astype("float32")
        return out
    except Exception as e:
        print(f"[drumsep] separate failed: {e}")
        return None
