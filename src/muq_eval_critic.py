"""MuQ-Eval reference-free music quality scorer.

Paper: MuQ-Eval (arXiv 2603.22677). Music-specific scorer trained on a
diverse music corpus; intended as DPO reward signal alternative to
aggregate audio-probe severity.

API mirrors audiobox_critic: lazy load, env toggle, graceful failure.

Env knobs:
    AIJOCKEY_MUQ_EVAL_ENABLE   0|1   default 0
    AIJOCKEY_MUQ_EVAL_MODEL    hf id default 'OpenMuQ/MuQ-Eval' (TODO verify on deploy)
    AIJOCKEY_MUQ_EVAL_DEVICE   str   default 'cuda' if available

Returns a single quality scalar in [0, 1] (higher = better). Wraps
whichever model is actually published — caller treats output as opaque
ratio, just like audiobox PQ.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_PIPE = None
_LOAD_FAILED = False


def enabled() -> bool:
    if os.environ.get("AIJOCKEY_MUQ_EVAL_ENABLE", "0") != "1":
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
            from transformers import AutoModel, AutoFeatureExtractor  # type: ignore
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = os.environ.get("AIJOCKEY_MUQ_EVAL_MODEL", "OpenMuQ/MuQ-Eval")
            proc = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            model.eval()
            if device == "cuda":
                model = model.cuda()
            _PIPE = {"model": model, "proc": proc, "device": device}
            print(f"[muq_eval] loaded {model_id} on {device}")
            return _PIPE
        except Exception as e:
            print(f"[muq_eval] load failed ({e.__class__.__name__}: {e})")
            _LOAD_FAILED = True
            return None


def score(audio_path: str | Path, device: str = "cuda") -> float | None:
    """Returns single quality scalar or None on failure."""
    if not enabled():
        return None
    pipe = _load(device=device)
    if pipe is None:
        return None
    try:
        import torch
        import librosa
        wav, sr = librosa.load(str(audio_path), sr=24000, mono=True, duration=30.0)
        inputs = pipe["proc"](wav, sampling_rate=sr, return_tensors="pt")
        if pipe["device"] == "cuda":
            inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = pipe["model"](**inputs)
        s = getattr(out, "scores", out)
        if hasattr(s, "cpu"):
            arr = s.cpu().squeeze().tolist()
            if isinstance(arr, list):
                return float(arr[0]) if arr else None
            return float(arr)
        return None
    except Exception as e:
        print(f"[muq_eval] score failed for {audio_path}: {e}")
        return None
