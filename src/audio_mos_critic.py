"""AudioMOS DORA-MOS reference-free quality scorer.

Paper: AudioMOS Challenge 2025 (arXiv 2509.01336). MOS predictor trained
for synthetic-audio quality estimation. Adds another head to our
critic stack alongside Audiobox + MuQ-Eval.

API mirrors audiobox_critic / muq_eval_critic.

Env knobs:
    AIJOCKEY_AUDIO_MOS_ENABLE   0|1   default 0
    AIJOCKEY_AUDIO_MOS_MODEL    hf id default 'audiomos/dora-mos' (TODO verify on deploy)
    AIJOCKEY_AUDIO_MOS_DEVICE   str   default 'cuda' if available

Output: MOS scalar in [1, 5] (higher = better). Caller divides by 5 or
keeps raw — same opaque-ratio handling as Audiobox PQ.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_PIPE = None
_LOAD_FAILED = False


def enabled() -> bool:
    if os.environ.get("AIJOCKEY_AUDIO_MOS_ENABLE", "0") != "1":
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
            model_id = os.environ.get("AIJOCKEY_AUDIO_MOS_MODEL", "audiomos/dora-mos")
            proc = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            model.eval()
            if device == "cuda":
                model = model.cuda()
            _PIPE = {"model": model, "proc": proc, "device": device}
            print(f"[audio_mos] loaded {model_id} on {device}")
            return _PIPE
        except Exception as e:
            print(f"[audio_mos] load failed ({e.__class__.__name__}: {e})")
            _LOAD_FAILED = True
            return None


def score(audio_path: str | Path, device: str = "cuda") -> float | None:
    """Returns MOS scalar (~1-5) or None."""
    if not enabled():
        return None
    pipe = _load(device=device)
    if pipe is None:
        return None
    try:
        import torch
        import librosa
        wav, sr = librosa.load(str(audio_path), sr=16000, mono=True, duration=30.0)
        inputs = pipe["proc"](wav, sampling_rate=sr, return_tensors="pt")
        if pipe["device"] == "cuda":
            inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = pipe["model"](**inputs)
        s = getattr(out, "mos", getattr(out, "scores", out))
        if hasattr(s, "cpu"):
            arr = s.cpu().squeeze().tolist()
            if isinstance(arr, list):
                return float(arr[0]) if arr else None
            return float(arr)
        return None
    except Exception as e:
        print(f"[audio_mos] score failed for {audio_path}: {e}")
        return None
