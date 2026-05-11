"""MuQ-Eval reference-free music-quality scorer.

Verified HF ID: zhudi2825/MuQ-Eval-A1 (MIT). NOT a standard
AutoModel.from_pretrained — uses a custom `MusicQualityModel` class
from github.com/dgtql/MuQ-Eval. Weights distributed as separate
config.yaml + model_state_dict.pt files via hf_hub_download.

Env knobs:
    AIJOCKEY_MUQ_EVAL_ENABLE   0|1   default 0
    AIJOCKEY_MUQ_EVAL_HF_ID    str   default 'zhudi2825/MuQ-Eval-A1'
    AIJOCKEY_MUQ_EVAL_DEVICE   str   default 'cuda' if available
    AIJOCKEY_MUQ_EVAL_REPO     path  to cloned github.com/dgtql/MuQ-Eval
                                       (needed for MusicQualityModel class)

Output: scalar quality MOS-like score in ~[1, 5] (higher = better).
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_PIPE: dict | None = None
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
            from omegaconf import OmegaConf  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore

            repo_path = os.environ.get("AIJOCKEY_MUQ_EVAL_REPO")
            if repo_path and Path(repo_path).exists():
                sp = str(Path(repo_path) / "src")
                if sp not in sys.path:
                    sys.path.insert(0, sp)
            try:
                # First the canonical import path (src/model.py exposes
                # MusicQualityModel). If the repo isn't cloned this fails.
                from model import MusicQualityModel  # type: ignore
            except Exception:
                # Fallback: try a package-style import.
                from muq_eval.model import MusicQualityModel  # type: ignore

            hf_id = os.environ.get("AIJOCKEY_MUQ_EVAL_HF_ID",
                                     "zhudi2825/MuQ-Eval-A1")
            cfg_p = hf_hub_download(hf_id, "config.yaml")
            wts_p = hf_hub_download(hf_id, "model_state_dict.pt")
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model = MusicQualityModel(OmegaConf.load(cfg_p))
            state = torch.load(wts_p, map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            model.eval()
            if device == "cuda":
                model = model.cuda()
            _PIPE = {"model": model, "device": device}
            print(f"[muq_eval] loaded {hf_id} on {device}")
            return _PIPE
        except Exception as e:
            print(f"[muq_eval] load failed ({e.__class__.__name__}: {e})")
            _LOAD_FAILED = True
            return None


def score(audio_path: str | Path, device: str = "cuda") -> float | None:
    """Return quality scalar (~1-5) or None."""
    if not enabled():
        return None
    pipe = _load(device=device)
    if pipe is None:
        return None
    try:
        import librosa
        import torch
        wav, _ = librosa.load(str(audio_path), sr=24000, mono=True,
                                duration=10.0)
        x = torch.from_numpy(wav).float().unsqueeze(0)
        if pipe["device"] == "cuda":
            x = x.cuda()
        with torch.inference_mode():
            out = pipe["model"](x)
        if hasattr(out, "item"):
            return float(out.item())
        if hasattr(out, "cpu"):
            arr = out.cpu().squeeze().tolist()
            return float(arr if not isinstance(arr, list) else arr[0])
        return float(out)
    except Exception as e:
        print(f"[muq_eval] score failed for {audio_path}: {e}")
        return None
