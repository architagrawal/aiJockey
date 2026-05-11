"""ChordFormer — chord progression detection with confidence.

Hyon/ChordFormer (ISMIR'24, Apache 2.0). WCSR 84% on Billboard.
Returns per-frame chord label + logits → confidence.

Augments Camelot key with chord-level info for jazz/hip-hop/pop
where modal mixture matters more than functional key.

Env:
    AIJOCKEY_CHORDFORMER_ENABLE   0|1   default 0
    AIJOCKEY_CHORDFORMER_REPO     path  to cloned Hyon/ChordFormer
    AIJOCKEY_CHORDFORMER_CKPT     path  to checkpoint .pt
    AIJOCKEY_CHORDFORMER_DEVICE   str   default 'cuda' if available
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

_LOCK = threading.Lock()
_PIPE = None
_LOAD_FAILED = False


def enabled() -> bool:
    if os.environ.get("AIJOCKEY_CHORDFORMER_ENABLE", "0") != "1":
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
            repo = os.environ.get("AIJOCKEY_CHORDFORMER_REPO")
            ckpt = os.environ.get("AIJOCKEY_CHORDFORMER_CKPT")
            if not repo or not Path(repo).exists():
                print(f"[chordformer] no repo at {repo}")
                _LOAD_FAILED = True
                return None
            if not ckpt or not Path(ckpt).exists():
                print(f"[chordformer] no ckpt at {ckpt}")
                _LOAD_FAILED = True
                return None
            if repo not in sys.path:
                sys.path.insert(0, repo)
            from model import ChordFormer  # type: ignore
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model = ChordFormer()
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            model.eval()
            if device == "cuda":
                model = model.cuda()
            _PIPE = {"model": model, "device": device}
            print(f"[chordformer] loaded {ckpt} on {device}")
            return _PIPE
        except Exception as e:
            print(f"[chordformer] load failed ({e.__class__.__name__}: {e})")
            _LOAD_FAILED = True
            return None


def detect_chords(audio_path: str | Path,
                    device: str = "cuda") -> dict | None:
    """Returns {"timestamps_s","chords","confidence"} or None."""
    if not enabled():
        return None
    pipe = _load(device=device)
    if pipe is None:
        return None
    try:
        import torch
        import librosa
        wav, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        x = torch.from_numpy(wav).float().unsqueeze(0)
        if pipe["device"] == "cuda":
            x = x.cuda()
        with torch.inference_mode():
            out = pipe["model"](x)
        # Expected shape: (1, T, n_chord_classes) or similar
        logits = out
        if hasattr(logits, "logits"):
            logits = logits.logits
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        return {
            "chord_indices": idx.squeeze(0).cpu().numpy().tolist(),
            "confidence": conf.squeeze(0).cpu().numpy().tolist(),
            "n_classes": int(probs.shape[-1]),
        }
    except Exception as e:
        print(f"[chordformer] detect failed for {audio_path}: {e}")
        return None
