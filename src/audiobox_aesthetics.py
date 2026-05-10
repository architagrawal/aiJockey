"""Meta Audiobox Aesthetics — reference-free 4-axis quality scorer.

Paper: "Meta Audiobox Aesthetics: Unified Automatic Quality Assessment"
       (arXiv 2502.05139)
Repo: https://github.com/facebookresearch/audiobox-aesthetics
HF model: facebook/audiobox-aesthetics

Returns 4 axes per audio file (each in roughly [0, 10]):
    PQ (Production Quality)     — fidelity, clarity, dynamics, freq balance
    PC (Production Complexity)  — concurrent audio components / scene density
    CE (Content Enjoyment)      — emotional impact, artistic expression
    CU (Content Usefulness)     — value as source material for creative tasks

Reference-free: no ground truth needed, just the rendered mix. This makes
it a drop-in second-opinion critic alongside our deterministic audio
probes (RMS / xcorr / phase).

Why we want this:
    - Replaces unreliable CriticV2 (val acc 0.77, codec bias) with zero
      training cost. Pretrained, open weights.
    - CLAP-score (currently used for retrieval) measures SEMANTIC
      alignment, not perceptual quality — paper explicitly warns that
      a clip can score high CLAP while sounding degraded.
    - Probes catch deterministic artifacts (energy mismatch, phase
      cancellation). Audiobox catches subjective polish (mastering,
      spatialization, dynamics) that probes miss.

Wire-in pattern (server/api.py):
    from audiobox_aesthetics import score
    aes = score(rendered_path)
    if aes:
        resp.headers['X-Aesthetics-PQ'] = f"{aes['PQ']:.2f}"
        resp.headers['X-Aesthetics-CE'] = f"{aes['CE']:.2f}"

Env:
    AIJOCKEY_AUDIOBOX_AESTHETICS  0|1  default 0 (opt-in)
    AIJOCKEY_AUDIOBOX_MODEL       hf-id  default 'facebook/audiobox-aesthetics'
    AIJOCKEY_AUDIOBOX_DEVICE      str   default 'cuda' if available

Lazy-load. Returns None on any failure — caller skips header attachment.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()
_PIPE = None
_DEVICE = None
_LOAD_FAILED = False


def enabled() -> bool:
    """True if env opts in. Doesn't probe import — that happens at load."""
    if os.environ.get("AIJOCKEY_AUDIOBOX_AESTHETICS", "0") != "1":
        return False
    if _LOAD_FAILED:
        return False
    return True


def _load(device: str = "cuda"):
    """Lazy-init the audiobox-aesthetics model. Returns pipe or None."""
    global _PIPE, _DEVICE, _LOAD_FAILED
    if _PIPE is not None:
        return _PIPE
    if _LOAD_FAILED:
        return None

    with _LOCK:
        if _PIPE is not None:
            return _PIPE
        if _LOAD_FAILED:
            return None

        try:
            import torch
            # Meta released audiobox-aesthetics as a standalone package.
            # The expected API:
            #   from audiobox_aesthetics.infer import initialize_predictor
            #   pred = initialize_predictor()
            #   result = pred.forward([{"path": "...wav"}])
            try:
                from audiobox_aesthetics.infer import initialize_predictor   # type: ignore
            except ImportError:
                # Fall through to HF transformers path if the standalone
                # package isn't installed but the HF model is reachable.
                _PIPE = _load_hf_fallback(device=device)
                if _PIPE is not None:
                    _DEVICE = _PIPE.get("device", device)
                    return _PIPE
                raise

            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            pred = initialize_predictor()
            _PIPE = {"kind": "standalone", "predictor": pred, "device": device}
            _DEVICE = device
            print(f"[audiobox_aesthetics] loaded standalone predictor on {device}")
            return _PIPE
        except Exception as e:
            print(f"[audiobox_aesthetics] load failed "
                  f"({e.__class__.__name__}: {e})")
            _LOAD_FAILED = True
            return None


def _load_hf_fallback(device: str):
    """Fallback path when audiobox_aesthetics package missing.

    Some users install only the HF model weights without the inference
    helper package. Returns a thin wrapper that loads weights via
    transformers and runs the same scoring math.
    """
    try:
        import torch
        from transformers import AutoModel, AutoFeatureExtractor   # type: ignore
        model_id = os.environ.get(
            "AIJOCKEY_AUDIOBOX_MODEL", "facebook/audiobox-aesthetics")
        proc = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
        if device == "cuda":
            model = model.cuda()
        return {"kind": "hf", "model": model, "proc": proc, "device": device}
    except Exception as e:
        print(f"[audiobox_aesthetics] HF fallback load failed: {e}")
        return None


def score(audio_path: str | Path,
          device: str = "cuda") -> dict[str, float] | None:
    """Score one rendered audio file. Returns 4-axis dict or None.

    Output:
        {"PQ": float, "PC": float, "CE": float, "CU": float}

    None on:
        - env not enabled
        - model load failure
        - audio file unreadable
    """
    if not enabled():
        return None
    pipe = _load(device=device)
    if pipe is None:
        return None

    audio_path = str(audio_path)
    try:
        if pipe["kind"] == "standalone":
            pred = pipe["predictor"]
            # API: pred.forward([{"path": "..."}]) → list of dicts with
            # axes as keys, values are floats.
            results = pred.forward([{"path": audio_path}])
            if not results:
                return None
            r = results[0]
            return {
                "PQ": float(r.get("PQ", r.get("production_quality", 0.0)) or 0.0),
                "PC": float(r.get("PC", r.get("production_complexity", 0.0)) or 0.0),
                "CE": float(r.get("CE", r.get("content_enjoyment", 0.0)) or 0.0),
                "CU": float(r.get("CU", r.get("content_usefulness", 0.0)) or 0.0),
            }

        # HF fallback path
        import torch
        import librosa
        wav, sr = librosa.load(audio_path, sr=16000, mono=True, duration=30.0)
        inputs = pipe["proc"](wav, sampling_rate=sr, return_tensors="pt")
        if pipe["device"] == "cuda":
            inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = pipe["model"](**inputs)
        # Custom HF model exposes .scores or similar attribute.
        scores = getattr(out, "scores", None)
        if scores is None:
            scores = out
        if hasattr(scores, "cpu"):
            arr = scores.cpu().squeeze().tolist()
            if len(arr) >= 4:
                return {
                    "PQ": float(arr[0]),
                    "PC": float(arr[1]),
                    "CE": float(arr[2]),
                    "CU": float(arr[3]),
                }
        return None
    except Exception as e:
        print(f"[audiobox_aesthetics] score failed for {audio_path}: {e}")
        return None


def severity_proxy(scores: dict[str, float] | None,
                   pq_target: float = 6.0,
                   ce_target: float = 6.0) -> float | None:
    """Derive a single severity-like number in [0, 1] for parity with
    the existing audio_probes severity scale.

    Lower is better (matches probe severity convention). Computed as:
        severity = max(0, target - measured) / target
    averaged across PQ + CE (the two most relevant axes for mix quality).

    Returns None when scores is None.
    """
    if not scores:
        return None
    pq = scores.get("PQ", 0.0)
    ce = scores.get("CE", 0.0)
    pq_def = max(0.0, (pq_target - pq) / pq_target) if pq_target > 0 else 0.0
    ce_def = max(0.0, (ce_target - ce) / ce_target) if ce_target > 0 else 0.0
    return float(min(1.0, (pq_def + ce_def) / 2.0))


__all__ = ["enabled", "score", "severity_proxy"]
