"""CriticV2 inference wrapper for /generate score-gating.

Loads a trained mix-critic checkpoint (`checkpoints/mix_critic_v2.pt`) on
first call, scores a rendered audio path, returns a [0, 1] quality score.
Intended to be consumed by `server/api.py` to attach `X-Critic-Score`
header alongside `X-Probe-Severity` — gives a second-opinion signal
beyond the deterministic probes.

Lazy-load: model + checkpoint only loaded on first score() call. If the
checkpoint doesn't exist, score() returns None — no error, caller skips
adding the header.

Why a wrapper: the actual model definition lives in
`src/training/mix_critic.py` (training-time architecture). Inference
should be decoupled — server doesn't import training modules at startup.

Env:
    AIJOCKEY_CRITIC_V2_CKPT     path  default 'checkpoints/mix_critic_v2.pt'
                                       falls through to mix_critic.pt (v1)
                                       if v2 missing
    AIJOCKEY_CRITIC_DEVICE      str   default 'cuda' if available
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np


_LOCK = threading.Lock()
_MODEL = None
_DEVICE = None
_LOAD_FAILED = False


def _ckpt_path() -> Path | None:
    """Resolve checkpoint path. Prefer v2, fall through to v1, else None."""
    explicit = os.environ.get('AIJOCKEY_CRITIC_V2_CKPT')
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    repo_root = Path(__file__).resolve().parent.parent
    for cand in ('checkpoints/mix_critic_v2.pt',
                 'checkpoints/mix_critic.pt'):
        p = repo_root / cand
        if p.exists():
            return p
    return None


def available() -> bool:
    """True if a checkpoint is on disk and load hasn't failed."""
    if _LOAD_FAILED:
        return False
    return _ckpt_path() is not None


def _load() -> bool:
    """Lazy-init model + checkpoint. Returns True on success.

    Idempotent under thread contention via _LOCK.
    """
    global _MODEL, _DEVICE, _LOAD_FAILED
    if _MODEL is not None:
        return True
    if _LOAD_FAILED:
        return False

    with _LOCK:
        if _MODEL is not None:    # double-check after acquire
            return True
        if _LOAD_FAILED:
            return False
        ckpt = _ckpt_path()
        if ckpt is None:
            _LOAD_FAILED = True
            return False
        try:
            import torch
            from training.mix_critic import MixCritic   # type: ignore

            dev = os.environ.get('AIJOCKEY_CRITIC_DEVICE')
            if not dev:
                dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            sd = torch.load(str(ckpt), map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            model = MixCritic()
            model.load_state_dict(sd, strict=False)
            model.eval()
            if dev == 'cuda':
                model = model.cuda()
            _MODEL = model
            _DEVICE = dev
            print(f"[critic_v2] loaded {ckpt} on {dev}")
            return True
        except Exception as e:
            print(f"[critic_v2] load failed ({e.__class__.__name__}: {e})")
            _LOAD_FAILED = True
            return False


def _audio_to_input(audio_path: str | Path,
                     window_seconds: float = 30.0,
                     sample_rate: int = 44100) -> np.ndarray | None:
    """Load + resample mono window. Returns None on failure."""
    try:
        import librosa
        wav, _sr = librosa.load(str(audio_path), sr=sample_rate, mono=True,
                                  duration=window_seconds)
        if wav.size == 0:
            return None
        return wav.astype(np.float32)
    except Exception as e:
        print(f"[critic_v2] audio load failed for {audio_path}: {e}")
        return None


def score(audio_path: str | Path,
           window_seconds: float = 30.0) -> float | None:
    """Score a rendered mix file. Returns float in [0, 1] or None.

    Scores from a single fixed-length window starting at a random offset
    in the file (for robustness vs intro/outro bias). Caller can wrap with
    multi-window averaging if desired.

    Returns None if:
      - checkpoint not on disk
      - model load failed
      - audio file unreadable
    """
    if not _load():
        return None
    audio = _audio_to_input(audio_path, window_seconds=window_seconds)
    if audio is None:
        return None
    try:
        import torch
        with torch.inference_mode():
            x = torch.from_numpy(audio).unsqueeze(0)
            if _DEVICE == 'cuda':
                x = x.cuda()
            logit = _MODEL(x)
            if hasattr(logit, 'logits'):
                logit = logit.logits
            # Critic outputs logit; sigmoid → [0, 1].
            prob = float(torch.sigmoid(logit).cpu().squeeze().item())
            return prob
    except Exception as e:
        print(f"[critic_v2] score failed for {audio_path}: {e}")
        return None


def score_batch(audio_paths: list[str | Path],
                 window_seconds: float = 30.0) -> list[float | None]:
    """Convenience wrapper. No batching at the model level (mix_critic
    architecture is small enough that per-call overhead dominates).
    """
    return [score(p, window_seconds=window_seconds) for p in audio_paths]


__all__ = ['available', 'score', 'score_batch']
