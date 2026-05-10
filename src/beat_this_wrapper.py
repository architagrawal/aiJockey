"""Beat-This! GPU beat + downbeat detection wrapper.

Repo: CPJKU/beat_this — joint beats + downbeats from one transformer pass.
MIT licensed. PyTorch native, ROCm-compatible.

Replaces librosa beat_track + downbeats=beats[::4] heuristic in analyze.py.
The heuristic corrupts ~3/4 of downbeats on swung / non-4/4 / pickup-bar
material; Beat-This! gets phrase-grid alignment within ~30 ms on 4/4 pop/EDM
and handles 3/4, 6/8, swung patterns the librosa fallback can't touch.

Lazy load: model is only imported on first call so import-time cost stays
free for paths that don't need beats.

Env:
  AIJOCKEY_BEAT_THIS    0|1   default 1; set 0 to force librosa fallback.
  AIJOCKEY_BEAT_THIS_CKPT  str   default 'final0'
  AIJOCKEY_BEAT_THIS_DBN   auto|0|1  default 'auto'. 'auto' picks DBN when
                                     madmom is importable, falls back to
                                     `minimal` postprocessor otherwise.
                                     The minimal postprocessor over-detects
                                     2-3x BPM on complex material (verified
                                     test1/test2: 333/231 BPM minimal vs
                                     103/117 BPM real); guards below catch
                                     it and force librosa fallback.
  AIJOCKEY_BEAT_THIS_TEMPO_CAP  float  default 220.0; output rejected (returns
                                       to caller's librosa fallback) when
                                       median-IBI tempo exceeds this BPM.

Install madmom for DBN postprocessor (CPJKU main has Py3.10 fix):
    pip install --no-build-isolation 'madmom @ git+https://github.com/CPJKU/madmom.git@main'
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


_MODEL = None
_DEVICE: Optional[str] = None
_DTYPE = None
_LOAD_FAILED = False


def _madmom_available() -> bool:
    try:
        import madmom  # noqa: F401
        return True
    except Exception:
        return False


def _use_dbn() -> bool:
    """Resolve AIJOCKEY_BEAT_THIS_DBN — 'auto' (default) means use DBN when
    madmom is importable, otherwise fall back to minimal (which the sanity
    guards will likely reject)."""
    v = os.environ.get('AIJOCKEY_BEAT_THIS_DBN', 'auto').lower()
    if v == 'auto':
        return _madmom_available()
    return v == '1'


def available() -> bool:
    """Return True if Beat-This! is installed and load did not previously fail."""
    if os.environ.get('AIJOCKEY_BEAT_THIS', '1') == '0':
        return False
    if _LOAD_FAILED:
        return False
    try:
        import beat_this  # noqa: F401
        return True
    except Exception:
        return False


def _load(device: str = 'cuda'):
    """Lazy-load Beat-This! model + checkpoint. Idempotent."""
    global _MODEL, _DEVICE, _DTYPE, _LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _LOAD_FAILED:
        return None
    try:
        import torch
        from beat_this.inference import File2Beats  # type: ignore

        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        ckpt = os.environ.get('AIJOCKEY_BEAT_THIS_CKPT', 'final0')
        # Beat-This! `File2Beats` ships its own preprocessor + post-processor.
        # `dbn=True` uses madmom DBN post-processor — accurate, requires
        # madmom (CPJKU main 0.17.dev0+ builds on Py3.10 with cython<3 +
        # --no-build-isolation). `dbn=False` minimal peak-picker produces
        # 2-3x BPM garbage on real DJ clips. Auto-detect: prefer DBN when
        # madmom importable, fall back with sanity guards otherwise.
        use_dbn = _use_dbn()
        _MODEL = File2Beats(checkpoint_path=ckpt, device=device, dbn=use_dbn)
        print(f"[beat_this] postprocessor: {'DBN (madmom)' if use_dbn else 'minimal'}")
        _DEVICE = device
        try:
            _DTYPE = torch.bfloat16 if (device == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float32
        except Exception:
            _DTYPE = torch.float32
        return _MODEL
    except Exception as e:
        print(f"[beat-this] load failed ({e.__class__.__name__}: {e}); "
              f"librosa fallback will be used")
        _LOAD_FAILED = True
        return None


def beats_from_array(audio_mono: np.ndarray, sr: int,
                     device: str = 'cuda') -> tuple[float, list[float], list[float]]:
    """Compute (tempo_bpm, beats_sec, downbeats_sec) from mono numpy audio.

    Beat-This! ships a `File2Beats` API expecting a path. We use the lower-level
    `Audio2Beats` helper that takes (audio, sr) directly — avoids a temp WAV
    write per clip in the hot path.
    """
    try:
        import torch
        from beat_this.inference import Audio2Beats  # type: ignore
    except Exception as e:
        raise RuntimeError(f"beat_this not importable: {e}")

    # Ensure model is loaded (also sets _DEVICE).
    if _MODEL is None:
        _load(device=device)
    if _LOAD_FAILED:
        raise RuntimeError("beat_this load previously failed")

    dev = _DEVICE or device
    if dev == 'cuda' and not torch.cuda.is_available():
        dev = 'cpu'

    # Audio2Beats wraps the same model — instantiate once per call is cheap
    # because it reuses the cached File2Beats internals. Most cost is the
    # forward pass, not the wrapper init.
    use_dbn = _use_dbn()
    a2b = Audio2Beats(checkpoint_path=os.environ.get('AIJOCKEY_BEAT_THIS_CKPT', 'final0'),
                      device=dev, dbn=use_dbn)
    audio = audio_mono.astype(np.float32)
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    beats, downbeats = a2b(audio, sr)
    beats = [float(t) for t in np.asarray(beats).tolist()]
    downbeats = [float(t) for t in np.asarray(downbeats).tolist()]
    if len(beats) > 1:
        ibis = np.diff(beats)
        tempo = 60.0 / float(np.median(ibis))
    else:
        tempo = 0.0
    # Sanity guard: minimal postprocessor (dbn=False) over-detects on complex
    # material and emits 2-3x true BPM with near-equal beats:downbeats counts.
    # Reject so the caller falls through to its librosa/madmom path.
    cap = float(os.environ.get('AIJOCKEY_BEAT_THIS_TEMPO_CAP', '220.0'))
    if tempo > cap:
        raise RuntimeError(f"beat_this output rejected: tempo={tempo:.1f} > cap={cap} "
                           f"(minimal postprocessor over-detection — install madmom + "
                           f"set AIJOCKEY_BEAT_THIS_DBN=1, or stay on librosa fallback)")
    if len(beats) > 0 and len(downbeats) >= len(beats) * 0.6:
        # downbeats should be ~1/4 of beats in 4/4. >0.6 means postprocessor
        # is treating most beats as downbeats — same failure mode.
        raise RuntimeError(f"beat_this output rejected: downbeats={len(downbeats)} "
                           f">= 60% of beats={len(beats)} (minimal postprocessor failure)")
    return tempo, beats, downbeats
