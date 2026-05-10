"""Mel-Band Roformer wrapper — vocal stem upgrade above htdemucs_ft.

Outperforms BS-Roformer on vocals/drums/other (per arXiv 2310.01809).
SDR 11.93 on Multisong vs htdemucs_ft ~9 dB → audibly cleaner stem-swap
and mashup transitions.

Architectural difference from `bs_roformer_wrapper.py`:
    - BS-Roformer splits frequency bins by hand-crafted band table.
    - Mel-Band Roformer maps bins via Mel scale (overlapping subbands).
      Better matches human auditory frequency resolution.

Both share the lucidrains/BS-RoFormer codebase. The wrapper here picks
the MelBandRoformer class explicitly, with checkpoint defaults known to
work (UVR Mel-Band Roformer vocals, KimberleyJSN release).

Drop-in for the 'vocals' stem only; drums/bass/other still come from
htdemucs_ft (Mel-Band Roformer published checkpoints are vocals-only at
the time of this branch).

Env:
    AIJOCKEY_MEL_BAND_ROFORMER       0|1   default 0 (opt-in)
    AIJOCKEY_MEL_BAND_ROFORMER_CKPT  path  default ''
                                            set to .ckpt path, e.g. UVR's
                                            'mel_band_roformer_vocals_fv4.ckpt'
    AIJOCKEY_MEL_BAND_DIM            int   default 384  (matches UVR ckpt)
    AIJOCKEY_MEL_BAND_DEPTH          int   default 12

Usage:
    from mel_band_roformer_wrapper import enabled, vocals_from_wav
    if enabled():
        new_vox = vocals_from_wav(wav, sr=44100, device='cuda')
        if new_vox is not None:
            stems['vocals'] = new_vox
"""
from __future__ import annotations

import os
import threading

import numpy as np


_LOCK = threading.Lock()
_MODEL = None
_DEVICE = None
_DTYPE = None
_LOAD_FAILED = False


def enabled() -> bool:
    """True if env opts in AND a checkpoint path is provided."""
    if os.environ.get("AIJOCKEY_MEL_BAND_ROFORMER", "0") != "1":
        return False
    return bool(os.environ.get("AIJOCKEY_MEL_BAND_ROFORMER_CKPT", "").strip())


def _load(device: str = "cuda"):
    """Lazy-load Mel-Band Roformer model + checkpoint. Idempotent."""
    global _MODEL, _DEVICE, _DTYPE, _LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _LOAD_FAILED:
        return None

    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        if _LOAD_FAILED:
            return None

        ckpt_path = os.environ.get("AIJOCKEY_MEL_BAND_ROFORMER_CKPT", "").strip()
        if not ckpt_path or not os.path.exists(ckpt_path):
            _LOAD_FAILED = True
            print(f"[mel_band_roformer] checkpoint not found at "
                  f"{ckpt_path or '<unset>'}; falling back to htdemucs_ft vocals")
            return None

        try:
            import torch
            from bs_roformer import MelBandRoformer   # type: ignore

            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"

            # UVR / KimberleyJSN published checkpoints typically use:
            #   dim=384, depth=12, stereo=True, num_stems=1 (vocals only)
            # Override via env when adopting a custom ckpt with different config.
            dim = int(os.environ.get("AIJOCKEY_MEL_BAND_DIM", "384"))
            depth = int(os.environ.get("AIJOCKEY_MEL_BAND_DEPTH", "12"))

            net = MelBandRoformer(
                dim=dim,
                depth=depth,
                stereo=True,
                num_stems=1,
            )
            sd = torch.load(ckpt_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # Strip common prefixes: 'model.' (PyTorch Lightning),
            # '_orig_mod.' (torch.compile wrapper), 'module.' (DataParallel)
            sd = {
                k.removeprefix("model.").removeprefix("_orig_mod.").removeprefix("module."): v
                for k, v in sd.items()
            }
            missing, unexpected = net.load_state_dict(sd, strict=False)
            if missing:
                print(f"[mel_band_roformer] warn: {len(missing)} missing keys "
                      f"(first: {missing[0] if missing else '-'})")
            if unexpected:
                print(f"[mel_band_roformer] warn: {len(unexpected)} unexpected keys")

            net.eval()
            if device == "cuda":
                net = net.cuda()
                try:
                    _DTYPE = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                              else torch.float16)
                except Exception:
                    _DTYPE = torch.float16
            else:
                _DTYPE = torch.float32

            _MODEL = net
            _DEVICE = device
            print(f"[mel_band_roformer] loaded {ckpt_path} on {device} "
                  f"({_DTYPE}, dim={dim}, depth={depth})")
            return _MODEL
        except Exception as e:
            print(f"[mel_band_roformer] load failed "
                  f"({e.__class__.__name__}: {e}); demucs vocals fallback")
            _LOAD_FAILED = True
            return None


def vocals_from_wav(wav, sr: int = 44100, device: str = "cuda"):
    """Run Mel-Band Roformer on (channels, T) torch.Tensor.

    Returns torch.Tensor (2, T') with vocal stem, or None on failure.
    Caller falls through to htdemucs_ft vocals on None.
    """
    try:
        import torch
    except Exception:
        return None
    model = _load(device=device)
    if model is None:
        return None
    try:
        x = wav.unsqueeze(0)    # (1, C, T)
        if _DEVICE == "cuda":
            x = x.cuda()
            if _DTYPE != torch.float32:
                with torch.inference_mode(), torch.amp.autocast(
                        device_type="cuda", dtype=_DTYPE):
                    y = model(x)
            else:
                with torch.inference_mode():
                    y = model(x)
        else:
            with torch.inference_mode():
                y = model(x)
        # Output convention: (B, num_stems, C, T). num_stems=1 → (1, 1, C, T).
        if y.dim() == 4:
            y = y[0, 0]
        elif y.dim() == 3:
            y = y[0]
        return y.detach().cpu().float()
    except Exception as e:
        print(f"[mel_band_roformer] forward failed ({e}); demucs vocals fallback")
        return None


__all__ = ["enabled", "vocals_from_wav"]
