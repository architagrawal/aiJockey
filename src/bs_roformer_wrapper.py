"""BS-Roformer / Mel-Band Roformer wrapper for vocal stem separation.

Repo: lucidrains/BS-RoFormer (MIT). Vocals SDR ~9 dB (htdemucs_ft) → ~11 dB.
Cleaner vocal stems → cleaner stem-swap, better mashup.

Drop-in for the 'vocals' source — drums/bass/other still come from htdemucs_ft.
The Analyzer composes: htdemucs_ft.stems(wav) - htdemucs_ft.vocals + bs_roformer.vocals.

Lazy load: heavy import only on first vocal-stem call. Falls back to demucs
vocals when not installed or when checkpoint missing.

Env:
  AIJOCKEY_BS_ROFORMER       0|1   default 0 (opt-in until checkpoint validated)
  AIJOCKEY_BS_ROFORMER_CKPT  path  default '' — set to .pth/.ckpt path, e.g.
                                   a UVR Mel-Band Roformer checkpoint.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


_MODEL = None
_DEVICE: Optional[str] = None
_DTYPE = None
_LOAD_FAILED = False


def enabled() -> bool:
    """True if env opts in AND a checkpoint path is provided."""
    if os.environ.get('AIJOCKEY_BS_ROFORMER', '0') != '1':
        return False
    return bool(os.environ.get('AIJOCKEY_BS_ROFORMER_CKPT', '').strip())


def _load(device: str = 'cuda'):
    """Lazy-load BS-Roformer model + checkpoint. Idempotent."""
    global _MODEL, _DEVICE, _DTYPE, _LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _LOAD_FAILED:
        return None
    ckpt_path = os.environ.get('AIJOCKEY_BS_ROFORMER_CKPT', '').strip()
    if not ckpt_path:
        _LOAD_FAILED = True
        print("[bs_roformer] AIJOCKEY_BS_ROFORMER_CKPT not set; skipping")
        return None
    try:
        import torch
        # Try lucidrains' bs_roformer first; fall through to mel_band if
        # the checkpoint is a Mel-Band Roformer instead.
        try:
            from bs_roformer import BSRoformer as _Net  # type: ignore
        except Exception:
            from bs_roformer import MelBandRoformer as _Net  # type: ignore

        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        # Default architectural args mirror the published "vocals" checkpoints.
        # If the user supplies a custom ckpt, they should override via
        # AIJOCKEY_BS_ROFORMER_CFG (JSON) — not implemented yet, defer until
        # we standardize on a specific upstream release.
        net = _Net(dim=512, depth=12, stereo=True, num_stems=1)
        sd = torch.load(ckpt_path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        # Strip common 'model.' prefix from PL/UVR checkpoints
        sd = {k.removeprefix('model.'): v for k, v in sd.items()}
        net.load_state_dict(sd, strict=False)
        net.eval()
        if device == 'cuda':
            net = net.cuda()
            try:
                _DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            except Exception:
                _DTYPE = torch.float16
        else:
            _DTYPE = torch.float32
        _MODEL = net
        _DEVICE = device
        print(f"[bs_roformer] loaded {ckpt_path} on {device} ({_DTYPE})")
        return _MODEL
    except Exception as e:
        print(f"[bs_roformer] load failed ({e.__class__.__name__}: {e}); "
              f"demucs vocals will be used")
        _LOAD_FAILED = True
        return None


def vocals_from_wav(wav, sr: int = 44100, device: str = 'cuda'):
    """Run BS-Roformer on (channels, T) torch.Tensor. Returns torch.Tensor (2, T').

    Returns None on failure — caller should fall through to demucs vocals.
    """
    try:
        import torch
    except Exception:
        return None
    model = _load(device=device)
    if model is None:
        return None
    try:
        x = wav.unsqueeze(0)  # (1, C, T)
        if _DEVICE == 'cuda':
            x = x.cuda()
            if _DTYPE != torch.float32:
                with torch.inference_mode(), torch.amp.autocast(
                        device_type='cuda', dtype=_DTYPE):
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
        print(f"[bs_roformer] forward failed ({e}); demucs vocals fallback")
        return None
