"""Frequency-masking-aware ducking for clip overlap regions.

When two clips overlap, fight zones (50-200 Hz kicks, 200-2k Hz
vocals/leads) produce comb-filter mud. This module computes per-bar
STFT energy in fight bands and applies a soft EQ-duck on the louder
clip's competing bands so the quieter clip "wins" each band.

Toggle: AIJOCKEY_FREQ_DUCK=1
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


FIGHT_BANDS = [
    (50.0, 200.0),    # sub/kick
    (200.0, 800.0),   # low-mid mud
    (1000.0, 3000.0), # vocal/lead fundamentals
    (4000.0, 8000.0), # presence/hi-hat clash
]


def freq_mask_duck(a: np.ndarray, b: np.ndarray, sr: int = 44100,
                    n_fft: int = 4096, hop: int = 1024,
                    duck_db: float = -4.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-band duck the louder of (A, B) in each FIGHT_BAND.

    Returns (a_out, b_out). Same shape as inputs (min length).
    """
    if not _HAS_TORCH:
        return a, b
    n = min(a.shape[1], b.shape[1])
    if n < n_fft * 2:
        return a, b
    try:
        ta = torch.from_numpy(a[:, :n].astype(np.float32))
        tb = torch.from_numpy(b[:, :n].astype(np.float32))
        win = torch.hann_window(n_fft)
        A = torch.stft(ta, n_fft=n_fft, hop_length=hop, window=win,
                        return_complex=True, center=True)
        B = torch.stft(tb, n_fft=n_fft, hop_length=hop, window=win,
                        return_complex=True, center=True)
        # freq axis
        freqs = torch.linspace(0, sr / 2.0, A.shape[-2])
        mag_a = A.abs(); mag_b = B.abs()
        gain = float(10.0 ** (duck_db / 20.0))
        for lo, hi in FIGHT_BANDS:
            band_mask = ((freqs >= lo) & (freqs < hi)).float().unsqueeze(-1)
            # Per-frame: which clip has more energy in this band?
            ea = (mag_a * band_mask.unsqueeze(0)).sum(dim=-2)
            eb = (mag_b * band_mask.unsqueeze(0)).sum(dim=-2)
            louder_a = (ea > eb).float().unsqueeze(0).unsqueeze(0)
            # Duck the louder one's magnitude in this band
            a_gain = 1.0 - (1.0 - gain) * (louder_a * band_mask)
            b_gain = 1.0 - (1.0 - gain) * ((1.0 - louder_a) * band_mask)
            A = A * a_gain.expand_as(A)
            B = B * b_gain.expand_as(B)
        a_out = torch.istft(A, n_fft=n_fft, hop_length=hop, window=win,
                             length=n).numpy()
        b_out = torch.istft(B, n_fft=n_fft, hop_length=hop, window=win,
                             length=n).numpy()
        return a_out.astype(np.float32), b_out.astype(np.float32)
    except Exception:
        return a, b
