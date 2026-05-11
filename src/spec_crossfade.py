"""Spectrogram-domain crossfade for cleaner transitions in 1-8 kHz.

Time-domain crossfades produce comb-filter artifacts in the
phase-cancellation zone when two coherent signals overlap. STFT-domain
masks let us blend by bin, locking phase to whichever clip dominates
energy per band.

Approach:
    1. STFT both overlap regions.
    2. Per (time, freq) bin: compute energy of A vs B.
    3. Mask = sigmoid of (energy_b - energy_a) modulated by a time ramp.
    4. Output magnitude = |A|*(1-mask) + |B|*mask.
    5. Output phase = phase of whichever bin has higher energy.
    6. iSTFT.

Falls through to time-domain xfade on any error.

Reference: Hybrid Demucs spectrogram processing pattern, applied to
junction overlap rather than source separation.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _equal_power_ramp(n: int) -> np.ndarray:
    """Smooth equal-power ramp, 0→1 over n samples."""
    t = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
    return np.sin(t) ** 2


def spectral_crossfade(a: np.ndarray, b: np.ndarray, sr: int = 44100,
                        n_fft: int = 2048, hop: int = 512,
                        sigmoid_k: float = 6.0) -> np.ndarray:
    """Blend two same-length stereo waveforms in STFT domain.

    Args:
        a, b: shape (channels, samples). Must have equal length.
        sr: sample rate.
        n_fft: STFT window.
        hop: STFT hop.
        sigmoid_k: steepness of bin-level blending decision.

    Returns:
        stereo waveform shape (channels, samples).
    """
    if not _HAS_TORCH:
        # Fallback: equal-power crossfade
        n = min(a.shape[1], b.shape[1])
        ramp = _equal_power_ramp(n)
        out = a[:, :n] * (1.0 - ramp) + b[:, :n] * ramp
        return out.astype(np.float32)
    try:
        n = min(a.shape[1], b.shape[1])
        if n < n_fft * 2:
            ramp = _equal_power_ramp(n)
            return (a[:, :n] * (1.0 - ramp) + b[:, :n] * ramp).astype(np.float32)
        ta = torch.from_numpy(a[:, :n].astype(np.float32))
        tb = torch.from_numpy(b[:, :n].astype(np.float32))
        win = torch.hann_window(n_fft)
        A = torch.stft(ta, n_fft=n_fft, hop_length=hop, window=win,
                        return_complex=True, center=True)
        B = torch.stft(tb, n_fft=n_fft, hop_length=hop, window=win,
                        return_complex=True, center=True)
        # Time ramp in STFT frame domain
        n_frames = A.shape[-1]
        time_ramp = torch.linspace(0.0, 1.0, n_frames, dtype=torch.float32)
        time_ramp = torch.sin(time_ramp * (np.pi / 2.0)) ** 2  # equal-power
        # Energy per bin
        ea = (A.abs() ** 2).mean(0)  # (freq, time)
        eb = (B.abs() ** 2).mean(0)
        # Per-bin blend bias: 0 (use A) → 1 (use B), shaped by sigmoid of
        # energy gap so the higher-energy clip "wins" each bin's phase.
        gap = (eb - ea) / (ea + eb + 1e-9)
        bin_mask = torch.sigmoid(gap * sigmoid_k)
        # Combine time ramp and bin mask multiplicatively
        mask = bin_mask * time_ramp.unsqueeze(0)
        mask = mask.unsqueeze(0)  # (1, freq, time) broadcast to channels
        # Phase: take A's phase where mask<0.5, else B's (avoids cancellation)
        phase_a = torch.angle(A)
        phase_b = torch.angle(B)
        use_b = mask > 0.5
        phase = torch.where(use_b, phase_b, phase_a)
        # Magnitude: linear blend
        mag = A.abs() * (1.0 - mask) + B.abs() * mask
        C = mag * torch.exp(1j * phase)
        out = torch.istft(C, n_fft=n_fft, hop_length=hop, window=win,
                           length=n)
        return out.numpy().astype(np.float32)
    except Exception:
        ramp = _equal_power_ramp(n)
        return (a[:, :n] * (1.0 - ramp) + b[:, :n] * ramp).astype(np.float32)
