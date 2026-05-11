"""Spectral hold / freeze transition.

Cheap rule-based glue for genre jumps: take a short tail of clip A,
freeze its spectrum (hold magnitude, randomize phase), stretch it
across the gap as a tonal pad while clip B fades in underneath. Hides
abrupt timbral shifts between e.g. EDM → ambient.

Pattern: short FFT window from A_tail → magnitude frozen, phase
randomized per frame → iSTFT → loop length → crossfade into B.

Toggle in callers via AIJOCKEY_SPECTRAL_HOLD; pure function below.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def freeze_pad(source: np.ndarray, sr: int, duration_seconds: float,
                n_fft: int = 4096, hop: int = 1024,
                seed: int = 42) -> np.ndarray:
    """Generate a `duration_seconds` tonal pad from `source`'s spectrum.

    Args:
        source: stereo (2, n) waveform — last ~1s used as frozen frame.
        sr: sample rate.
        duration_seconds: output length.
        n_fft, hop: STFT sizing.
        seed: phase-randomization seed (reproducible).

    Returns:
        (2, sr*duration_seconds) stereo pad.
    """
    n_out = int(sr * duration_seconds)
    if not _HAS_TORCH or source.shape[1] < n_fft:
        return np.zeros((source.shape[0], n_out), dtype=np.float32)
    try:
        tail = source[:, -n_fft:]
        t = torch.from_numpy(tail.astype(np.float32))
        win = torch.hann_window(n_fft)
        # One-frame STFT to capture spectral envelope
        S = torch.stft(t, n_fft=n_fft, hop_length=hop, window=win,
                        return_complex=True, center=True)
        mag = S.abs()
        # Build a long synthetic STFT by replicating magnitude over
        # required frames, with fresh randomized phase per frame.
        n_frames_target = max(1, n_out // hop + 4)
        mag_repeated = mag[:, :, -1:].expand(-1, -1, n_frames_target)
        rng = np.random.default_rng(seed)
        phase = torch.from_numpy(
            rng.uniform(-np.pi, np.pi,
                         size=mag_repeated.shape).astype(np.float32))
        synthetic = mag_repeated * torch.exp(1j * phase)
        out = torch.istft(synthetic, n_fft=n_fft, hop_length=hop, window=win,
                           length=n_out)
        # Gentle envelope: short fade-in and fade-out so pad doesn't pop.
        ramp_n = min(int(sr * 0.1), n_out // 4)
        env = np.ones(n_out, dtype=np.float32)
        if ramp_n > 0:
            env[:ramp_n] = np.linspace(0.0, 1.0, ramp_n)
            env[-ramp_n:] = np.linspace(1.0, 0.0, ramp_n)
        pad = out.numpy().astype(np.float32) * env[None, :]
        return pad
    except Exception:
        return np.zeros((source.shape[0], n_out), dtype=np.float32)


def spectral_hold_transition(out_full: np.ndarray, in_full: np.ndarray,
                              sr: int, hold_seconds: float = 1.0,
                              xfade_seconds: float = 0.5) -> np.ndarray:
    """Glue A→B via a frozen-spectrum pad of length `hold_seconds`.

    Output = A + [pad of A's spectrum] + B (with the pad crossfaded
    into B's first xfade_seconds).
    """
    n_pad = int(sr * hold_seconds)
    n_xf = int(sr * xfade_seconds)
    if n_pad <= 0:
        return np.concatenate([out_full, in_full], axis=1)
    pad = freeze_pad(out_full, sr, hold_seconds)
    # Crossfade pad → in_full
    n_xf = min(n_xf, pad.shape[1], in_full.shape[1])
    if n_xf > 0:
        t = np.linspace(0, np.pi / 2, n_xf, dtype=np.float32)
        fade_out = np.cos(t)
        fade_in = np.sin(t)
        blended = pad[:, -n_xf:] * fade_out + in_full[:, :n_xf] * fade_in
        pad_pre = pad[:, :-n_xf]
        in_post = in_full[:, n_xf:]
        return np.concatenate([out_full, pad_pre, blended, in_post],
                                axis=1).astype(np.float32)
    return np.concatenate([out_full, pad, in_full], axis=1).astype(np.float32)
