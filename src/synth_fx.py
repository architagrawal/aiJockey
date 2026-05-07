"""
Procedural FX synthesis — always-available DJ sound effects with no
licensing, no downloads, no external dependencies beyond numpy/scipy.

All functions return stereo np.ndarray of shape (2, T) at SR=44100.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt

SR = 44100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lp(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(4, max(20.0, min(cutoff, sr * 0.49)), btype='low',
                 fs=sr, output='sos')
    return np.stack([sosfilt(sos, ch) for ch in x])


def _hp(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(4, max(20.0, min(cutoff, sr * 0.49)), btype='high',
                 fs=sr, output='sos')
    return np.stack([sosfilt(sos, ch) for ch in x])


def _bp(x: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    low = max(20.0, low)
    high = min(sr * 0.49, max(low + 10, high))
    sos = butter(4, [low, high], btype='band', fs=sr, output='sos')
    return np.stack([sosfilt(sos, ch) for ch in x])


def _stereo_noise(n: int, gain: float = 0.1) -> np.ndarray:
    return (np.random.randn(2, n) * gain).astype(np.float32)


def _exp_env(n: int, attack_frac: float = 0.05, release_curve: float = 3.0) -> np.ndarray:
    a = max(1, int(n * attack_frac))
    attack = np.linspace(0.0, 1.0, a, dtype=np.float32)
    release = np.exp(-np.linspace(0, release_curve, n - a)).astype(np.float32)
    return np.concatenate([attack, release])


# ---------------------------------------------------------------------------
# FX
# ---------------------------------------------------------------------------

def riser_uplift(beats: float, bpm: float = 128.0,
                 cutoff_start: float = 200.0, cutoff_end: float = 10000.0,
                 gain_start: float = 0.2, gain_end: float = 0.9) -> np.ndarray:
    """White-noise riser with rising LP cutoff + rising gain. Build into a drop."""
    beat_dur = 60.0 / max(bpm, 1.0)
    n = max(1, int(beats * beat_dur * SR))
    noise = _stereo_noise(n, 0.2)
    out = np.zeros_like(noise)
    chunk = max(1, SR // 50)
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        progress = i / max(1, n)
        cutoff = cutoff_start + (cutoff_end - cutoff_start) * progress
        gain = gain_start + (gain_end - gain_start) * progress
        out[:, i:end] = _lp(noise[:, i:end], SR, cutoff) * gain
    return out


def downsweep(beats: float, bpm: float = 128.0,
              freq_start: float = 8000.0, freq_end: float = 60.0) -> np.ndarray:
    """Downward filter sweep on noise. Outro / breakdown lead-in."""
    beat_dur = 60.0 / max(bpm, 1.0)
    n = max(1, int(beats * beat_dur * SR))
    noise = _stereo_noise(n, 0.25)
    out = np.zeros_like(noise)
    chunk = max(1, SR // 50)
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        progress = i / max(1, n)
        cutoff = freq_start - (freq_start - freq_end) * progress
        gain = 0.7 - 0.5 * progress
        out[:, i:end] = _lp(noise[:, i:end], SR, cutoff) * gain
    return out


def snare_roll(beats: float = 4.0, bpm: float = 128.0,
               start_subdiv: int = 4, end_subdiv: int = 32) -> np.ndarray:
    """
    Accelerating snare roll — subdivisions go from start_subdiv to end_subdiv
    over `beats`. Each hit = filtered noise burst.
    """
    beat_dur = 60.0 / max(bpm, 1.0)
    total_n = max(1, int(beats * beat_dur * SR))
    out = np.zeros((2, total_n), dtype=np.float32)
    # Determine hit times via interpolated subdivision
    hits: list[float] = []
    t = 0.0
    while t < beats:
        progress = t / beats
        subdiv = start_subdiv + (end_subdiv - start_subdiv) * progress
        step = 1.0 / subdiv
        hits.append(t)
        t += step
    # Each hit: short bandpass-noise burst
    hit_n = int(0.04 * SR)  # 40 ms
    env = _exp_env(hit_n, attack_frac=0.05, release_curve=4.0)
    for ht in hits:
        s = int(ht * beat_dur * SR)
        if s + hit_n > total_n:
            break
        burst = _bp(_stereo_noise(hit_n, 0.6), SR, 800, 4000)
        out[:, s:s + hit_n] += burst * env
    # Gradual gain build
    n = total_n
    gain = np.linspace(0.3, 1.0, n).astype(np.float32)
    return np.clip(out * gain, -1.0, 1.0)


def sub_drop(beats: float = 1.0, bpm: float = 128.0,
             freq_start: float = 80.0, freq_end: float = 25.0) -> np.ndarray:
    """Sub-bass drop — sine sweep down. Used for impact moments."""
    beat_dur = 60.0 / max(bpm, 1.0)
    n = max(1, int(beats * beat_dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    progress = t / max(t[-1], 1e-6)
    freq = freq_start - (freq_start - freq_end) * progress
    phase = 2 * np.pi * np.cumsum(freq) / SR
    sig = (np.sin(phase) * 0.8).astype(np.float32)
    env = _exp_env(n, attack_frac=0.02, release_curve=2.0)
    sig = sig * env
    return np.stack([sig, sig])


def impact(decay_sec: float = 1.5) -> np.ndarray:
    """White-noise impact with long reverb-ish decay. Drop re-entry punctuation."""
    n = int(decay_sec * SR)
    noise = _stereo_noise(n, 0.8)
    # Multi-band: punchy lows + bright highs, decaying
    low = _lp(noise, SR, 200) * 1.5
    mid = _bp(noise, SR, 200, 4000) * 0.6
    high = _hp(noise, SR, 4000) * 0.4
    sig = (low + mid + high).astype(np.float32)
    env = _exp_env(n, attack_frac=0.001, release_curve=4.0)
    return np.clip(sig * env, -1.0, 1.0)


def vinyl_stop(duration_sec: float = 0.4, base_freq: float = 200.0) -> np.ndarray:
    """
    Vinyl-stop FX — pitched tone descends to near-zero rapidly while
    amplitude fades. Use during spinback transition.
    """
    n = max(1, int(duration_sec * SR))
    t = np.arange(n, dtype=np.float32) / SR
    progress = t / max(t[-1], 1e-6)
    freq = base_freq * (1.0 - progress) ** 2
    phase = 2 * np.pi * np.cumsum(freq) / SR
    sig = (np.sin(phase) * 0.6).astype(np.float32)
    # Add noise tail (vinyl scratchiness)
    noise = _stereo_noise(n, 0.05)[0]
    sig = sig + noise * (1.0 - progress)
    env = (1.0 - progress ** 0.5).astype(np.float32)
    sig = sig * env
    return np.stack([sig, sig])


def airhorn(beats: float = 1.0, bpm: float = 128.0) -> np.ndarray:
    """Synthetic air-horn — saw wave + filter resonance. Hype moment."""
    beat_dur = 60.0 / max(bpm, 1.0)
    n = max(1, int(beats * beat_dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    # Two slightly detuned saws
    saw1 = 2 * (t * 440 - np.floor(0.5 + t * 440))
    saw2 = 2 * (t * 442 - np.floor(0.5 + t * 442))
    sig = (saw1 + saw2) * 0.4
    sig = _lp(np.stack([sig, sig]), SR, 2500)
    env = _exp_env(n, attack_frac=0.05, release_curve=1.5)
    return (sig * env).astype(np.float32)


def hihat_roll(beats: float = 2.0, bpm: float = 128.0,
               subdiv: int = 16) -> np.ndarray:
    """Closed hi-hat roll. Build moment."""
    beat_dur = 60.0 / max(bpm, 1.0)
    total_n = max(1, int(beats * beat_dur * SR))
    out = np.zeros((2, total_n), dtype=np.float32)
    hit_n = int(0.025 * SR)
    env = _exp_env(hit_n, attack_frac=0.02, release_curve=6.0)
    n_hits = int(beats * subdiv / 4)
    for i in range(n_hits):
        ht = i / subdiv * 4
        s = int(ht * beat_dur * SR)
        if s + hit_n > total_n:
            break
        burst = _hp(_stereo_noise(hit_n, 0.5), SR, 6000)
        out[:, s:s + hit_n] += burst * env * 0.7
    return np.clip(out, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Registry — type -> generator
# ---------------------------------------------------------------------------

SYNTHESIZERS = {
    'risers': riser_uplift,
    'sweeps': downsweep,
    'snare_rolls': snare_roll,
    'sub_drops': sub_drop,
    'impacts': impact,
    'vinyl': vinyl_stop,
    'airhorns': airhorn,
    'hihat_rolls': hihat_roll,
}


def synthesize(fx_type: str, bpm: float = 128.0, beats: float = 1.0) -> np.ndarray | None:
    """
    Generate FX of given type at given tempo + length.
    Returns None if unknown type.
    """
    fn = SYNTHESIZERS.get(fx_type)
    if fn is None:
        return None
    # Dispatch with appropriate args per fx type
    if fx_type == 'impacts':
        return fn(decay_sec=max(0.3, beats * 60.0 / bpm))
    if fx_type == 'vinyl':
        return fn(duration_sec=max(0.2, beats * 60.0 / bpm))
    if fx_type == 'sub_drops':
        return fn(beats=beats, bpm=bpm)
    return fn(beats=beats, bpm=bpm)
