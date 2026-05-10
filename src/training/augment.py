"""Audio augmentations for training data efficiency.

Apply in dataloader, on-the-fly. Each augmentation maps (audio, sr) -> audio.
Stack random subsets per batch to multiply effective dataset 5-10x.

Reference: Phase A polish plan §16.1.
"""
from __future__ import annotations
import io
import random
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Time-domain
# ---------------------------------------------------------------------------

def pitch_shift(wav: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """rubberband pitch shift. wav (C, T) -> (C, T)."""
    if abs(semitones) < 0.01:
        return wav
    try:
        import pyrubberband as pyrb
    except ImportError:
        return wav
    x = wav.T.astype(np.float32)
    try:
        y = pyrb.pitch_shift(x, sr, semitones)
        return y.T.astype(np.float32)
    except Exception:
        return wav


def time_stretch(wav: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """rubberband time stretch. rate>1 = faster. wav (C, T) -> (C, T')."""
    if abs(rate - 1.0) < 0.005:
        return wav
    try:
        import pyrubberband as pyrb
    except ImportError:
        return wav
    x = wav.T.astype(np.float32)
    try:
        y = pyrb.time_stretch(x, sr, rate)
        return y.T.astype(np.float32)
    except Exception:
        return wav


def speed_perturb(wav: np.ndarray, sr: int, factor: float) -> tuple[np.ndarray, int]:
    """Resample-based speed change (alters both pitch + tempo).
    Returns (audio, sr) where audio has been resampled to factor*sr then
    relabeled at the original sr — net effect is faster/slower playback
    with pitch shift, length scaled by 1/factor.
    """
    if abs(factor - 1.0) < 0.005:
        return wav, sr
    try:
        import librosa
    except ImportError:
        return wav, sr
    target_sr = int(sr * factor)
    # Last axis is time for both (T,) and (C, T) layouts.
    y = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr,
                         axis=-1)
    return y.astype(np.float32), sr


def gain_jitter(wav: np.ndarray, db: float) -> np.ndarray:
    return (wav * (10.0 ** (db / 20.0))).astype(np.float32)


def mixup(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Linear interpolate two equal-length audio arrays. alpha in [0,1]."""
    n = min(a.shape[-1], b.shape[-1])
    return (alpha * a[..., :n] + (1.0 - alpha) * b[..., :n]).astype(np.float32)


# ---------------------------------------------------------------------------
# Frequency-domain (SpecAugment)
# ---------------------------------------------------------------------------

def spec_augment(mel: torch.Tensor, n_time_masks: int = 2, n_freq_masks: int = 2,
                 max_time_mask: int = 30, max_freq_mask: int = 15) -> torch.Tensor:
    """Time + frequency masking on mel-spectrogram. mel: (..., F, T)."""
    out = mel.clone()
    F, T = out.shape[-2], out.shape[-1]
    for _ in range(n_freq_masks):
        f = random.randint(0, max_freq_mask)
        if f == 0:
            continue
        f0 = random.randint(0, max(0, F - f))
        out[..., f0:f0 + f, :] = 0
    for _ in range(n_time_masks):
        t = random.randint(0, max_time_mask)
        if t == 0:
            continue
        t0 = random.randint(0, max(0, T - t))
        out[..., :, t0:t0 + t] = 0
    return out


# ---------------------------------------------------------------------------
# Codec roundtrip — kills codec bias (STATUS bug #3 root cause)
# ---------------------------------------------------------------------------

def codec_roundtrip(wav: np.ndarray, sr: int, codec: str = 'mp3',
                    bitrate: int = 192) -> np.ndarray:
    """Encode + decode in memory to introduce codec artifacts. Forces critic
    to learn codec-invariant features.
    """
    try:
        import torchaudio
    except ImportError:
        return wav
    if wav.ndim == 1:
        wav = wav[None, :]
    t = torch.from_numpy(wav.astype(np.float32))
    buf = io.BytesIO()
    try:
        torchaudio.save(buf, t, sr, format=codec,
                        compression=int(bitrate) if codec == 'mp3' else None)
        buf.seek(0)
        out, _ = torchaudio.load(buf)
        return out.numpy().astype(np.float32)
    except Exception:
        return wav.astype(np.float32)


# ---------------------------------------------------------------------------
# Composite — random augmentation chain for training pipelines
# ---------------------------------------------------------------------------

class AugChain:
    """Stochastic augmentation pipeline. Each step has independent probability.

    Use as: augmented = AugChain(p_pitch=0.5, ...)(wav, sr)
    """

    def __init__(self,
                 p_pitch: float = 0.5, pitch_range: float = 2.0,
                 p_stretch: float = 0.5, stretch_range: float = 0.1,
                 p_gain: float = 0.7, gain_db: float = 6.0,
                 p_codec: float = 0.3, codec: str = 'mp3', bitrate: int = 192):
        self.p_pitch = p_pitch
        self.pitch_range = pitch_range
        self.p_stretch = p_stretch
        self.stretch_range = stretch_range
        self.p_gain = p_gain
        self.gain_db = gain_db
        self.p_codec = p_codec
        self.codec = codec
        self.bitrate = bitrate

    def __call__(self, wav: np.ndarray, sr: int = 44100) -> np.ndarray:
        if random.random() < self.p_pitch:
            wav = pitch_shift(wav, sr, random.uniform(-self.pitch_range, self.pitch_range))
        if random.random() < self.p_stretch:
            rate = 1.0 + random.uniform(-self.stretch_range, self.stretch_range)
            wav = time_stretch(wav, sr, rate)
        if random.random() < self.p_gain:
            wav = gain_jitter(wav, random.uniform(-self.gain_db, self.gain_db))
        if random.random() < self.p_codec:
            wav = codec_roundtrip(wav, sr, codec=self.codec, bitrate=self.bitrate)
        return wav


__all__ = [
    'pitch_shift', 'time_stretch', 'speed_perturb', 'gain_jitter',
    'mixup', 'spec_augment', 'codec_roundtrip', 'AugChain',
]
