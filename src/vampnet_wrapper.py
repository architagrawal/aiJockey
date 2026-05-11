"""VampNet wrapper for music-inpainting bridge synthesis.

VampNet (Flores Garcia et al., ISMIR 2023, MIT) is a masked-token
music model trained for inpainting. We exploit its native gap-fill
behavior for clip-to-clip DJ transitions: feed [clip_A_tail, ZERO_GAP,
clip_B_head] as a single signal, encode through VampNet's codec
(DAC-derived), construct a mask that PRESERVES context tokens but
masks the gap tokens, and let VampNet's coarse + coarse-to-fine
stages fill the gap with musically-coherent content.

Output: 44.1 kHz stereo waveform of the bridge region only (or full
A+bridge+B if requested).

Env knobs:
    AIJOCKEY_VAMPNET_ENABLE   0|1   default 0
    AIJOCKEY_VAMPNET_CKPT_DIR str   default 'hugofloresgarcia/vampnet'
                                       (HF repo id with all checkpoints)
    AIJOCKEY_VAMPNET_DEVICE   str   default 'cuda' if available

Public API:
    enabled() -> bool
    generate_bridge(a_tail_path, b_head_path, gap_seconds, out_path,
                     ...) -> str | None
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_PIPE = None
_LOAD_FAILED = False


def enabled() -> bool:
    if os.environ.get("AIJOCKEY_VAMPNET_ENABLE", "0") != "1":
        return False
    return not _LOAD_FAILED


def _load(device: str | None = None):
    global _PIPE, _LOAD_FAILED
    if _PIPE is not None:
        return _PIPE
    if _LOAD_FAILED:
        return None
    with _LOCK:
        if _PIPE is not None:
            return _PIPE
        try:
            import torch
            from vampnet.interface import Interface  # type: ignore
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpt_dir = os.environ.get("AIJOCKEY_VAMPNET_CKPT_DIR")
            iface_kwargs = {"device": device}
            if ckpt_dir:
                # Local checkpoint directory layout: codec.pth / coarse.pth
                # / c2f.pth — match VampNet's Interface signature.
                iface_kwargs.update({
                    "codec_ckpt": str(Path(ckpt_dir) / "codec.pth"),
                    "coarse_ckpt": str(Path(ckpt_dir) / "coarse.pth"),
                    "coarse2fine_ckpt": str(Path(ckpt_dir) / "c2f.pth"),
                })
            iface = Interface(**iface_kwargs)
            _PIPE = {"interface": iface, "device": device,
                      "sr": getattr(iface, "codec", None) and
                            getattr(iface.codec, "sample_rate", 44100) or 44100}
            print(f"[vampnet] loaded on {device}")
            return _PIPE
        except Exception as e:
            print(f"[vampnet] load failed ({e.__class__.__name__}: {e})")
            _LOAD_FAILED = True
            return None


def _load_audio_mono(path: str, target_sr: int = 44100,
                      seconds: float | None = None) -> "Any":
    import librosa
    duration = float(seconds) if seconds else None
    wav, _ = librosa.load(str(path), sr=target_sr, mono=True, duration=duration)
    return wav


def generate_bridge(a_tail_path: str | Path,
                     b_head_path: str | Path,
                     gap_seconds: float,
                     out_path: str | Path,
                     *,
                     context_seconds: float = 4.0,
                     temperature: float = 1.0,
                     top_p: float = 0.9,
                     mask_top_amount: float = 1.0,
                     return_full: bool = False) -> str | None:
    """Inpaint a `gap_seconds` bridge between two clips.

    Args:
        a_tail_path, b_head_path: audio paths to source A's tail and
            B's head. Each is loaded mono at 44.1 kHz, last/first
            `context_seconds` used as context.
        gap_seconds: bridge length to synthesize.
        out_path: write WAV here.
        context_seconds: how much of A/B to keep as inpaint context.
        temperature, top_p: VampNet decode params.
        mask_top_amount: 1.0 = mask all tokens in gap; <1 = keep some
            random anchor tokens (Vamp's "iterative" pattern).
        return_full: if True, return A_context + bridge + B_context as
            one continuous waveform. If False, just the bridge region.

    Returns:
        out_path on success, None on failure.
    """
    if not enabled():
        return None
    pipe = _load()
    if pipe is None:
        return None
    try:
        import numpy as np
        import torch
        import torchaudio
        from audiotools import AudioSignal  # type: ignore  (vampnet dep)

        iface = pipe["interface"]
        sr = int(pipe["sr"])

        # Load A_tail (last context_seconds) and B_head (first context_seconds)
        a_full = _load_audio_mono(str(a_tail_path), target_sr=sr)
        b_full = _load_audio_mono(str(b_head_path), target_sr=sr)
        n_ctx = int(sr * context_seconds)
        a_tail = a_full[-n_ctx:] if len(a_full) >= n_ctx else a_full
        b_head = b_full[:n_ctx] if len(b_full) >= n_ctx else b_full

        n_gap = int(sr * gap_seconds)
        gap = np.zeros(n_gap, dtype=np.float32)

        full_in = np.concatenate([a_tail, gap, b_head]).astype(np.float32)
        sig_in = AudioSignal(full_in[None, :], sample_rate=sr)
        sig_in = sig_in.to(pipe["device"])

        # Encode to codec tokens
        z = iface.encode(sig_in)   # (1, n_codebooks, T_tokens)

        # Build mask: 0 means keep, 1 means inpaint
        n_tokens = z.shape[-1]
        # Convert sample positions to token positions
        a_tokens = int(round(len(a_tail) / len(full_in) * n_tokens))
        gap_tokens = int(round(len(gap) / len(full_in) * n_tokens))
        mask = torch.zeros_like(z, dtype=torch.long)
        mask[..., a_tokens : a_tokens + gap_tokens] = 1
        if mask_top_amount < 1.0:
            # Randomly unmask some anchor tokens to give VampNet rhythm hints
            rng = torch.rand_like(mask.float())
            keep_anchor = rng < (1.0 - mask_top_amount)
            mask = mask & (~keep_anchor.bool()).long()

        # Run coarse + c2f vamp
        z_out = iface.coarse_to_fine(
            iface.coarse_vamp(
                z, mask=mask,
                temperature=float(temperature),
                top_p=float(top_p),
                return_signal=False,
            ),
            temperature=float(temperature),
        )
        sig_out = iface.decode(z_out)

        # Trim to requested region
        wav_out = sig_out.samples.detach().cpu().numpy().squeeze()
        if wav_out.ndim == 2:
            wav_out = wav_out.mean(0)  # mono safe
        if return_full:
            final = wav_out
        else:
            a_n = len(a_tail)
            final = wav_out[a_n : a_n + n_gap]
        # Make stereo by duplicating
        if final.ndim == 1:
            final = np.stack([final, final], axis=0)
        final = np.clip(final, -1.0, 1.0).astype(np.float32)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_path), torch.from_numpy(final), sr)
        return str(out_path) if Path(out_path).exists() else None
    except Exception as e:
        print(f"[vampnet] generate failed: {e}")
        return None
