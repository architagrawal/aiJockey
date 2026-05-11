"""Glitch repair via VampNet inpainting on probe-flagged junctions.

When audio_probes flags a junction with severity > threshold, regen
that 0.5-1.5s window via VampNet's masked-inpaint, keeping surrounding
context as constraint. Replaces only the broken bit.

Toggle: AIJOCKEY_VAMPNET_REPAIR=1, AIJOCKEY_VAMPNET_REPAIR_THRESH=0.7
"""
from __future__ import annotations

import os

import numpy as np


def enabled() -> bool:
    if os.environ.get("AIJOCKEY_VAMPNET_REPAIR", "0") != "1":
        return False
    try:
        from vampnet_wrapper import enabled as vw_en
        return vw_en()
    except Exception:
        return False


def repair_glitch(audio: np.ndarray, sr: int,
                    t_glitch_s: float,
                    window_s: float = 0.8,
                    context_s: float = 2.5,
                    temperature: float = 0.95) -> np.ndarray:
    """Inpaint a `window_s` slice around `t_glitch_s` using VampNet.

    Args:
        audio: stereo (2, n) waveform.
        sr: sample rate.
        t_glitch_s: center time of glitch (sec).
        window_s: width of region to regenerate.
        context_s: how much surrounding context VampNet sees on each side.

    Returns repaired audio (same shape). On any failure, returns input.
    """
    if not enabled():
        return audio
    try:
        from vampnet_wrapper import _load
        pipe = _load()
        if pipe is None:
            return audio
        iface = pipe["interface"]
        import torch
        from audiotools import AudioSignal  # type: ignore

        n = audio.shape[1]
        center = int(t_glitch_s * sr)
        half = int(window_s * sr / 2)
        ctx = int(context_s * sr)
        lo = max(0, center - half - ctx)
        hi = min(n, center + half + ctx)
        snippet = audio[:, lo:hi]
        if snippet.shape[1] < int(sr * 1.0):
            return audio

        # Mono down for VampNet (it's mono-trained)
        mono = snippet.mean(0, keepdims=True).astype(np.float32)
        sig = AudioSignal(mono, sample_rate=sr).to(iface.device)
        z = iface.encode(sig)
        n_tokens = z.shape[-1]
        # Compute token positions for the glitch window
        snip_n = snippet.shape[1]
        glitch_start = max(0, center - half - lo)
        glitch_end = min(snip_n, center + half - lo)
        tok_lo = int(round(glitch_start / snip_n * n_tokens))
        tok_hi = int(round(glitch_end / snip_n * n_tokens))
        if tok_hi <= tok_lo:
            return audio
        mask = torch.zeros_like(z, dtype=torch.long)
        mask[..., tok_lo:tok_hi] = 1
        z_c = iface.coarse_vamp(z, mask=mask, temperature=float(temperature))
        z_out = iface.coarse_to_fine(z_c, temperature=float(temperature))
        sig_out = iface.decode(z_out)
        wav = sig_out.samples.detach().cpu().numpy().squeeze()
        if wav.ndim == 2:
            wav = wav.mean(0)
        # Splice repaired window back into original
        repaired = audio.copy()
        # Stretch wav to snippet length if VampNet returned different size
        if wav.shape[0] != snippet.shape[1]:
            # Pad / truncate
            if wav.shape[0] < snippet.shape[1]:
                wav = np.pad(wav, (0, snippet.shape[1] - wav.shape[0]))
            else:
                wav = wav[: snippet.shape[1]]
        # Build stereo by duplicating mono output, then crossfade only the glitch region
        wav_st = np.stack([wav, wav], axis=0).astype(np.float32)
        gs = glitch_start
        ge = glitch_end
        # Linear xfade edges to hide seam
        edge = max(1, int(sr * 0.05))
        for i in range(edge):
            t = i / edge
            repaired[:, lo + gs + i] = ((1 - t) * repaired[:, lo + gs + i]
                                          + t * wav_st[:, gs + i])
        repaired[:, lo + gs + edge : lo + ge - edge] = wav_st[:, gs + edge : ge - edge]
        for i in range(edge):
            t = i / edge
            j = ge - edge + i
            repaired[:, lo + j] = ((1 - t) * wav_st[:, j]
                                     + t * repaired[:, lo + j])
        return repaired.astype(np.float32)
    except Exception as e:
        print(f"[vampnet_repair] failed at {t_glitch_s:.2f}s: {e}")
        return audio
