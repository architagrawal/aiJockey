"""Stable Audio Open 1.0 / Small wrapper for bridge generation.

Replaces ACE-Step path. Stability AI Community License: commercial-OK
below $1M ARR, CC-trained data (cleanest provenance), BPM/key honored
in text prompt, ~6 GB DiT + 14 GB VAE peak.

Env knobs:
    AIJOCKEY_STABLE_AUDIO_ENABLE 0|1  default 0
    AIJOCKEY_STABLE_AUDIO_MODEL  hf id default 'stabilityai/stable-audio-open-1.0'
    AIJOCKEY_STABLE_AUDIO_STEPS  int  default 100
    AIJOCKEY_STABLE_AUDIO_CFG    float default 7.0
    AIJOCKEY_STABLE_AUDIO_DEVICE str  default 'cuda' if available

Install (MI300X):
    pip install stable-audio-tools
    # weights pull on first use via HF, accept license once
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_PIPE = None
_LOAD_FAILED = False


def enabled() -> bool:
    if os.environ.get("AIJOCKEY_STABLE_AUDIO_ENABLE", "0") != "1":
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
            from stable_audio_tools import get_pretrained_model  # type: ignore
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = os.environ.get("AIJOCKEY_STABLE_AUDIO_MODEL",
                                        "stabilityai/stable-audio-open-1.0")
            model, model_config = get_pretrained_model(model_id)
            sample_rate = model_config["sample_rate"]
            sample_size = model_config["sample_size"]
            if device == "cuda":
                model = model.cuda()
            _PIPE = {"model": model, "config": model_config,
                      "sr": sample_rate, "sample_size": sample_size,
                      "device": device}
            print(f"[stable_audio] loaded {model_id} on {device} "
                  f"(sr={sample_rate})")
            return _PIPE
        except Exception as e:
            print(f"[stable_audio] load failed: {e}")
            _LOAD_FAILED = True
            return None


def generate_bridge(*, bpm: float, key: str,
                     duration_seconds: float,
                     caption: str,
                     out_path: str | Path,
                     steps: int | None = None,
                     guidance_scale: float | None = None,
                     seed: int | None = None,
                     negative_prompt: str | None = None) -> str | None:
    """Synthesize a short bridge clip.

    Prompt format used internally (concat):
        "<bpm> BPM <key>. <caption>"
    Stability's T5 conditioner honors BPM/key strings.

    Returns out_path on success, None on failure.
    """
    if not enabled():
        return None
    pipe = _load()
    if pipe is None:
        return None
    if steps is None:
        steps = int(os.environ.get("AIJOCKEY_STABLE_AUDIO_STEPS", "100"))
    if guidance_scale is None:
        guidance_scale = float(os.environ.get("AIJOCKEY_STABLE_AUDIO_CFG", "7.0"))
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import torchaudio
        from stable_audio_tools.inference.generation import (
            generate_diffusion_cond,  # type: ignore
        )
        bpm_i = int(round(float(bpm))) if bpm else None
        prompt_parts = []
        if bpm_i:
            prompt_parts.append(f"{bpm_i} BPM")
        if key:
            prompt_parts.append(str(key))
        if caption:
            prompt_parts.append(str(caption))
        prompt = ". ".join(prompt_parts)
        cond = [{"prompt": prompt,
                  "seconds_start": 0,
                  "seconds_total": float(duration_seconds)}]
        gen_kwargs = dict(
            model=pipe["model"],
            steps=steps,
            cfg_scale=guidance_scale,
            conditioning=cond,
            sample_size=pipe["sample_size"],
            sigma_min=0.3, sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=pipe["device"],
        )
        if seed is not None:
            gen_kwargs["seed"] = int(seed)
        if negative_prompt:
            gen_kwargs["negative_conditioning"] = [
                {"prompt": negative_prompt,
                 "seconds_start": 0,
                 "seconds_total": float(duration_seconds)},
            ]
        with torch.inference_mode():
            audio = generate_diffusion_cond(**gen_kwargs)
        # audio: (1, 2, T) on device — convert + write
        audio = audio.squeeze(0).clamp(-1, 1).to(torch.float32).cpu()
        # Trim to requested duration
        n = int(pipe["sr"] * duration_seconds)
        audio = audio[:, :n]
        torchaudio.save(out_path, audio, pipe["sr"])
        return out_path if Path(out_path).exists() else None
    except Exception as e:
        print(f"[stable_audio] generate failed for {out_path}: {e}")
        return None
