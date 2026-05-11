"""InspireMusic-1.5B-48kHz wrapper for AR-streaming bridge generation.

FunAudioLLM (Tongyi Lab, Apache 2.0). Qwen2.5-based AR transformer +
super-res flow-matching head. 48 kHz output. AR generation is
naturally streamable for low-latency DJ use.

Env:
    AIJOCKEY_INSPIRE_ENABLE   0|1   default 0
    AIJOCKEY_INSPIRE_MODEL    hf id default 'FunAudioLLM/InspireMusic-1.5B-48k'
    AIJOCKEY_INSPIRE_DEVICE   str   default 'cuda' if available

Install (MI300X):
    pip install funmusic   # or clone FunAudioLLM/InspireMusic
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_PIPE = None
_LOAD_FAILED = False


def enabled() -> bool:
    if os.environ.get("AIJOCKEY_INSPIRE_ENABLE", "0") != "1":
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
            try:
                from inspiremusic.cli.inference import InspireMusicUnified as _InspireMusic  # type: ignore
            except Exception:
                from funmusic.inspiremusic import InspireMusicUnified as _InspireMusic  # type: ignore
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = os.environ.get(
                "AIJOCKEY_INSPIRE_MODEL",
                "FunAudioLLM/InspireMusic-1.5B-48k",
            )
            engine = _InspireMusic(model_dir=model_id, device=device)
            _PIPE = {"engine": engine, "device": device, "sr": 48000,
                      "model_id": model_id}
            print(f"[inspire_music] loaded {model_id} on {device}")
            return _PIPE
        except Exception as e:
            print(f"[inspire_music] load failed ({e.__class__.__name__}: {e})")
            _LOAD_FAILED = True
            return None


def generate_bridge(*, caption: str,
                     duration_seconds: float,
                     out_path: str | Path,
                     bpm: float | None = None,
                     key: str | None = None,
                     temperature: float = 0.9,
                     top_p: float = 0.9,
                     seed: int | None = None) -> str | None:
    """Synthesize an AR-streamed bridge clip at 48 kHz.

    BPM / key are passed as text-prompt hints since InspireMusic does
    not expose typed conditioning fields like ACE-Step.
    """
    if not enabled():
        return None
    pipe = _load()
    if pipe is None:
        return None
    try:
        import torchaudio
        import torch
        parts = []
        if bpm:
            parts.append(f"{int(round(float(bpm)))} BPM")
        if key:
            parts.append(str(key))
        if caption:
            parts.append(str(caption))
        prompt = ". ".join(parts)
        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        wav_t = pipe["engine"].inference(
            task="text_to_music",
            text=prompt,
            time_start=0.0,
            time_end=float(duration_seconds),
            output_sample_rate=pipe["sr"],
            seed=int(seed) if seed is not None else None,
            temperature=float(temperature),
            top_p=float(top_p),
        )
        if hasattr(wav_t, "cpu"):
            wav_t = wav_t.detach().cpu()
        if wav_t.dim() == 1:
            wav_t = wav_t.unsqueeze(0)
        if wav_t.shape[0] == 1:
            wav_t = wav_t.repeat(2, 1)   # duplicate to stereo
        wav_t = torch.clamp(wav_t, -1.0, 1.0).float()
        torchaudio.save(out_path, wav_t, pipe["sr"])
        return out_path if Path(out_path).exists() else None
    except Exception as e:
        print(f"[inspire_music] generate failed for {out_path}: {e}")
        return None
