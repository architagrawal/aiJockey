"""ACE-Step generative bridge wrapper for AiJockey.

ACE-Step is an Apache-2.0 music-generation diffusion model (3.5B params)
that takes structured metadata (bpm, keyscale, duration, vocal_language)
plus a caption and emits 44.1 kHz stereo WAV.

Use case in AiJockey: when the planner needs a transition between two
clips that have no library-clip bridge (incompatible BPM, missing genre
in pool, or vocal-clash), synthesize a short bridge with matched BPM/key.

This module is a *thin lazy wrapper*. ACE-Step is large (~12 GB VRAM
peak, ~30s first-load weights). We never import its pipeline at module
import — only when generate_bridge() is first called.

Env knobs:
    AIJOCKEY_ACE_STEP_ENABLE     0|1   default 0 (opt-in)
    AIJOCKEY_ACE_STEP_MODEL      hf id default 'ACE-Step/ACE-Step-v1-3.5B'
    AIJOCKEY_ACE_STEP_STEPS      int   default 27 (paper default)
    AIJOCKEY_ACE_STEP_CFG        float default 7.5
    AIJOCKEY_ACE_STEP_DEVICE     str   default 'cuda' if available

Install (MI300X / ROCm path):
    pip install -r requirements-rocm.txt   # from ACE-Step-1.5 repo
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_PIPE = None
_LOAD_FAILED = False


def enabled() -> bool:
    """True if env opts in and prior load did not fail."""
    if os.environ.get("AIJOCKEY_ACE_STEP_ENABLE", "0") != "1":
        return False
    return not _LOAD_FAILED


def _load(device: str | None = None):
    """Lazy init ACE-Step pipeline. Returns pipe or None."""
    global _PIPE, _LOAD_FAILED
    if _PIPE is not None:
        return _PIPE
    if _LOAD_FAILED:
        return None
    with _LOCK:
        if _PIPE is not None:
            return _PIPE
        if _LOAD_FAILED:
            return None
        try:
            import torch
            from acestep.pipeline_ace_step import ACEStepPipeline  # type: ignore
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = os.environ.get(
                "AIJOCKEY_ACE_STEP_MODEL", "ACE-Step/ACE-Step-v1-3.5B")
            pipe = ACEStepPipeline(checkpoint_dir=None,
                                    model_id_or_path=model_id,
                                    device=device,
                                    torch_dtype=(torch.float16
                                                  if device == "cuda"
                                                  else torch.float32))
            _PIPE = {"pipeline": pipe, "device": device, "model_id": model_id}
            print(f"[ace_step] loaded {model_id} on {device}")
            return _PIPE
        except Exception as e:
            print(f"[ace_step] load failed ({e.__class__.__name__}: {e})")
            _LOAD_FAILED = True
            return None


def generate_bridge(*, bpm: float,
                     key: str,
                     duration_seconds: float,
                     caption: str,
                     out_path: str | Path,
                     steps: int | None = None,
                     guidance_scale: float | None = None,
                     vocal_language: str | None = None,
                     negative_prompt: str | None = None,
                     seed: int | None = None) -> str | None:
    """Synthesize a transition bridge.

    Args:
        bpm: target tempo in BPM (planner's target_bpm).
        key: musical key, accepts Camelot (e.g. "8B") OR letter form
             (e.g. "F minor"). ACE-Step uses "keyscale" param —
             we forward verbatim; caller may pre-translate Camelot.
        duration_seconds: output length (typically 16-30s for bridges).
        caption: free-text brief (e.g. "atmospheric uplifting trance pad").
                 Do NOT include BPM/key here — pass via metadata fields.
        out_path: where to write 44.1kHz stereo WAV.
        steps: denoising steps (default 27 from env).
        guidance_scale: CFG (default 7.5 from env).
        vocal_language: omit / None for instrumental.
        negative_prompt: optional CFG-negative caption (e.g. "drums,
                          vocals" to suppress those).
        seed: deterministic seed; None = random.

    Returns:
        Output path on success, None on failure.
    """
    if not enabled():
        return None
    pipe = _load()
    if pipe is None:
        return None
    if steps is None:
        try:
            steps = int(os.environ.get("AIJOCKEY_ACE_STEP_STEPS", "27"))
        except Exception:
            steps = 27
    if guidance_scale is None:
        try:
            guidance_scale = float(os.environ.get("AIJOCKEY_ACE_STEP_CFG", "7.5"))
        except Exception:
            guidance_scale = 7.5
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        kwargs = dict(
            caption=caption,
            bpm=int(round(float(bpm))) if bpm else None,
            keyscale=key,
            duration=float(duration_seconds),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            output_path=out_path,
        )
        if vocal_language:
            kwargs["vocal_language"] = vocal_language
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            kwargs["seed"] = int(seed)
        pipe["pipeline"].infer(**kwargs)
        if Path(out_path).exists():
            return out_path
        return None
    except Exception as e:
        print(f"[ace_step] generate failed for {out_path}: {e}")
        return None


# Camelot → letter helper (basic). Many DJ rigs use Camelot;
# ACE-Step understands letter+mode form better.
_CAMELOT_TO_KEY = {
    "1A": "Ab minor", "1B": "B major",
    "2A": "Eb minor", "2B": "F# major",
    "3A": "Bb minor", "3B": "Db major",
    "4A": "F minor",  "4B": "Ab major",
    "5A": "C minor",  "5B": "Eb major",
    "6A": "G minor",  "6B": "Bb major",
    "7A": "D minor",  "7B": "F major",
    "8A": "A minor",  "8B": "C major",
    "9A": "E minor",  "9B": "G major",
    "10A": "B minor", "10B": "D major",
    "11A": "F# minor", "11B": "A major",
    "12A": "Db minor", "12B": "E major",
}


def camelot_to_letter(camelot: str) -> str:
    """Convert Camelot (e.g. '8A') to 'A minor'. Pass-through unknown."""
    return _CAMELOT_TO_KEY.get((camelot or "").strip().upper(), camelot)
