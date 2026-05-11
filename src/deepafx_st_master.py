"""DeepAFx-ST learned mastering style-transfer wrapper.

Reference: Steinmetz, Adobe Research (arXiv 2207.08759). Differentiable
DSP chain (parametric EQ + multi-band compressor) whose parameters are
predicted from a *reference clip*. Replaces our hand-tuned multi-band
+ tape-sat chain.

Env knobs:
    AIJOCKEY_DEEPAFX_ENABLE   0|1  default 0
    AIJOCKEY_DEEPAFX_REF      path to reference audio (e.g. Tomorrowland
                                  mainstage reference)
    AIJOCKEY_DEEPAFX_CKPT     path to DeepAFx-ST weights
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_PIPE = None
_LOAD_FAILED = False


def enabled() -> bool:
    if os.environ.get("AIJOCKEY_DEEPAFX_ENABLE", "0") != "1":
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
            # DeepAFx-ST ships its model classes in `deepafx_st`. The repo
            # is github.com/adobe-research/DeepAFx-ST. Import path may
            # differ in their distribution; try common names.
            try:
                from deepafx_st.system import System as _System  # type: ignore
            except Exception:
                from deepafx_st.processors.system import System as _System  # type: ignore
            ckpt = os.environ.get("AIJOCKEY_DEEPAFX_CKPT")
            if not ckpt or not Path(ckpt).exists():
                print(f"[deepafx_st] missing checkpoint: {ckpt}")
                _LOAD_FAILED = True
                return None
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            sys_model = _System.load_from_checkpoint(ckpt, map_location=device)
            sys_model.eval()
            _PIPE = {"system": sys_model, "device": device}
            print(f"[deepafx_st] loaded {ckpt} on {device}")
            return _PIPE
        except Exception as e:
            print(f"[deepafx_st] load failed: {e}")
            _LOAD_FAILED = True
            return None


def master_styled(target_path: str | Path,
                    output_path: str | Path,
                    reference_path: str | Path | None = None) -> str | None:
    """Apply DeepAFx-ST: predict EQ+comp params from reference, run on target.

    Returns output_path on success, None on failure / disabled."""
    if not enabled():
        return None
    pipe = _load()
    if pipe is None:
        return None
    ref = reference_path or os.environ.get("AIJOCKEY_DEEPAFX_REF")
    if not ref or not Path(ref).exists():
        print(f"[deepafx_st] no reference: {ref}")
        return None
    try:
        import torch
        import torchaudio
        target_w, sr_t = torchaudio.load(str(target_path))
        ref_w, sr_r = torchaudio.load(str(ref))
        if sr_r != sr_t:
            ref_w = torchaudio.functional.resample(ref_w, sr_r, sr_t)
        if pipe["device"] == "cuda":
            target_w = target_w.cuda()
            ref_w = ref_w.cuda()
        with torch.inference_mode():
            out, _ = pipe["system"](target_w.unsqueeze(0), ref_w.unsqueeze(0))
        out = out.squeeze(0).clamp(-1, 1).cpu()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), out, sr_t)
        return str(output_path) if Path(output_path).exists() else None
    except Exception as e:
        print(f"[deepafx_st] master failed: {e}")
        return None
