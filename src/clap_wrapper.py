"""
CLAP audio embedding wrapper.

Tries laion-clap first (preferred, simpler API). Falls back to HF transformers
ClapModel when laion-clap not available (e.g., Python 3.12 where the laion-clap
wheel build fails).

Both backends use the same underlying model: laion/clap-htsat-unfused.
Output: 512-dim embedding per audio chunk at 48 kHz.
"""
from __future__ import annotations
import numpy as np


_BACKEND = None  # 'laion' | 'transformers' | None
_MODEL = None
_PROCESSOR = None


def _try_laion():
    global _BACKEND, _MODEL
    try:
        from laion_clap import CLAP_Module
        m = CLAP_Module(enable_fusion=False)
        m.load_ckpt()
        _MODEL = m
        _BACKEND = 'laion'
        return True
    except Exception:
        return False


def _try_transformers():
    global _BACKEND, _MODEL, _PROCESSOR
    try:
        import torch
        from transformers import ClapModel, ClapProcessor
        ckpt = "laion/clap-htsat-unfused"
        _MODEL = ClapModel.from_pretrained(ckpt)
        _PROCESSOR = ClapProcessor.from_pretrained(ckpt)
        _MODEL.eval()
        if torch.cuda.is_available():
            _MODEL = _MODEL.cuda()
        _BACKEND = 'transformers'
        return True
    except Exception as e:
        print(f'transformers ClapModel load failed: {e}')
        return False


def load_clap() -> None:
    """Lazy-load CLAP. Idempotent."""
    if _MODEL is not None:
        return
    if _try_laion():
        print('CLAP backend: laion-clap')
        return
    if _try_transformers():
        print('CLAP backend: transformers')
        return
    raise RuntimeError('No CLAP backend available. Install laion-clap or transformers.')


def get_audio_embedding(audio_48k: np.ndarray) -> np.ndarray:
    """
    Compute CLAP embedding for audio at 48 kHz.
    Input: (T,) or (1, T) float32 numpy array, mono, 48000 Hz.
    Output: (1, 512) numpy array.
    """
    load_clap()
    if audio_48k.ndim == 2 and audio_48k.shape[0] == 1:
        audio_48k = audio_48k[0]
    audio_48k = audio_48k.astype(np.float32)

    if _BACKEND == 'laion':
        return _MODEL.get_audio_embedding_from_data(
            audio_48k[None, :], use_tensor=False
        ).astype(np.float32)

    # transformers backend — kwarg renamed `audios` -> `audio` in transformers 5.x
    import torch
    try:
        inputs = _PROCESSOR(
            audio=audio_48k, sampling_rate=48000, return_tensors='pt',
        )
    except (TypeError, ValueError):
        inputs = _PROCESSOR(
            audios=audio_48k, sampling_rate=48000, return_tensors='pt',
        )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = _MODEL.get_audio_features(**inputs)
    # transformers 4.x: tensor directly. 5.x: ModelOutput object — pick attr.
    emb = _extract_embedding(out)
    return emb.cpu().numpy().astype(np.float32)


def _extract_embedding(out):
    import torch
    if isinstance(out, torch.Tensor):
        return out
    # transformers ModelOutput — try known attrs in order
    for attr in ('audio_embeds', 'embeds', 'pooler_output', 'last_hidden_state'):
        v = getattr(out, attr, None)
        if v is not None and hasattr(v, 'shape'):
            # last_hidden_state needs pooling (mean over time)
            if attr == 'last_hidden_state' and v.dim() == 3:
                return v.mean(dim=1)
            return v
    raise RuntimeError(f'unable to extract embedding from {type(out)}, '
                       f'attrs: {[a for a in dir(out) if not a.startswith("_")]}')


class CLAP_Module:
    """
    Drop-in shim for laion_clap.CLAP_Module — same interface, uses whichever
    backend is available.
    """

    def __init__(self, enable_fusion: bool = False):
        self.enable_fusion = enable_fusion

    def load_ckpt(self, *args, **kwargs):
        load_clap()

    def get_audio_embedding_from_data(self, audio: np.ndarray,
                                      use_tensor: bool = False) -> np.ndarray:
        # Input shape (N, T) per laion-clap convention. We treat first row as audio.
        if audio.ndim == 2:
            audio = audio[0]
        return get_audio_embedding(audio)
