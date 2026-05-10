"""Centralized efficiency hooks. Used by every training/inference path.

Reference: Phase A polish plan §16.2 / §16.3.

Toggles via env vars (defaults aim for max throughput on MI300X):
    AIJOCKEY_DTYPE        bfloat16|float16|float32   (default bfloat16)
    AIJOCKEY_COMPILE      0|1                        (default 1)
    AIJOCKEY_FLASH_ATTN   0|1|2                      (default 2)
    AIJOCKEY_QLORA        0|1                        (default 0; turn on for >7B)
    AIJOCKEY_OPTIMIZER    adamw|lion|sophia|adamw8bit  (default lion)
"""
from __future__ import annotations
import os
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Dtype + autocast
# ---------------------------------------------------------------------------

def get_dtype() -> torch.dtype:
    name = os.getenv('AIJOCKEY_DTYPE', 'bfloat16').lower()
    return {
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float16': torch.float16,
        'fp16': torch.float16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }.get(name, torch.bfloat16)


def autocast_ctx() -> Any:
    dtype = get_dtype()
    if dtype == torch.float32 or not torch.cuda.is_available():
        import contextlib
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type='cuda', dtype=dtype)


# ---------------------------------------------------------------------------
# Compile + Flash Attention
# ---------------------------------------------------------------------------

def maybe_compile(model: torch.nn.Module,
                  mode: str = 'reduce-overhead') -> torch.nn.Module:
    """Wrap with torch.compile if AIJOCKEY_COMPILE=1 and torch>=2."""
    if os.getenv('AIJOCKEY_COMPILE', '1') == '0':
        return model
    if not hasattr(torch, 'compile'):
        return model
    try:
        return torch.compile(model, mode=mode)
    except Exception as e:
        print(f"warn: torch.compile failed ({e}), running eager")
        return model


def hf_attn_implementation() -> str | None:
    """Value to pass as `attn_implementation=` to HF transformers `from_pretrained`.
    Returns None if env requests default.
    """
    v = os.getenv('AIJOCKEY_FLASH_ATTN', '2')
    if v == '0':
        return 'eager'
    if v == '1':
        return 'sdpa'
    return 'flash_attention_2'


# ---------------------------------------------------------------------------
# QLoRA / quantization for big-model fine-tune
# ---------------------------------------------------------------------------

def qlora_quant_config():
    """Return BitsAndBytesConfig for 4-bit QLoRA load. None if disabled."""
    if os.getenv('AIJOCKEY_QLORA', '0') == '0':
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=get_dtype(),
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )


def lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05,
                target_modules: list[str] | None = None):
    """Standard LoRA config for HF PEFT."""
    try:
        from peft import LoraConfig
    except ImportError:
        return None
    return LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules or ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias='none', task_type='CAUSAL_LM',
    )


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def make_optimizer(params, lr: float = 1e-4, weight_decay: float = 0.01):
    """Pick optimizer per AIJOCKEY_OPTIMIZER env. Lion default (faster than
    AdamW, half optimizer state memory)."""
    name = os.getenv('AIJOCKEY_OPTIMIZER', 'lion').lower()
    if name == 'lion':
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=lr * 0.3, weight_decay=weight_decay)
        except ImportError:
            pass
    if name == 'sophia':
        try:
            from sophia_optimizer import SophiaG
            return SophiaG(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            pass
    if name == 'adamw8bit':
        try:
            import bitsandbytes.optim as bnbopt
            return bnbopt.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# DPO/ORPO/KTO trainer factory
# ---------------------------------------------------------------------------

def preference_trainer(method: str = 'orpo'):
    """Return TRL trainer class for the chosen preference method.

    orpo: half VRAM vs DPO, no reference model
    dpo: classic
    kto: point-wise feedback (thumbs up/down per sample, no pairs)
    """
    try:
        from trl import DPOTrainer, ORPOTrainer, KTOTrainer
    except ImportError as e:
        raise ImportError(
            "trl required for preference training. pip install trl") from e
    return {
        'dpo': DPOTrainer,
        'orpo': ORPOTrainer,
        'kto': KTOTrainer,
    }[method.lower()]


# ---------------------------------------------------------------------------
# Inference quantization for Director
# ---------------------------------------------------------------------------

def int8_quant_config():
    """8-bit weight quant for inference (Director per-junction). Halves
    VRAM, ~2x speed on transformers attention."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        return None
    return BitsAndBytesConfig(load_in_8bit=True)


__all__ = [
    'get_dtype', 'autocast_ctx', 'maybe_compile', 'hf_attn_implementation',
    'qlora_quant_config', 'lora_config', 'make_optimizer',
    'preference_trainer', 'int8_quant_config',
]
