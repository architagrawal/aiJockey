"""
Optional HF generative fill between segments (Phase B). Default OFF.

Env: AIJOCKEY_MUSICGEN=1 to enable (loads heavy model; not wired in execute by default).
"""

from __future__ import annotations

import os
from typing import Any


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_MUSICGEN", "").lower() in ("1", "true", "yes")


def maybe_extend_transition(
    tail_mono: Any,
    sr: int,
    bpm_hint: float,
) -> Any:
    """
    Return None (disabled) or stretched np audio for a short bridge.
    Not integrated in execute.py by default — stub for future flag.
    """
    if not enabled():
        return None
    return None
