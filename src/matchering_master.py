"""Matchering 2.0 reference-track mastering wrapper.

MIT-licensed Python tool that matches a target audio's EQ + dynamics +
RMS to a reference recording. Drop-in alternative to our rule-based
multi-band + tape-sat master when a strong genre reference exists.

Env knobs:
    AIJOCKEY_MATCHERING_ENABLE   0|1  default 0
    AIJOCKEY_MATCHERING_REF      path to reference audio
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_AVAILABLE = None


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_MATCHERING_ENABLE", "0") == "1"


def _check() -> bool:
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    with _LOCK:
        if _AVAILABLE is not None:
            return _AVAILABLE
        try:
            import matchering  # type: ignore  # noqa: F401
            _AVAILABLE = True
        except Exception:
            _AVAILABLE = False
        return _AVAILABLE


def master_with_reference(target_path: str | Path,
                            output_path: str | Path,
                            reference_path: str | Path | None = None,
                            target_lufs: float = -9.0) -> str | None:
    """Run Matchering target → reference EQ + dynamics match. Returns
    output path on success, None on failure / disabled / missing pkg."""
    if not enabled() or not _check():
        return None
    ref = reference_path or os.environ.get("AIJOCKEY_MATCHERING_REF")
    if not ref or not Path(ref).exists():
        print(f"[matchering] no reference: {ref}")
        return None
    try:
        import matchering as mg  # type: ignore
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mg.process(
            target=str(target_path),
            reference=str(ref),
            results=[
                mg.pcm24(str(output_path)),
            ],
        )
        return str(output_path)
    except Exception as e:
        print(f"[matchering] process failed: {e}")
        return None
