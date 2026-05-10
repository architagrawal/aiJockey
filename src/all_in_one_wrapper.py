"""All-In-One Music Structure Analyzer wrapper.

Repo: mir-aidj/all-in-one (Bytedance, MIT licensed)
Paper: "All-In-One Metrical And Functional Structure Analysis With
Neighborhood Attentions on Demixed Audio" (arXiv 2307.16425)

Joint output from a SINGLE transformer pass:
    - tempo (BPM)
    - beats (timestamps)
    - downbeats (timestamps)
    - segments [{start, end, label}] where label ∈
        {intro, outro, break, bridge, inst, solo, verse, chorus, start, end}
    - per-stem embeddings (drums/bass/other/vocals) per time step

This replaces THREE current modules in one shot:
    - Beat-This! (beats + downbeats only)
    - librosa MFCC + agglomerative segmentation (sections without semantic labels)
    - madmom DBN (deprecated Py3.12 path)

Key value vs current pipeline:
    - Director's Phase 1 "drop only on drop-section" rule becomes
      EMPIRICALLY GROUNDED — `chorus` and `solo` labels available directly
      instead of energy-heuristic guessing.
    - Single GPU forward pass per clip → cheaper than running
      Beat-This! + librosa segmentation separately.

Install:
    pip install allin1
    # First call downloads ~500MB checkpoint to ~/.cache/all-in-one/

Env:
    AIJOCKEY_ALL_IN_ONE      0|1   default 0 (opt-in until validated at scale)
    AIJOCKEY_AIO_DEVICE      str   default 'cuda' if available
    AIJOCKEY_AIO_CHECKPOINT  hf-id default published checkpoint

Lazy load: model only imports on first call so startup stays cheap.

Falls through to caller's existing beat/segment code on any failure —
never breaks the pipeline.
"""
from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np


_LOCK = threading.Lock()
_LOAD_FAILED = False
_ANALYZER_CACHE: dict[str, Any] = {}   # cache analyzer instances per device


def enabled() -> bool:
    """True if env opts in AND import is available."""
    if os.environ.get("AIJOCKEY_ALL_IN_ONE", "0") != "1":
        return False
    if _LOAD_FAILED:
        return False
    try:
        import allin1   # noqa: F401
        return True
    except Exception:
        return False


def _load(device: str = "cuda"):
    """Idempotent load. Returns analyzer or None on failure."""
    global _LOAD_FAILED
    if _LOAD_FAILED:
        return None
    if device in _ANALYZER_CACHE:
        return _ANALYZER_CACHE[device]
    with _LOCK:
        if device in _ANALYZER_CACHE:
            return _ANALYZER_CACHE[device]
        if _LOAD_FAILED:
            return None
        try:
            import allin1
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            # allin1.analyze() doesn't expose model loading separately; the
            # first call loads the checkpoint and caches internally. We
            # store a sentinel here so callers don't re-import.
            _ANALYZER_CACHE[device] = {"module": allin1, "device": device}
            print(f"[all_in_one] ready (device={device})")
            return _ANALYZER_CACHE[device]
        except Exception as e:
            print(f"[all_in_one] load failed ({e.__class__.__name__}: {e})")
            _LOAD_FAILED = True
            return None


def analyze_audio_path(audio_path: str | os.PathLike,
                       device: str = "cuda",
                       include_embeddings: bool = False) -> dict | None:
    """Run All-In-One on a file path. Returns dict or None on failure.

    Output schema (normalized to match our existing analyze.py contract):
        {
          "tempo": float,
          "beats": [float, ...],          # seconds
          "downbeats": [float, ...],      # seconds
          "sections": [
            {"start": float, "end": float, "label": str, "energy": float},
            ...
          ],
          "embeddings": np.ndarray | None,   # (T, 4, D) per-stem if requested
          "source": "all_in_one"
        }

    `energy` is filled with a constant 0.5 by default (callers can replace
    with their own RMS-based estimate); kept in the schema so existing
    section-aware code in execute.py works unchanged.

    On failure: returns None. Caller falls back to its existing path.
    """
    state = _load(device=device)
    if state is None:
        return None

    try:
        allin1 = state["module"]
        # allin1.analyze accepts list[str] or single path; returns list of
        # AnalysisResult dataclasses or a single one depending on input.
        result = allin1.analyze(
            str(audio_path),
            out_dir=None,    # don't dump JSON to disk
            visualize=False,
            sonify=False,
            include_embeddings=include_embeddings,
            device=state["device"],
        )
    except Exception as e:
        print(f"[all_in_one] analyze failed for {audio_path}: {e}")
        return None

    if isinstance(result, list):
        if not result:
            return None
        result = result[0]

    try:
        # AnalysisResult fields: bpm, beats (list[float]), downbeats (list[float]),
        # segments (list of {start, end, label}). Normalize.
        sections = []
        for seg in getattr(result, "segments", []) or []:
            if isinstance(seg, dict):
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                label = str(seg.get("label", "?"))
            else:
                start = float(getattr(seg, "start", 0.0))
                end = float(getattr(seg, "end", 0.0))
                label = str(getattr(seg, "label", "?"))
            sections.append({
                "start": start,
                "end": end,
                "label": label,
                "energy": _label_to_energy(label),
                "type": label,    # back-compat with older section schema
            })

        out = {
            "tempo": float(getattr(result, "bpm", 0.0) or 0.0),
            "beats": [float(t) for t in (getattr(result, "beats", []) or [])],
            "downbeats": [float(t) for t in (getattr(result, "downbeats", []) or [])],
            "sections": sections,
            "embeddings": getattr(result, "embeddings", None) if include_embeddings else None,
            "source": "all_in_one",
        }
        return out
    except Exception as e:
        print(f"[all_in_one] result normalization failed: {e}")
        return None


# Map All-In-One's section labels to a rough energy proxy in [0, 1].
# Used to backfill the `energy` field in our existing section schema so
# downstream code (sections[i].get('energy')) keeps working.
#
# Numeric values are deliberate: they preserve the relative ordering used
# by execute.py's energy-arc planning without introducing new tunables.
_LABEL_ENERGY = {
    "intro":   0.30,
    "outro":   0.25,
    "break":   0.35,
    "bridge":  0.50,
    "inst":    0.55,
    "verse":   0.60,
    "solo":    0.75,
    "chorus":  0.85,
    "drop":    0.95,    # not in canonical list but tolerated
    "start":   0.20,
    "end":     0.15,
}


def _label_to_energy(label: str) -> float:
    return _LABEL_ENERGY.get(str(label).lower(), 0.5)


# ---------------------------------------------------------------------------
# Compatibility helpers — drop-in replacements for existing analyze.py paths
# ---------------------------------------------------------------------------

def beats_and_downbeats(audio_path: str | os.PathLike,
                         device: str = "cuda") -> tuple[float, list[float], list[float]] | None:
    """Drop-in replacement for analyze.Analyzer.beats_and_downbeats() that
    uses All-In-One. Returns (tempo, beats, downbeats) or None on failure.

    Caller pattern:
        out = all_in_one_wrapper.beats_and_downbeats(path)
        if out is not None:
            tempo, beats, downbeats = out
        else:
            # fall through to madmom / Beat-This! / librosa
    """
    r = analyze_audio_path(audio_path, device=device)
    if r is None:
        return None
    return r["tempo"], r["beats"], r["downbeats"]


def sections_for_clip(audio_path: str | os.PathLike,
                      device: str = "cuda") -> list[dict] | None:
    """Drop-in replacement for analyze.Analyzer.sections() that uses
    All-In-One labeled segments. Returns list of section dicts or None.
    """
    r = analyze_audio_path(audio_path, device=device)
    if r is None:
        return None
    return r["sections"]


__all__ = [
    "enabled",
    "analyze_audio_path",
    "beats_and_downbeats",
    "sections_for_clip",
]
