"""TPS-based harmonic compatibility for non-Western mixing.

Camelot wheel assumes 12-TET diatonic relationships. Degrades on:
    - Indian classical (raga / shruti / 22-microtone systems)
    - Turkish makam (microtonal)
    - Hip-hop / jazz chord progressions where modal mixture matters

This module provides an alternative `harmonic_distance(key_a, key_b)`
that falls back to:
    - Lerdahl Tonal Pitch Space (TPS) distance for triadic keys
    - Shruti-cents proximity for Indian-classical-tagged keys
    - Camelot distance otherwise (existing default)

Genre is the gate: if genre slug ∈ INDIAN_TAGS, route to shruti.
Else if non-Western or jazz/hip-hop, use TPS. Else Camelot.
"""
from __future__ import annotations

import os
import re

INDIAN_TAGS = {"hindustani", "carnatic", "raga", "indian_classical",
                "punjabi", "bollywood", "ghazal"}
MAKAM_TAGS = {"makam", "turkish", "arabic_maqam"}
TPS_TAGS = {"jazz", "hip_hop", "trap", "rnb", "pop", "rock", "blues",
             "neo_soul"}


_CAMELOT_RE = re.compile(r"^([0-9]{1,2})([AB])$", re.IGNORECASE)
_LETTER_RE = re.compile(r"^([A-G][b#]?)\s*(maj|min|major|minor|m)?$",
                          re.IGNORECASE)


# Pitch class lookup
_NOTE_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
             "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
             "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}

_CAMELOT_TO_PC: dict[str, tuple[int, str]] = {
    # (PC, mode)
    "1A": (8, "minor"),  "1B": (11, "major"),
    "2A": (3, "minor"),  "2B": (6, "major"),
    "3A": (10, "minor"), "3B": (1, "major"),
    "4A": (5, "minor"),  "4B": (8, "major"),
    "5A": (0, "minor"),  "5B": (3, "major"),
    "6A": (7, "minor"),  "6B": (10, "major"),
    "7A": (2, "minor"),  "7B": (5, "major"),
    "8A": (9, "minor"),  "8B": (0, "major"),
    "9A": (4, "minor"),  "9B": (7, "major"),
    "10A": (11, "minor"), "10B": (2, "major"),
    "11A": (6, "minor"),  "11B": (9, "major"),
    "12A": (1, "minor"),  "12B": (4, "major"),
}


def _parse_key(k: str) -> tuple[int, str] | None:
    """Returns (pitch_class 0-11, mode 'major'/'minor') or None."""
    if not k:
        return None
    s = str(k).strip()
    m = _CAMELOT_RE.match(s)
    if m:
        return _CAMELOT_TO_PC.get(s.upper())
    m = _LETTER_RE.match(s)
    if m:
        note, mode = m.group(1).capitalize(), (m.group(2) or "major").lower()
        if note not in _NOTE_PC:
            return None
        mode = "minor" if mode in ("min", "m", "minor") else "major"
        return _NOTE_PC[note], mode
    return None


def _camelot_distance(a: str, b: str) -> int:
    """Simple Camelot wheel distance (0-12)."""
    am = _CAMELOT_RE.match((a or "").upper().strip())
    bm = _CAMELOT_RE.match((b or "").upper().strip())
    if not am or not bm:
        return 6
    an, am2 = int(am.group(1)), am.group(2)
    bn, bm2 = int(bm.group(1)), bm.group(2)
    num_dist = min(abs(an - bn), 12 - abs(an - bn))
    mode_dist = 0 if am2 == bm2 else 1
    return num_dist + mode_dist


def _tps_distance(a: str, b: str) -> float:
    """Lerdahl TPS-style chord distance for triadic keys.

    distance = |pitch_class_circle_of_fifths| + 2*|mode_change|
    Normalized to roughly [0, 6] to match Camelot scale.
    """
    pa, pb = _parse_key(a), _parse_key(b)
    if pa is None or pb is None:
        return float(_camelot_distance(a, b))
    pca, ma = pa
    pcb, mb = pb
    # Distance on circle of fifths (PC * 7 mod 12)
    cof_a = (pca * 7) % 12
    cof_b = (pcb * 7) % 12
    region_dist = min(abs(cof_a - cof_b), 12 - abs(cof_a - cof_b))
    mode_penalty = 0.0 if ma == mb else 1.5
    # Lerdahl basic-space: scale region by 1.0, modal swap by 1.5
    return float(region_dist + mode_penalty)


def _shruti_distance(a: str, b: str) -> float:
    """Approximate microtonal proximity for Indian classical keys.

    Without raga tags, we treat key strings as tonics. Distance = cents
    between tonics, normalized to Camelot-like 0-6 scale. If keys are
    identical, distance 0.
    """
    pa, pb = _parse_key(a), _parse_key(b)
    if pa is None or pb is None:
        return 3.0  # neutral fallback
    pca, _ = pa
    pcb, _ = pb
    pc_diff = min(abs(pca - pcb), 12 - abs(pca - pcb))
    # 12 semitones = 1200 cents. Map 0-6 semitone diff → 0-3 distance.
    return float(pc_diff * 0.5)


def harmonic_distance(key_a: str, key_b: str,
                       genre_a: str | None = None,
                       genre_b: str | None = None) -> float:
    """Return a [0, ~12] harmonic distance, lower = more compatible.

    Router picks measurement based on genre tags. Defaults to Camelot.
    Toggle off: AIJOCKEY_TPS_ROUTER=0 → always Camelot.
    """
    if os.environ.get("AIJOCKEY_TPS_ROUTER", "1") == "0":
        return float(_camelot_distance(key_a, key_b))
    ga = (genre_a or "").lower().strip()
    gb = (genre_b or "").lower().strip()
    if ga in INDIAN_TAGS or gb in INDIAN_TAGS:
        return _shruti_distance(key_a, key_b)
    if ga in MAKAM_TAGS or gb in MAKAM_TAGS:
        # Without makam tonic estimator, use shruti as a reasonable proxy
        return _shruti_distance(key_a, key_b)
    if ga in TPS_TAGS or gb in TPS_TAGS:
        return _tps_distance(key_a, key_b)
    return float(_camelot_distance(key_a, key_b))
