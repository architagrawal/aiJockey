"""Tempo octave normalization.

Beat trackers (Beat-This!, librosa, madmom) frequently lock onto the
half-time or double-time interpretation of a track's tempo when the music
admits both as valid pulse candidates. Trap at "63 BPM" is canonically
127; jersey club at "260 BPM" is canonically 130; broken-beat / drill at
"75" is canonically 150. Music-IR ambiguity, not a bug in the tracker.

Downstream consequences when ignored:
  - Planner picks BPM-band buckets wrong (62 ≠ 130 cluster)
  - stretch_and_pitch attempts 2× / 0.5× speed → audible artifact
  - same_genre_tight_mix keying fails — half-time tempo doesn't match siblings

This module:
  1. Normalizes tempo to a canonical band when it's a known octave error
  2. Provides a "treat as no-op" guard for stretch ratios near 0.5 / 2.0

No upstream beat tracker change. Lives at the planner / execute boundary.
"""
from __future__ import annotations


# Canonical electronic-music BPM band. Most tracks land here. Half-time
# trackers report 60-90; double-time report 240-360.
CANONICAL_LO = 90.0
CANONICAL_HI = 180.0

# Genres whose canonical tempo can sit OUTSIDE 90-180 by design.
# Drum-and-bass canonically 165-185, jersey club 130-150 sometimes reported
# as half-time, footwork 150-160, ballads 60-90 (NOT halftime).
_GENRES_HIGH_TEMPO = {'drum_and_bass', 'dnb', 'hardcore', 'speedcore',
                       'happy_hardcore', 'jungle', 'footwork',
                       'gabber', 'breakcore'}
_GENRES_LOW_TEMPO = {'ambient', 'lofi', 'lofi_hip_hop', 'chillout',
                      'downtempo', 'classical', 'neo_classical',
                      'ballad', 'ghazal'}


def normalize_tempo(bpm: float, genre: str | None = None,
                    canonical_lo: float = CANONICAL_LO,
                    canonical_hi: float = CANONICAL_HI) -> float:
    """Map a tempo into the canonical band by doubling / halving as needed.

    Genre overrides: dnb is allowed > canonical_hi without halving;
    ambient is allowed < canonical_lo without doubling.

    Returns the normalized BPM. If input already in band, returned unchanged.
    """
    if bpm <= 0:
        return bpm
    g = (genre or '').lower().strip()
    if g in _GENRES_HIGH_TEMPO:
        # DnB canonically 165-185; allow up to 2× canonical_hi.
        canonical_hi = canonical_hi * 2.0
    if g in _GENRES_LOW_TEMPO:
        # Ambient / ballads: allow down to half canonical_lo without doubling.
        canonical_lo = canonical_lo / 2.0

    out = float(bpm)
    # Keep doubling until in band (handles 30 → 60 → 120).
    safety = 8
    while out < canonical_lo and safety > 0:
        out *= 2.0
        safety -= 1
    # Keep halving until in band (handles 480 → 240 → 120).
    safety = 8
    while out > canonical_hi and safety > 0:
        out /= 2.0
        safety -= 1
    return out


def is_octave_equivalent(src_bpm: float, dst_bpm: float,
                          tolerance: float = 0.05) -> bool:
    """True when (dst/src) is within `tolerance` of 0.5, 1.0, or 2.0.

    Lets callers skip stretching that would introduce 2× / 0.5× artifacts
    when src and dst are octave-equivalent — they're already aligned at the
    perceptual pulse level.
    """
    if src_bpm <= 0 or dst_bpm <= 0:
        return False
    ratio = dst_bpm / src_bpm
    for target in (0.5, 1.0, 2.0):
        if abs(ratio - target) / target <= tolerance:
            return True
    return False


def safe_stretch_ratio(src_bpm: float, dst_bpm: float,
                        max_ratio: float | None = None) -> float:
    """Return a stretch ratio that won't double-speed or half-speed audio.

    If src and dst are octave-equivalent (within tolerance), returns 1.0
    (no-op stretch) — perceptually they're already aligned.
    Otherwise returns dst_bpm / src_bpm clamped to [1/max_ratio, max_ratio]
    when max_ratio supplied.
    """
    if src_bpm <= 0 or dst_bpm <= 0:
        return 1.0
    if is_octave_equivalent(src_bpm, dst_bpm):
        return 1.0
    ratio = dst_bpm / src_bpm
    if max_ratio and max_ratio > 1.0:
        if ratio > max_ratio:
            ratio = max_ratio
        elif ratio < (1.0 / max_ratio):
            ratio = 1.0 / max_ratio
    return ratio


__all__ = ['normalize_tempo', 'is_octave_equivalent', 'safe_stretch_ratio',
            'CANONICAL_LO', 'CANONICAL_HI']
