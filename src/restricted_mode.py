"""
Demo-safe restricted mode.

Locks scope to 3 genres + 12 reliable techniques. Drops the 3 most
artifact-prone techniques for stable demo output. Adds phrase enforcement
+ BPM filtering at planner level.

Toggled via PlannerConfig.restricted = True or --restricted CLI flag.
"""
from __future__ import annotations

# Demo-safe technique whitelist. Per docs/dj_research.md §6+§10, all
# frequency-band swaps, filter sweeps, stem swaps, echoes, and overlays
# are safe even over vocals. Excluded: time/pitch warps + per-sample
# manipulation (pitch_bend, scratch_fill, spinback, beat_juggle, chop,
# loop_roll, spectral_hold, tape_stop, bpm_warp, forward_spin) — these
# need rubberband tuning + stem alignment we haven't shipped yet.
DEMO_SAFE_TECHNIQUES = [
    # Crossfade variants
    'crossfade', 'short_crossfade', 'long_crossfade',
    # Frequency-band swaps (vocal-safe)
    'eq_swap', 'bass_swap', 'highs_swap', 'frequency_blend',
    # Filter sweeps
    'filter_fade', 'highpass_sweep_in', 'band_filter_sweep',
    # Cuts + drops
    'cut', 'punch_in', 'silence_drop', 'fade_in',
    # Drum manipulations (vocal-compatible)
    'drum_break', 'kickless_swap', 'drum_replace',
    # Stem-aware (vocal-compatible by design)
    'stem_swap', 'instrumental_swap', 'mashup', 'acapella_drop',
    # Echo / reverb / harmonic overlays
    'echo_out', 'reverb_wash', 'harmonic_overlay',
    'riser_overlay', 'impact_overlay',
    # Loops (vocal-safe variants)
    'loop_tighten', 'loop_callback',
    # Bridge / build
    'snare_buildup', 'build_riser_drop',
]

# Multi-genre restricted scope
RESTRICTED_GENRES = ['edm', 'house', 'techno', 'progressive', 'trance', 'electronic']
RESTRICTED_BPM_MIN = 115
RESTRICTED_BPM_MAX = 135
RESTRICTED_BPM_TOLERANCE_PCT = 0.05  # ±5% stretch max
RESTRICTED_MAX_KEY_DIST = 3           # Camelot wheel distance


def is_clip_in_scope(clip: dict) -> bool:
    """Check if clip fits restricted scope."""
    tempo = clip.get('tempo', 0)
    if not (RESTRICTED_BPM_MIN <= tempo <= RESTRICTED_BPM_MAX):
        return False
    return True


def is_pair_compatible(clip_a: dict, clip_b: dict) -> bool:
    """Check tempo + key compat for a candidate transition."""
    from camelot import camelot_distance
    ta = clip_a.get('tempo', 0)
    tb = clip_b.get('tempo', 0)
    if ta <= 0 or tb <= 0:
        return False
    if abs(ta - tb) / max(ta, tb) > RESTRICTED_BPM_TOLERANCE_PCT:
        return False
    if camelot_distance(clip_a.get('key', '?'), clip_b.get('key', '?')) > RESTRICTED_MAX_KEY_DIST:
        return False
    return True


def filter_technique(tech_dict: dict, restricted: bool = True) -> dict:
    """If restricted, replace blocked techniques with safe fallback."""
    if not restricted:
        return tech_dict
    name = tech_dict.get('name', 'crossfade')
    if name in DEMO_SAFE_TECHNIQUES:
        return tech_dict
    # Map blocked to safe equivalent
    fallback_map = {
        'pitch_bend': 'crossfade',
        'scratch_fill': 'cut',
        'spinback': 'echo_out',
    }
    return {**tech_dict, 'name': fallback_map.get(name, 'crossfade')}


def snap_to_phrase_boundary(clip: dict, target_sec: float,
                            phrase_bars: int = 16) -> float:
    """Snap target time to nearest 16-bar phrase boundary using clip's downbeats."""
    from phrase import snap_to_phrase
    downbeats = clip.get('downbeats', [])
    return snap_to_phrase(target_sec, downbeats, bars_per_phrase=phrase_bars)
