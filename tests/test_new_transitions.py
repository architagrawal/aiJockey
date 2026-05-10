"""Smoke tests for the 17 new transition primitives + vocal-aware mapping.

torch is needed by transitions.py module-level import; if absent, we
skip the audio-level smoke tests but still run the catalog/mapping tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Catalog + mapping (no torch required)
# ---------------------------------------------------------------------------

from transition_catalog import (
    list_techniques,
    vocal_safe_techniques,
    vocal_unsafe_names,
    technique_for_context,
    reload as catalog_reload,
)


def setup_function(_):
    catalog_reload()


def test_catalog_has_17_more_implemented_post_upgrade():
    impl = list_techniques(status="implemented")
    names = {t["name"] for t in impl}
    new_ones = {"bass_swap", "highs_swap", "highpass_sweep_in", "punch_in",
                "chop", "loop_roll", "beat_juggle", "acapella_drop",
                "instrumental_swap", "kickless_swap", "drum_replace",
                "reverb_wash", "forward_spin", "tape_stop", "spectral_hold",
                "bpm_warp", "harmonic_overlay"}
    missing = new_ones - names
    assert not missing, f"catalog says these aren't implemented: {missing}"


def test_vocal_safe_partition_disjoint():
    safe = {t["name"] for t in vocal_safe_techniques(status=None)}
    unsafe = vocal_unsafe_names()
    assert not (safe & unsafe), f"name in BOTH safe and unsafe: {safe & unsafe}"


def test_vocal_unsafe_includes_aggressive():
    unsafe = vocal_unsafe_names()
    aggressive = {"chop", "tape_stop", "spinback", "forward_spin",
                  "build_riser_drop", "drum_break", "kickless_swap",
                  "drum_replace", "scratch_fill"}
    missing = aggressive - unsafe
    assert not missing, f"aggressive techniques NOT in unsafe set: {missing}"


def test_vocal_safe_includes_smooth():
    safe = {t["name"] for t in vocal_safe_techniques(status=None)}
    smooth = {"crossfade", "eq_swap", "bass_swap", "highs_swap",
              "filter_fade", "echo_out", "reverb_wash", "stem_swap",
              "mashup", "instrumental_swap", "harmonic_overlay"}
    missing = smooth - safe
    assert not missing, f"smooth techniques NOT in safe set: {missing}"


def test_technique_for_context_vocal_active_excludes_aggressive():
    """When vocal_active=True, aggressive techniques must not appear."""
    out = technique_for_context(tier="loop", vocal_active=True, status=None)
    names = {t["name"] for t in out}
    forbidden = {"chop", "loop_roll", "loop_tighten", "beat_juggle"}
    bad = forbidden & names
    assert not bad, f"vocal-active loop tier returned forbidden: {bad}"


def test_technique_for_context_instrumental_allows_aggressive():
    """When vocal_active=False, aggressive techniques are allowed."""
    out = technique_for_context(tier="loop", vocal_active=False, status="implemented")
    names = {t["name"] for t in out}
    # Should include at least one aggressive loop technique
    assert names & {"loop_tighten", "loop_roll", "beat_juggle", "chop"}


def test_technique_for_context_section_filter():
    """Drop-tier techniques should be filtered out for intro section."""
    out = technique_for_context(tier="drop", section_label="intro",
                                  vocal_active=False, status="implemented")
    names = {t["name"] for t in out}
    # silence_drop / build_riser_drop both have intro in incompatible_with
    forbidden = {"silence_drop", "build_riser_drop", "acapella_drop"}
    bad = forbidden & names
    assert not bad, f"drop-tier on intro returned forbidden: {bad}"


def test_technique_for_context_returns_best_for_first():
    """Techniques where section is in best_for should rank above neutrals."""
    out = technique_for_context(tier="major", section_label="break",
                                  vocal_active=False, status="implemented")
    if not out:
        pytest.skip("no major techniques for break")
    # First technique should have 'break' in its best_for
    first_best_for = [s.lower() for s in (out[0].get("best_for") or [])]
    assert "break" in first_best_for, \
        f"first technique {out[0]['name']} doesn't list break in best_for"


# ---------------------------------------------------------------------------
# Mapping wire-in
# ---------------------------------------------------------------------------

def test_tier_to_technique_vocal_active_returns_safe_only():
    pytest.importorskip("torch", reason="transition_mapping pulls execute via test paths")
    from transition_mapping import tier_to_technique
    # Cycle through several junction indices on loop tier with vocal_active=True
    aggressive = {"chop", "tape_stop", "drum_replace", "kickless_swap",
                  "spinback", "forward_spin", "build_riser_drop",
                  "snare_buildup", "scratch_fill", "loop_tighten",
                  "loop_roll", "beat_juggle", "pitch_bend", "bpm_warp",
                  "spectral_hold"}
    for j in range(6):
        tech = tier_to_technique("loop", j, vocal_active=True)
        assert tech["name"] not in aggressive, \
            f"vocal_active=True returned aggressive {tech['name']} at junction {j}"


def test_tier_to_technique_section_filter_no_drop_on_intro():
    pytest.importorskip("torch")
    from transition_mapping import tier_to_technique
    # Drop tier on intro section should never pick build_riser_drop / silence_drop
    for j in range(6):
        tech = tier_to_technique("drop", j, vocal_active=False, section_label="intro")
        assert tech["name"] not in {"silence_drop", "build_riser_drop", "acapella_drop"}, \
            f"drop tier on intro returned {tech['name']}"


def test_tier_to_technique_legacy_fallback_has_tier():
    """Even when catalog path fails, legacy returns techniques with tier field."""
    from transition_mapping import tier_to_technique
    tech = tier_to_technique("minor", 0)
    assert tech.get("tier") == "minor"
    assert "name" in tech


# ---------------------------------------------------------------------------
# Audio-level smoke (only when torch available)
# ---------------------------------------------------------------------------

@pytest.fixture
def audio_setup():
    pytest.importorskip("torch")
    import numpy as np
    SR = 44100
    beat = 60.0 / 120.0
    out = np.random.randn(2, SR * 8).astype(np.float32) * 0.3
    inn = np.random.randn(2, SR * 8).astype(np.float32) * 0.3
    drums = np.random.randn(2, SR * 8).astype(np.float32) * 0.2
    vox = np.random.randn(2, SR * 8).astype(np.float32) * 0.2
    return SR, beat, out, inn, drums, vox


def test_smoke_all_new_transitions(audio_setup):
    """Each new transition returns shape (2, T>0) on standard inputs."""
    SR, beat, out, inn, drums, vox = audio_setup
    import transitions as T

    cases = [
        ("bass_swap", T.bass_swap_transition(out, inn, SR, 4, beat)),
        ("highs_swap", T.highs_swap_transition(out, inn, SR, 4, beat)),
        ("highpass_sweep_in", T.highpass_sweep_in_transition(out, inn, SR, 4, beat)),
        ("punch_in", T.punch_in_transition(out, inn, SR)),
        ("chop", T.chop_transition(out, inn, SR, beat, n_chops=4)),
        ("loop_roll", T.loop_roll_transition(out, inn, SR, beat, steps=4)),
        ("beat_juggle", T.beat_juggle_transition(out, inn, SR, beat, n_juggles=2)),
        ("acapella_drop", T.acapella_drop_transition(out, vox, inn, SR,
                                                       vocal_only_bars=2,
                                                       beat_dur=beat)),
        ("instrumental_swap", T.instrumental_swap_transition(out, vox, inn,
                                                              SR, 4, beat)),
        ("kickless_swap", T.kickless_swap_transition(out, drums, inn, drums,
                                                       SR, 4, beat)),
        ("drum_replace", T.drum_replace_transition(out, drums, inn, drums,
                                                     SR, 4, beat)),
        ("reverb_wash", T.reverb_wash_transition(out, inn, SR, 4, beat)),
        ("forward_spin", T.forward_spin_transition(out, inn, SR, beat)),
        ("tape_stop", T.tape_stop_transition(out, inn, SR, beat)),
        ("spectral_hold", T.spectral_hold_transition(out, inn, SR, 2, beat)),
        ("bpm_warp", T.bpm_warp_transition(out, inn, SR, 4, beat,
                                            out_bpm=120, in_bpm=130)),
        ("harmonic_overlay", T.harmonic_overlay_transition(out, inn, SR, 4, beat)),
    ]
    for name, result in cases:
        assert result.ndim == 2, f"{name}: wrong ndim {result.ndim}"
        assert result.shape[0] == 2, f"{name}: not stereo {result.shape}"
        assert result.shape[1] > 0, f"{name}: empty output"
        assert result.dtype.kind == "f", f"{name}: non-float dtype {result.dtype}"
