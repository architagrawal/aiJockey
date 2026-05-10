"""Smoke + correctness tests for src/candidate_picker.py."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from candidate_picker import (
    enabled,
    build_candidates,
    score_candidate,
    pick_best_junction,
    _type_fit_score,
    _label_energy,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in (
        "AIJOCKEY_CANDIDATE_PICKER",
        "AIJOCKEY_PICKER_W_ENERGY",
        "AIJOCKEY_PICKER_W_TYPE_FIT",
        "AIJOCKEY_PICKER_W_VOCAL",
        "AIJOCKEY_PICKER_W_KEY",
        "AIJOCKEY_PICKER_W_BPM",
        "AIJOCKEY_PICKER_W_DURATION",
        "AIJOCKEY_PICKER_MIN_SCORE",
    ):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# enabled()
# ---------------------------------------------------------------------------

def test_enabled_default_false():
    assert enabled() is False


def test_enabled_with_env(monkeypatch):
    monkeypatch.setenv("AIJOCKEY_CANDIDATE_PICKER", "1")
    assert enabled() is True


# ---------------------------------------------------------------------------
# Section-label energy defaults
# ---------------------------------------------------------------------------

def test_label_energy_chorus_higher_than_intro():
    assert _label_energy("chorus") > _label_energy("intro")
    assert _label_energy("drop") > _label_energy("verse")


def test_label_energy_unknown_fallback():
    assert _label_energy("UNKNOWN") == 0.5


# ---------------------------------------------------------------------------
# Type-fit lookup table
# ---------------------------------------------------------------------------

def test_type_fit_drop_loves_chorus():
    assert _type_fit_score("build_riser_drop", "chorus") > 0.8
    assert _type_fit_score("build_riser_drop", "intro") < 0.3


def test_type_fit_drum_break_loves_break():
    assert _type_fit_score("drum_break", "break") > 0.9


def test_type_fit_unknown_neutral():
    assert _type_fit_score("not_a_real_transition", "verse") == 0.5
    assert _type_fit_score("crossfade", "not_a_real_section") == 0.5


# ---------------------------------------------------------------------------
# build_candidates
# ---------------------------------------------------------------------------

def test_build_candidates_filters_short():
    sections = [
        {"start": 0, "end": 5, "label": "intro"},      # too short
        {"start": 5, "end": 35, "label": "verse"},     # ok
        {"start": 35, "end": 100, "label": "chorus"},  # gets truncated
    ]
    cands = build_candidates({}, sections, min_seconds=8.0, max_seconds=40.0)
    assert len(cands) == 2
    assert cands[0]["label"] == "verse"
    assert cands[1]["label"] == "chorus"
    # Long chorus truncated to start + max_seconds
    assert cands[1]["duration"] <= 40.0


def test_build_candidates_marks_vocals():
    sections = [
        {"start": 0, "end": 30, "label": "verse"},
        {"start": 30, "end": 60, "label": "inst"},
    ]
    cands = build_candidates({}, sections)
    by_label = {c["label"]: c for c in cands}
    assert by_label["verse"]["has_vocals"] is True
    assert by_label["inst"]["has_vocals"] is False


def test_build_candidates_uses_meta_sections_when_explicit_none():
    meta = {"sections": [{"start": 0, "end": 30, "label": "chorus"}]}
    cands = build_candidates(meta)
    assert len(cands) == 1
    assert cands[0]["label"] == "chorus"


def test_build_candidates_empty_when_no_sections():
    assert build_candidates({}, []) == []
    assert build_candidates({}) == []


# ---------------------------------------------------------------------------
# score_candidate
# ---------------------------------------------------------------------------

def test_score_perfect_energy_match_high_contrib():
    cand = {"start": 0, "end": 30, "label": "chorus", "energy": 0.85,
             "duration": 30, "has_vocals": True}
    score, br = score_candidate(
        cand, {"key": "8A", "tempo": 128.0},
        target_energy=0.85, target_bpm=128.0, target_key="8A",
        transition_type="crossfade",
    )
    # Energy match perfect → +1.0 * weight = +1.0
    # BPM match → +1.0 * 0.8 = +0.8
    # Key match → +1.0 * 1.2 = +1.2
    assert br["energy"] > 0.5
    assert br["bpm"] > 0.5
    assert br["key"] > 1.0
    assert score > 1.0


def test_score_energy_mismatch_negative_contrib():
    cand = {"start": 0, "end": 30, "label": "intro", "energy": 0.2,
             "duration": 30, "has_vocals": False}
    score, br = score_candidate(
        cand, {"key": "8A", "tempo": 128.0},
        target_energy=0.95,    # huge mismatch
        target_bpm=128.0, target_key="8A",
        transition_type="build_riser_drop",
    )
    assert br["energy"] < 0    # mismatch
    assert br["type_fit"] < 0  # intro is bad fit for drop tier


def test_score_drop_tier_picks_chorus_over_intro():
    chorus_cand = {"start": 0, "end": 30, "label": "chorus", "energy": 0.85,
                   "duration": 30, "has_vocals": True}
    intro_cand = {"start": 0, "end": 30, "label": "intro", "energy": 0.30,
                  "duration": 30, "has_vocals": False}
    prev = {"key": "8A", "tempo": 128.0}
    common_kw = dict(target_bpm=128.0, target_key="8A", target_energy=0.9,
                     transition_type="build_riser_drop")
    s_chorus, _ = score_candidate(chorus_cand, prev, **common_kw)
    s_intro, _ = score_candidate(intro_cand, prev, **common_kw)
    assert s_chorus > s_intro, f"chorus={s_chorus} should beat intro={s_intro}"


def test_score_vocal_segment_penalized():
    voc = {"start": 0, "end": 30, "label": "verse", "energy": 0.6,
            "duration": 30, "has_vocals": True}
    inst = {"start": 0, "end": 30, "label": "inst", "energy": 0.6,
             "duration": 30, "has_vocals": False}
    prev = {"key": "8A", "tempo": 128.0}
    s_voc, _ = score_candidate(voc, prev, target_energy=0.6,
                                transition_type="crossfade")
    s_inst, _ = score_candidate(inst, prev, target_energy=0.6,
                                transition_type="crossfade")
    # inst should score higher than verse (no vocals + similar type fit)
    assert s_inst > s_voc


def test_score_bpm_strain_penalized():
    cand = {"start": 0, "end": 30, "label": "verse", "energy": 0.6,
             "duration": 30, "has_vocals": True}
    prev = {"key": "8A", "tempo": 100.0}
    s_close, _ = score_candidate(cand, prev,
                                  target_bpm=102.0, transition_type="crossfade")
    s_far, _ = score_candidate({**cand}, prev,
                                target_bpm=140.0, transition_type="crossfade")
    assert s_close > s_far


# ---------------------------------------------------------------------------
# pick_best_junction
# ---------------------------------------------------------------------------

def test_pick_returns_none_when_no_candidates():
    assert pick_best_junction({"key": "8A"}, []) is None


def test_pick_returns_best_with_breakdown():
    cands = [
        {"start": 0, "end": 30, "label": "intro", "energy": 0.3,
         "duration": 30, "has_vocals": False},
        {"start": 30, "end": 60, "label": "chorus", "energy": 0.9,
         "duration": 30, "has_vocals": True},
        {"start": 60, "end": 90, "label": "drop", "energy": 0.95,
         "duration": 30, "has_vocals": False},
    ]
    best = pick_best_junction(
        {"key": "8A", "tempo": 128.0}, cands,
        target_bpm=128.0, target_key="8A", target_energy=0.95,
        transition_type="build_riser_drop",
    )
    assert best is not None
    assert best["label"] in ("drop", "chorus")    # both score well
    assert "score" in best
    assert "breakdown" in best
    assert best["rank"] == 1
    assert len(best["all_scores"]) == 3


def test_pick_respects_min_score(monkeypatch):
    """Set unattainable min_score → returns None."""
    cands = [
        {"start": 0, "end": 30, "label": "verse", "energy": 0.5,
         "duration": 30, "has_vocals": True},
    ]
    monkeypatch.setenv("AIJOCKEY_PICKER_MIN_SCORE", "10.0")
    out = pick_best_junction({"key": "8A"}, cands,
                              target_energy=0.5, transition_type="crossfade")
    assert out is None


def test_pick_all_scores_sorted_descending():
    cands = [
        {"start": 0,  "end": 30, "label": "intro",  "energy": 0.2,
         "duration": 30, "has_vocals": False},
        {"start": 30, "end": 60, "label": "chorus", "energy": 0.9,
         "duration": 30, "has_vocals": True},
        {"start": 60, "end": 90, "label": "verse",  "energy": 0.6,
         "duration": 30, "has_vocals": True},
    ]
    best = pick_best_junction(
        {"key": "8A", "tempo": 128.0}, cands,
        target_bpm=128.0, target_energy=0.85, transition_type="crossfade",
    )
    scores = [s["score"] for s in best["all_scores"]]
    assert scores == sorted(scores, reverse=True)


def test_pick_weight_override_changes_winner():
    """Boosting type_fit weight should push drop-tier to prefer chorus."""
    cands = [
        {"start": 0,  "end": 30, "label": "verse",  "energy": 0.85,
         "duration": 30, "has_vocals": True},      # ideal energy, bad type fit
        {"start": 30, "end": 60, "label": "chorus", "energy": 0.5,
         "duration": 30, "has_vocals": True},      # bad energy, ideal type fit
    ]
    prev = {"key": "8A", "tempo": 128.0}
    common_kw = dict(target_bpm=128.0, target_key="8A", target_energy=0.85,
                     transition_type="build_riser_drop")
    out_balanced = pick_best_junction(prev, cands, **common_kw)
    out_type_heavy = pick_best_junction(prev, cands,
                                         weights={"type_fit": 5.0},
                                         **common_kw)
    # Balanced may pick verse (energy match wins). Type-heavy should pick chorus.
    assert out_type_heavy["label"] == "chorus"
