"""Tests for director JSON parsing + sanitization. No HF model loaded."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Force fallback path: no LLM call
os.environ["AIJOCKEY_USE_DIRECTOR_LLM"] = "0"

from director import (
    ALLOWED_ARCS,
    ALLOWED_TRANSITION_TIERS,
    _extract_json_object,
    _sanitize_out,
    estimate_max_transitions_for_pool,
    run_director,
)


def test_extract_json_plain():
    assert _extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_wrapped():
    assert _extract_json_object('garbage {"a": 1} trailing') == {"a": 1}


def test_extract_json_invalid():
    assert _extract_json_object('not json at all') is None


def test_sanitize_invalid_arc_falls_back():
    out = _sanitize_out({"arc": "nonsense"}, "build", "user prompt", 4, None)
    assert out["arc"] in ALLOWED_ARCS
    assert out["arc"] == "build"


def test_sanitize_tier_padding():
    out = _sanitize_out({"transition_tiers": ["minor"]}, "build", "p", 5, None)
    assert len(out["transition_tiers"]) == 5
    assert all(t in ALLOWED_TRANSITION_TIERS for t in out["transition_tiers"])


def test_sanitize_tier_truncation():
    out = _sanitize_out(
        {"transition_tiers": ["major"] * 20}, "build", "p", 4, None
    )
    assert len(out["transition_tiers"]) == 4


def test_sanitize_invalid_tier_coerced():
    out = _sanitize_out(
        {"transition_tiers": ["weird", "MAJOR", "minor"]}, "build", "p", 3, None
    )
    assert out["transition_tiers"][0] == "minor"  # invalid -> minor
    assert out["transition_tiers"][1] == "major"  # uppercase ok
    assert out["transition_tiers"][2] == "minor"


def test_sanitize_clamps_budgets():
    out = _sanitize_out(
        {"surprise_budget": 999, "callback_budget": -3}, "build", "p", 2, None
    )
    assert 0 <= out["surprise_budget"] <= 50
    assert 0 <= out["callback_budget"] <= 5


def test_sanitize_coherence_forces_tight_mix():
    out = _sanitize_out({}, "build", "p", 2, coherence_hint=0.85)
    assert out["same_genre_tight_mix"] is True


def test_sanitize_low_coherence_default_false():
    out = _sanitize_out({}, "build", "p", 2, coherence_hint=0.30)
    assert out["same_genre_tight_mix"] is False


def test_sanitize_accent_hints_filtered():
    out = _sanitize_out(
        {
            "accent_hints": [
                {"junction_index": 1, "fx_category": "risers", "beats": 3.0},
                "garbage",
                {"junction_index": "bad", "fx_category": "x"},
            ]
        },
        "build", "p", 2, None,
    )
    assert len(out["accent_hints"]) == 1
    assert out["accent_hints"][0]["fx_category"] == "risers"


def test_run_director_fallback_shape():
    # AIJOCKEY_USE_DIRECTOR_LLM=0 -> deterministic fallback path
    out = run_director("happy peak set", arc_preset="peak",
                       clip_count_estimate=5,
                       max_transitions_hint=4)
    assert out["arc"] == "peak"
    assert len(out["transition_tiers"]) == 4
    assert all(t in ALLOWED_TRANSITION_TIERS for t in out["transition_tiers"])
    assert isinstance(out["text_prompt"], str) and out["text_prompt"]


def test_estimate_max_transitions_bounds():
    assert 8 <= estimate_max_transitions_for_pool(0, 600.0) <= 64
    assert 8 <= estimate_max_transitions_for_pool(15, 1800.0) <= 64
