"""Smoke tests for transition_catalog reader."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transition_catalog import (
    all_techniques,
    list_techniques,
    get,
    techniques_for_section,
    blocked_for_section,
    techniques_by_tier,
    categories,
    reload,
)


def setup_function(_):
    reload()


def test_catalog_loads():
    techs = all_techniques()
    assert len(techs) >= 30, f"catalog has only {len(techs)} entries"


def test_each_entry_has_required_fields():
    required = {"name", "category", "tier", "implementation_status",
                "typical_bars", "best_for", "incompatible_with",
                "description", "params"}
    for t in all_techniques():
        missing = required - set(t.keys())
        assert not missing, f"{t.get('name')} missing fields: {missing}"


def test_implemented_filter_returns_subset():
    impl = list_techniques(status="implemented")
    all_techs = all_techniques()
    assert 0 < len(impl) < len(all_techs)
    for t in impl:
        assert t["implementation_status"] == "implemented"


def test_tier_filter():
    drops = list_techniques(tier="drop", status=None)
    for t in drops:
        assert t["tier"] == "drop"
    assert len(drops) >= 3


def test_get_known_technique():
    cf = get("crossfade")
    assert cf is not None
    assert cf["category"] == "fade"
    assert cf["tier"] == "minor"


def test_get_unknown_returns_none():
    assert get("not_a_real_technique") is None


def test_techniques_for_drop_section():
    drop_compatible = techniques_for_section("drop", status="implemented")
    names = {t["name"] for t in drop_compatible}
    # silence_drop is implemented + best_for drop
    assert "silence_drop" in names
    # filter_fade has drop in incompatible_with → must NOT appear
    assert "filter_fade" not in names


def test_techniques_for_intro_section():
    intro = techniques_for_section("intro", status=None)
    names = {t["name"] for t in intro}
    # long_crossfade best_for intro
    assert "long_crossfade" in names


def test_blocked_for_drop_section_includes_filter_fade():
    blocked = blocked_for_section("drop")
    assert "filter_fade" in blocked


def test_blocked_for_intro_includes_aggressive():
    blocked = set(blocked_for_section("intro"))
    # intro is incompatible with drop / cut tier transitions
    assert "cut" in blocked or "spinback" in blocked or "build_riser_drop" in blocked


def test_techniques_by_tier_summary():
    summary = techniques_by_tier()
    assert "minor" in summary
    assert "major" in summary
    assert "drop" in summary
    assert "cut" in summary
    assert "loop" in summary
    assert len(summary["major"]) >= 5


def test_categories_summary():
    cats = categories()
    assert "fade" in cats
    assert "filter" in cats
    assert "stem" in cats


def test_reload_picks_up_catalog():
    """Reload doesn't crash even if called multiple times."""
    reload()
    a = all_techniques()
    reload()
    b = all_techniques()
    assert len(a) == len(b)


def test_implemented_techniques_match_known_set():
    """Implemented techniques should include the ones currently in transitions.py."""
    impl_names = {t["name"] for t in list_techniques(status="implemented")}
    known_real = {
        "crossfade", "eq_swap", "filter_fade", "silence_drop",
        "drum_break", "stem_swap", "mashup", "echo_out",
        "spinback", "pitch_bend", "loop_tighten", "loop_callback",
        "scratch_fill", "cut",
    }
    missing = known_real - impl_names
    assert not missing, f"catalog says these aren't implemented but they are: {missing}"
