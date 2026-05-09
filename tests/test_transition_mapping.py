"""Unit tests for LLM tier → technique mapping."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transition_mapping import ALLOWLIST_NAMES, tier_to_technique


def test_minor_known():
    t = tier_to_technique("minor", 0)
    assert t["name"] in ALLOWLIST_NAMES
    assert t["name"] in ("eq_swap", "crossfade")


def test_major_known():
    t = tier_to_technique("major", 9)
    assert t["name"] in ALLOWLIST_NAMES


def test_invalid_tier_defaults_minor():
    t = tier_to_technique("???", 0)
    assert t["name"] in ("eq_swap", "crossfade")
