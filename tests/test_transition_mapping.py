"""Unit tests for LLM tier → technique mapping + Phase 1 vocab gating."""

import os
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


def test_drop_known():
    t = tier_to_technique("drop", 5)
    assert t["name"] in ALLOWLIST_NAMES


# ---------------------------------------------------------------------------
# Phase 1 vocab gating (director.py downgrades cut/loop -> major)
# ---------------------------------------------------------------------------

def _sanitize_with_phase(phase: str, raw: dict, max_t: int = 5) -> dict:
    os.environ["AIJOCKEY_PHASE"] = phase
    if "director" in sys.modules:
        del sys.modules["director"]
    import director
    return director._sanitize_out(raw, arc_fallback="build",
                                   user_prompt="test",
                                   max_transitions=max_t,
                                   coherence_hint=None)


def test_phase1_downgrades_cut_loop():
    raw = {
        "transition_tiers": ["cut", "loop", "drop", "minor", "major"],
        "arc": "build", "text_prompt": "test",
    }
    out = _sanitize_with_phase("1", raw)
    tiers = out["transition_tiers"]
    # cut + loop should be downgraded to major in Phase 1
    assert "cut" not in tiers
    assert "loop" not in tiers
    # drop / minor / major preserved
    assert "drop" in tiers
    assert "minor" in tiers
    assert "major" in tiers


def test_phase2_keeps_full_vocab():
    raw = {
        "transition_tiers": ["cut", "loop", "drop", "minor", "major"],
        "arc": "build", "text_prompt": "test",
    }
    out = _sanitize_with_phase("2", raw)
    tiers = out["transition_tiers"]
    for t in ("cut", "loop", "drop", "minor", "major"):
        assert t in tiers
    # restore default for subsequent tests
    os.environ["AIJOCKEY_PHASE"] = "1"
