"""Tests for constitutional rules validator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import constitutional as C


def _drop_violating_timeline():
    return [
        {"clip_id": "a", "target_bpm": 128.0, "target_key": "8A",
         "segment": {"type": "intro", "start": 0.0, "end": 32.0}},
        {"clip_id": "b", "target_bpm": 128.0, "target_key": "8A",
         "segment": {"type": "breakdown", "start": 0.0, "end": 32.0},
         "transition_in": {"tier": "drop", "name": "build_riser_drop", "bars": 8}},
    ]


def test_drop_section_rejected_on_breakdown():
    tl = _drop_violating_timeline()
    violations = C.check_drop_section(tl)
    assert any(v.rule == "drop_section" for v in violations)


def test_breakdown_pair_rejected():
    tl = [
        {"clip_id": "a", "target_bpm": 128.0,
         "segment": {"type": "breakdown", "start": 0.0, "end": 32.0}},
        {"clip_id": "b", "target_bpm": 128.0,
         "segment": {"type": "breakdown", "start": 0.0, "end": 32.0},
         "transition_in": {"tier": "minor", "name": "crossfade", "bars": 8}},
    ]
    violations = C.check_breakdown_pair(tl)
    assert violations and violations[0].rule == "breakdown_pair"


def test_bpm_drift_rejected():
    tl = [
        {"clip_id": "a", "target_bpm": 100.0,
         "segment": {"type": "verse", "start": 0.0, "end": 32.0}},
        {"clip_id": "b", "target_bpm": 130.0,
         "segment": {"type": "verse", "start": 0.0, "end": 32.0},
         "transition_in": {"tier": "minor", "name": "crossfade", "bars": 8}},
    ]
    violations = C.check_bpm_drift(tl)
    assert any(v.rule == "bpm_drift" for v in violations)


def test_repair_downgrades_drop():
    tl = _drop_violating_timeline()
    violations = C.check_drop_section(tl)
    C.repair(tl, violations)
    assert tl[1]["transition_in"]["tier"] == "minor"
    assert tl[1]["transition_in"]["name"] == "crossfade"


def test_validate_returns_all_violations():
    tl = _drop_violating_timeline()
    out = C.validate(tl, clips_meta={})
    assert any(v.rule == "drop_section" for v in out)
