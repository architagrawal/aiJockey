"""Smoke tests for tier-1 model wrappers.

Each wrapper must:
  1. Import cleanly when its underlying lib is absent.
  2. enabled() reports False unless env opts in.
  3. enabled() reports False when env opts in but lib missing.
  4. Public-API functions return None gracefully (don't crash).

Real model loads are NOT exercised here — would require multi-GB
checkpoint downloads. End-to-end validation runs on the droplet
separately.
"""
from __future__ import annotations

import os
import sys
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload(name: str):
    """Force fresh import (module-level state is sticky)."""
    if name in sys.modules:
        del sys.modules[name]
    return importlib.import_module(name)


@pytest.fixture(autouse=True)
def _clear_tier1_env(monkeypatch):
    """Each test starts with all tier-1 knobs unset."""
    for k in (
        "AIJOCKEY_ALL_IN_ONE",
        "AIJOCKEY_MEL_BAND_ROFORMER",
        "AIJOCKEY_MEL_BAND_ROFORMER_CKPT",
        "AIJOCKEY_AUDIOBOX_AESTHETICS",
    ):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# all_in_one_wrapper
# ---------------------------------------------------------------------------

def test_all_in_one_imports_without_lib():
    """Wrapper imports even when allin1 isn't installed."""
    mod = _reload("all_in_one_wrapper")
    assert hasattr(mod, "enabled")
    assert hasattr(mod, "analyze_audio_path")
    assert hasattr(mod, "beats_and_downbeats")
    assert hasattr(mod, "sections_for_clip")


def test_all_in_one_disabled_by_default():
    mod = _reload("all_in_one_wrapper")
    assert mod.enabled() is False


def test_all_in_one_disabled_when_lib_missing(monkeypatch):
    """Even when env enabled, returns False if allin1 not importable."""
    monkeypatch.setenv("AIJOCKEY_ALL_IN_ONE", "1")
    mod = _reload("all_in_one_wrapper")
    # If allin1 is actually installed in the test env, skip — we can't
    # reliably make it unimportable. Test only the absent-lib branch.
    try:
        import allin1   # noqa: F401
        pytest.skip("allin1 actually installed; absent-lib path not testable")
    except ImportError:
        pass
    assert mod.enabled() is False


def test_all_in_one_returns_none_on_failure(monkeypatch):
    """analyze_audio_path returns None instead of raising when load fails."""
    monkeypatch.setenv("AIJOCKEY_ALL_IN_ONE", "1")
    mod = _reload("all_in_one_wrapper")
    result = mod.analyze_audio_path("/nonexistent/path.wav", device="cpu")
    assert result is None


def test_all_in_one_label_to_energy_known_labels():
    mod = _reload("all_in_one_wrapper")
    # chorus > verse > inst > intro
    assert mod._label_to_energy("chorus") > mod._label_to_energy("verse")
    assert mod._label_to_energy("verse") > mod._label_to_energy("intro")
    assert mod._label_to_energy("UNKNOWN_LABEL_XYZ") == 0.5


# ---------------------------------------------------------------------------
# mel_band_roformer_wrapper
# ---------------------------------------------------------------------------

def test_mel_band_imports_without_lib():
    mod = _reload("mel_band_roformer_wrapper")
    assert hasattr(mod, "enabled")
    assert hasattr(mod, "vocals_from_wav")


def test_mel_band_disabled_by_default():
    mod = _reload("mel_band_roformer_wrapper")
    assert mod.enabled() is False


def test_mel_band_disabled_without_checkpoint_path(monkeypatch):
    """Env enabled but no checkpoint path → still disabled."""
    monkeypatch.setenv("AIJOCKEY_MEL_BAND_ROFORMER", "1")
    monkeypatch.setenv("AIJOCKEY_MEL_BAND_ROFORMER_CKPT", "")
    mod = _reload("mel_band_roformer_wrapper")
    assert mod.enabled() is False


def test_mel_band_enabled_with_path(monkeypatch, tmp_path):
    """Env + checkpoint path set → enabled() True (load lazy)."""
    monkeypatch.setenv("AIJOCKEY_MEL_BAND_ROFORMER", "1")
    monkeypatch.setenv("AIJOCKEY_MEL_BAND_ROFORMER_CKPT", str(tmp_path / "fake.ckpt"))
    mod = _reload("mel_band_roformer_wrapper")
    assert mod.enabled() is True


def test_mel_band_returns_none_on_missing_checkpoint(monkeypatch, tmp_path):
    """Vocals call with bad checkpoint path returns None, no crash."""
    monkeypatch.setenv("AIJOCKEY_MEL_BAND_ROFORMER", "1")
    monkeypatch.setenv("AIJOCKEY_MEL_BAND_ROFORMER_CKPT",
                       str(tmp_path / "definitely_not_real.ckpt"))
    mod = _reload("mel_band_roformer_wrapper")
    # Try to use it — should fail-fast, return None
    try:
        import torch
        wav = torch.zeros(2, 44100)
        result = mod.vocals_from_wav(wav, sr=44100, device="cpu")
        assert result is None
    except ImportError:
        pytest.skip("torch not installed in test env")


# ---------------------------------------------------------------------------
# audiobox_aesthetics
# ---------------------------------------------------------------------------

def test_audiobox_imports_without_lib():
    mod = _reload("audiobox_aesthetics")
    assert hasattr(mod, "enabled")
    assert hasattr(mod, "score")
    assert hasattr(mod, "severity_proxy")


def test_audiobox_disabled_by_default():
    mod = _reload("audiobox_aesthetics")
    assert mod.enabled() is False


def test_audiobox_score_returns_none_when_disabled():
    mod = _reload("audiobox_aesthetics")
    assert mod.score("/nonexistent.wav") is None


def test_audiobox_severity_proxy_handles_none():
    mod = _reload("audiobox_aesthetics")
    assert mod.severity_proxy(None) is None
    assert mod.severity_proxy({}) is None


def test_audiobox_severity_proxy_high_quality_low_severity():
    """High PQ + CE → low severity (quality good, no fix needed)."""
    mod = _reload("audiobox_aesthetics")
    sev = mod.severity_proxy(
        {"PQ": 8.0, "PC": 5.0, "CE": 8.0, "CU": 7.0},
        pq_target=6.0, ce_target=6.0,
    )
    assert sev == 0.0    # both above target → zero deficit


def test_audiobox_severity_proxy_low_quality_high_severity():
    """Low PQ + CE → high severity (quality bad, fix needed)."""
    mod = _reload("audiobox_aesthetics")
    sev = mod.severity_proxy(
        {"PQ": 2.0, "PC": 4.0, "CE": 2.0, "CU": 3.0},
        pq_target=6.0, ce_target=6.0,
    )
    # ((6-2)/6 + (6-2)/6) / 2 = 0.667
    assert 0.6 < sev < 0.7


def test_audiobox_severity_proxy_in_unit_range():
    mod = _reload("audiobox_aesthetics")
    for pq in (0.0, 3.0, 6.0, 10.0):
        for ce in (0.0, 3.0, 6.0, 10.0):
            sev = mod.severity_proxy(
                {"PQ": pq, "PC": 5.0, "CE": ce, "CU": 5.0},
            )
            assert 0.0 <= sev <= 1.0
