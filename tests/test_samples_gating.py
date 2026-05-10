"""Sample lib whitelist gating."""
import os
import sys
from pathlib import Path

import pytest

# samples.py imports torchaudio. Skip tests if not installed (laptop env).
pytest.importorskip("torchaudio")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_phase1_disallows_meme_types():
    os.environ["AIJOCKEY_PHASE"] = "1"
    if "samples" in sys.modules:
        del sys.modules["samples"]
    import samples
    bank = samples.SampleBank(samples_dir="samples")
    # Meme/novelty types should silently return silence
    out = bank.get_fx("airhorns", bpm=128.0, beats=1.0)
    assert out.shape[0] == 2
    assert (out == 0).all()
    # `has` should report False on disallowed types
    assert not bank.has("airhorns")


def test_phase1_allows_dj_fx_types():
    os.environ["AIJOCKEY_PHASE"] = "1"
    if "samples" in sys.modules:
        del sys.modules["samples"]
    import samples
    bank = samples.SampleBank(samples_dir="samples")
    for t in samples.PHASE1_ALLOWED_TYPES:
        # Either real samples or synth fallback should respond
        assert bank.has(t), f"Phase 1 should allow {t}"


def test_phase2_allows_all_types():
    os.environ["AIJOCKEY_PHASE"] = "2"
    for mod in ("samples", "synth_fx"):
        if mod in sys.modules:
            del sys.modules[mod]
    import samples
    import synth_fx
    # Precondition: airhorns must actually have a synth fallback. Without
    # this assert, the test silently passes on environments missing it.
    assert "airhorns" in synth_fx.SYNTHESIZERS, \
        "test precondition: synth_fx.SYNTHESIZERS must contain 'airhorns'"
    bank = samples.SampleBank(samples_dir="samples")
    assert bank.has("airhorns")
    os.environ["AIJOCKEY_PHASE"] = "1"
