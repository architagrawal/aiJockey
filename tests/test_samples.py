"""Tests for SampleBank — falls back to synth when manifest empty."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import json
import numpy as np
from samples import SampleBank


def test_empty_bank_falls_back_to_synth(tmp_path):
    # No manifest -> bank empty -> all FX come from synth
    bank = SampleBank(samples_dir=str(tmp_path))
    out = bank.get_fx('impacts', bpm=128, beats=1)
    assert out.shape[0] == 2
    assert out.shape[1] > 0
    assert np.isfinite(out).all()


def test_get_fx_unknown_type_returns_silence(tmp_path):
    bank = SampleBank(samples_dir=str(tmp_path))
    out = bank.get_fx('totally_made_up', bpm=128, beats=1)
    assert (out == 0).all()


def test_list_available_includes_all_synth(tmp_path):
    bank = SampleBank(samples_dir=str(tmp_path))
    types = bank.list_available_types()
    for required in ('impacts', 'risers', 'sweeps', 'snare_rolls',
                     'sub_drops', 'vinyl', 'airhorns', 'hihat_rolls'):
        assert required in types


def test_fit_length_pads(tmp_path):
    bank = SampleBank(samples_dir=str(tmp_path))
    target_n = 44100  # 1 sec
    out = bank.get_fx('vinyl', bpm=128, beats=1)  # vinyl_stop yields ~beat_dur
    # Should match target length closely (synth path)
    assert out.shape[1] == target_n


def test_missing_manifest_no_crash(tmp_path):
    # Just creating a bank with empty dir should not crash
    bank = SampleBank(samples_dir=str(tmp_path / 'nonexistent'))
    assert isinstance(bank.bank, dict)


def test_bad_manifest_no_crash(tmp_path):
    # Malformed JSON should warn but not crash
    (tmp_path / 'manifest.json').write_text('not json')
    bank = SampleBank(samples_dir=str(tmp_path))
    assert bank.bank == {}
