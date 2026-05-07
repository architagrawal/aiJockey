"""
Smart sample resolver. Layers:
1. Real samples loaded from samples/manifest.json (curated CC0 audio)
2. Procedural fallback via synth_fx.py (always available)

Public API:
    get_fx(type, bpm, beats)  -> np.ndarray (2, T)
    list_available_types()    -> list[str]
    SampleBank class          -> manage real samples + synth fallback
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torchaudio

from synth_fx import synthesize, SYNTHESIZERS

SR = 44100


class SampleBank:
    """
    Holds real samples loaded from manifest. On request, returns best-matching
    real sample (closest BPM + length) or synthesized fallback.
    """

    def __init__(self, samples_dir: str = 'samples'):
        self.samples_dir = Path(samples_dir)
        # type -> list of {file, audio, bpm, length_beats, key}
        self.bank: dict[str, list[dict]] = {}
        self._load()

    def _load(self) -> None:
        manifest_path = self.samples_dir / 'manifest.json'
        if not manifest_path.exists():
            return
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception as e:
            print(f"warn: failed to read {manifest_path}: {e}")
            return
        for entry in manifest:
            fp = self.samples_dir / entry['file']
            if not fp.exists():
                print(f"warn: sample missing: {fp}")
                continue
            try:
                wav, sr = torchaudio.load(str(fp))
                if sr != SR:
                    wav = torchaudio.functional.resample(wav, sr, SR)
                if wav.size(0) == 1:
                    wav = wav.repeat(2, 1)
                elif wav.size(0) > 2:
                    wav = wav[:2]
            except Exception as e:
                print(f"warn: failed to load {fp}: {e}")
                continue
            self.bank.setdefault(entry['type'], []).append({
                'file': entry['file'],
                'audio': wav.numpy().astype(np.float32),
                'bpm': entry.get('bpm', None),  # can be 'agnostic' or number
                'length_beats': entry.get('length_beats', None),
                'key': entry.get('key', None),
            })

    def has_real(self, fx_type: str) -> bool:
        return bool(self.bank.get(fx_type))

    def has(self, fx_type: str) -> bool:
        """True if either real samples OR synth available for this type."""
        return self.has_real(fx_type) or fx_type in SYNTHESIZERS

    def list_available_types(self) -> list[str]:
        return sorted(set(list(self.bank.keys()) + list(SYNTHESIZERS.keys())))

    def get_fx(self, fx_type: str, bpm: float = 128.0,
               beats: float = 1.0,
               prefer_real: bool = True) -> np.ndarray:
        """
        Return FX audio (2, T). Tries real bank first if prefer_real, else synth.
        Falls back to other source if first choice missing. Returns silence
        if neither available.
        """
        target_n = int(beats * 60.0 / max(bpm, 1.0) * SR)
        if prefer_real and self.has_real(fx_type):
            sample = self._best_match(fx_type, bpm, beats)
            if sample is not None:
                return self._fit_length(sample, target_n)
        # Synth fallback
        synth = synthesize(fx_type, bpm=bpm, beats=beats)
        if synth is not None:
            return self._fit_length(synth, target_n)
        # Last resort: real even if not preferred
        if self.has_real(fx_type):
            sample = self._best_match(fx_type, bpm, beats)
            if sample is not None:
                return self._fit_length(sample, target_n)
        return np.zeros((2, max(1, target_n)), dtype=np.float32)

    def _best_match(self, fx_type: str, bpm: float, beats: float) -> np.ndarray | None:
        items = self.bank.get(fx_type, [])
        if not items:
            return None
        # Score: prefer matching BPM (or 'agnostic'), prefer matching length_beats
        def score(item: dict) -> float:
            s = 0.0
            ib = item.get('bpm')
            if ib == 'agnostic' or ib is None:
                s += 0.3
            else:
                try:
                    diff = abs(float(ib) - bpm) / max(bpm, 1.0)
                    s += max(0.0, 1.0 - diff * 5)  # 20% off -> 0
                except Exception:
                    s += 0.1
            il = item.get('length_beats')
            if il is None:
                s += 0.3
            else:
                try:
                    s += max(0.0, 1.0 - abs(float(il) - beats) / max(beats, 1.0))
                except Exception:
                    s += 0.1
            return s
        best = max(items, key=score)
        return best['audio']

    def _fit_length(self, audio: np.ndarray, target_n: int) -> np.ndarray:
        """Trim or zero-pad to target_n samples. Channels preserved."""
        if target_n <= 0:
            return audio
        cur_n = audio.shape[1]
        if cur_n == target_n:
            return audio
        if cur_n > target_n:
            return audio[:, :target_n]
        pad = target_n - cur_n
        return np.pad(audio, ((0, 0), (0, pad)))


# Module-level convenience singleton
_default_bank: SampleBank | None = None


def get_default_bank(samples_dir: str = 'samples') -> SampleBank:
    global _default_bank
    if _default_bank is None or _default_bank.samples_dir != Path(samples_dir):
        _default_bank = SampleBank(samples_dir)
    return _default_bank


def get_fx(fx_type: str, bpm: float = 128.0, beats: float = 1.0,
           samples_dir: str = 'samples', prefer_real: bool = True) -> np.ndarray:
    return get_default_bank(samples_dir).get_fx(fx_type, bpm, beats, prefer_real)
