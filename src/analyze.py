"""Per-clip analysis: stems, beats, downbeats, key, sections, hooks, energy, CLAP.

Outputs to cache/<clip_id>.json (metadata) + cache/<clip_id>.npz (clap+energy arrays).
Stems saved to cache/stems/<clip_id>/{drums,bass,other,vocals}.wav.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Allow sibling imports when run as `python src/analyze.py`
sys.path.insert(0, str(Path(__file__).parent))

import json
from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np
import torch
import torchaudio
import librosa

from camelot import krumhansl_key, to_camelot
from hooks import detect_hooks
from phrase import detect_phrase_length

SR = 44100  # working sample rate


@dataclass
class ClipAnalysis:
    clip_id: str
    path: str
    duration: float
    sample_rate: int
    tempo: float
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    phrase_bars: int = 16
    key: str = '?'
    key_confidence: float = 0.0
    sections: list[dict] = field(default_factory=list)
    hooks: list[dict] = field(default_factory=list)


class Analyzer:
    def __init__(self, device: str = 'cuda', demucs_model: str = 'htdemucs'):
        self.device = device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu'
        if device != self.device:
            print(f"warn: requested {device}, using {self.device}")
        # Lazy imports — heavy
        from demucs.pretrained import get_model
        self.demucs = get_model(demucs_model).to(self.device)
        self.demucs.eval()
        # Use wrapper that picks laion-clap if available, else HF transformers
        from clap_wrapper import CLAP_Module
        self.clap = CLAP_Module(enable_fusion=False)
        self.clap.load_ckpt()
        self.use_madmom = False
        try:
            import madmom
            self.beat_proc = madmom.features.beats.RNNBeatProcessor()
            self.beat_track = madmom.features.beats.BeatTrackingProcessor(fps=100)
            self.dbn_proc = madmom.features.downbeats.RNNDownBeatProcessor()
            self.dbn_track = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
                beats_per_bar=[3, 4], fps=100)
            self.use_madmom = True
        except Exception as e:
            print(f"[analyze] madmom unavailable ({e.__class__.__name__}); using librosa beat fallback")

    def load(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.size(0) == 1:
            wav = wav.repeat(2, 1)
        elif wav.size(0) > 2:
            wav = wav[:2]
        return wav

    def stems(self, wav: torch.Tensor) -> dict[str, torch.Tensor]:
        from demucs.apply import apply_model
        with torch.no_grad():
            x = wav.unsqueeze(0).to(self.device)
            sources = apply_model(self.demucs, x, split=True, overlap=0.25)[0]
        return {n: sources[i].cpu() for i, n in enumerate(self.demucs.sources)}

    def beats_and_downbeats(self, wav: torch.Tensor) -> tuple[float, list[float], list[float]]:
        mono = wav.mean(0).numpy().astype(np.float32)
        if self.use_madmom:
            beat_act = self.beat_proc(mono)
            beats = [float(t) for t in self.beat_track(beat_act)]
            db_act = self.dbn_proc(mono)
            db_out = self.dbn_track(db_act)
            downbeats = [float(t) for t, b in db_out if int(b) == 1]
        else:
            tempo_lr, beat_frames = librosa.beat.beat_track(y=mono, sr=SR, units='time')
            beats = [float(t) for t in beat_frames]
            # downbeat heuristic: every 4th beat (assume 4/4)
            downbeats = beats[::4]
        if len(beats) > 1:
            ibis = np.diff(beats)
            tempo = 60.0 / float(np.median(ibis))
        else:
            tempo = 0.0
        return tempo, beats, downbeats

    def key_camelot(self, wav: torch.Tensor) -> tuple[str, float]:
        mono = wav.mean(0).numpy().astype(np.float32)
        chroma = librosa.feature.chroma_cqt(y=mono, sr=SR)
        chroma_mean = chroma.mean(axis=1)
        pitch_class, mode, conf = krumhansl_key(chroma_mean)
        return to_camelot(pitch_class, mode), float(conf)

    def sections(self, wav: torch.Tensor, n_segments: int = 8,
                 stems: dict[str, torch.Tensor] | None = None) -> list[dict]:
        """
        Segment via full-mix MFCC (timbral changes), label energy via drums+bass
        stem RMS (drops characterized by heavy low-end). Falls back to full-mix
        RMS if stems unavailable.
        """
        mono = wav.mean(0).numpy().astype(np.float32)
        mfcc = librosa.feature.mfcc(y=mono, sr=SR, n_mfcc=13)
        bounds = librosa.segment.agglomerative(mfcc, k=n_segments)
        bound_times = librosa.frames_to_time(bounds, sr=SR).tolist()
        bound_times = [0.0] + bound_times + [len(mono) / SR]
        bound_times = sorted(set(bound_times))

        # Energy reference: drums+bass stems if available (Agent 3 stem-aware)
        if stems and 'drums' in stems and 'bass' in stems:
            db_mix = (stems['drums'] + stems['bass']).mean(0).numpy().astype(np.float32)
            energy_signal = db_mix
        else:
            energy_signal = mono

        rms = librosa.feature.rms(y=energy_signal)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=SR)
        out: list[dict] = []
        for i in range(len(bound_times) - 1):
            s, e = bound_times[i], bound_times[i + 1]
            mask = (rms_times >= s) & (rms_times < e)
            energy = float(rms[mask].mean()) if mask.any() else 0.0
            out.append({'start': float(s), 'end': float(e), 'energy': energy})
        if not out:
            return out
        peak_e = max(s['energy'] for s in out)
        for i, s in enumerate(out):
            rel = s['energy'] / peak_e if peak_e > 0 else 0.0
            if i == 0 and rel < 0.5:
                s['type'] = 'intro'
            elif i == len(out) - 1 and rel < 0.5:
                s['type'] = 'outro'
            elif rel > 0.85:
                s['type'] = 'drop'
            elif rel > 0.6:
                s['type'] = 'verse'
            else:
                s['type'] = 'breakdown'
        return out

    def energy_curve(self, wav: torch.Tensor, hop_hz: int = 10) -> np.ndarray:
        mono = wav.mean(0).numpy().astype(np.float32)
        hop = SR // hop_hz
        rms = librosa.feature.rms(y=mono, frame_length=hop * 2, hop_length=hop)[0]
        return rms.astype(np.float32)

    def clap_embedding(self, wav: torch.Tensor) -> np.ndarray:
        mono = wav.mean(0).numpy().astype(np.float32)
        mono_48 = librosa.resample(mono, orig_sr=SR, target_sr=48000)
        # CLAP_Module here is our wrapper — works with both backends.
        emb = self.clap.get_audio_embedding_from_data(mono_48[None, :], use_tensor=False)
        # transformers backend returns (1, 512); laion returns (1, 512). Both fine.
        return emb[0].astype(np.float32)

    def analyze(self, path: str, clip_id: str, cache_dir: Path) -> ClipAnalysis:
        wav = self.load(path)
        # Stems (slow)
        stems = self.stems(wav)
        stems_dir = cache_dir / 'stems' / clip_id
        stems_dir.mkdir(parents=True, exist_ok=True)
        for name, s in stems.items():
            torchaudio.save(str(stems_dir / f'{name}.wav'), s, SR)
        # Beats/downbeats/tempo
        tempo, beats, downbeats = self.beats_and_downbeats(wav)
        # Key
        key, key_conf = self.key_camelot(wav)
        # Sections (stem-aware: uses drums+bass RMS for energy labeling)
        sections = self.sections(wav, stems=stems)
        # Energy curve
        energy = self.energy_curve(wav)
        # Phrase length
        phrase_bars = detect_phrase_length(downbeats, energy.tolist())
        # Hooks
        mono = wav.mean(0).numpy().astype(np.float32)
        hook_list = detect_hooks(mono, SR, downbeats)
        # CLAP embedding
        clap = self.clap_embedding(wav)

        return ClipAnalysis(
            clip_id=clip_id, path=str(path),
            duration=wav.size(1) / SR, sample_rate=SR,
            tempo=tempo, beats=beats, downbeats=downbeats,
            phrase_bars=phrase_bars,
            key=key, key_confidence=key_conf,
            sections=sections, hooks=hook_list,
        ), clap, energy


def analyze_pool(clip_dir: str, cache_dir: str, device: str = 'cuda',
                 force: bool = False) -> None:
    analyzer = Analyzer(device=device)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    audio_exts = ('*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg')
    paths: list[Path] = []
    for ext in audio_exts:
        paths.extend(sorted(Path(clip_dir).glob(ext)))
    if not paths:
        print(f"no audio in {clip_dir}")
        return
    for p in paths:
        clip_id = p.stem
        json_path = cache_path / f'{clip_id}.json'
        npz_path = cache_path / f'{clip_id}.npz'
        if not force and json_path.exists() and npz_path.exists():
            print(f"skip {clip_id} (cached)")
            continue
        print(f"analyzing {clip_id}")
        result, clap, energy = analyzer.analyze(str(p), clip_id, cache_path)
        np.savez_compressed(str(npz_path), clap=clap, energy=energy)
        with open(json_path, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        print(f"  saved {json_path.name} + {npz_path.name}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', required=True)
    ap.add_argument('--cache', default='cache')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    analyze_pool(args.clips, args.cache, args.device, args.force)
