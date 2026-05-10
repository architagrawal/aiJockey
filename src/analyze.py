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
import os
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
    def __init__(self, device: str = 'cuda', demucs_model: str | None = None):
        self.device = device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu'
        if device != self.device:
            print(f"warn: requested {device}, using {self.device}")
        # Lazy imports — heavy
        from demucs.pretrained import get_model
        # Default to htdemucs_ft (fine-tuned variant): same architecture as
        # htdemucs but ~0.5 dB SDR cleaner stems on MUSDB18, no extra cost.
        # Override with AIJOCKEY_DEMUCS_MODEL env var (e.g. 'htdemucs',
        # 'mdx_extra_q', or a custom checkpoint name).
        if demucs_model is None:
            demucs_model = os.environ.get('AIJOCKEY_DEMUCS_MODEL', 'htdemucs_ft')
        print(f"[analyze] demucs model: {demucs_model}")
        self.demucs = get_model(demucs_model).to(self.device)
        self.demucs.eval()
        # Use wrapper that picks laion-clap if available, else HF transformers
        from clap_wrapper import CLAP_Module
        self.clap = CLAP_Module(enable_fusion=False)
        self.clap.load_ckpt()
        # Phase A polish §16.2 efficiency hooks: bf16 + torch.compile.
        # Demucs benefits substantially from compile on MI300X.
        # NOTE: htdemucs_ft (and any BagOfModels) wraps multiple models and
        # does NOT expose `.segment` — torch.compile then breaks
        # demucs.apply.apply_model with AttributeError. Skip compile for
        # BagOfModels variants; bf16 autocast still applies in stems().
        try:
            from training.efficiency import maybe_compile, get_dtype
            from demucs.apply import BagOfModels
            self._compute_dtype = get_dtype()
            if self.device == 'cuda' and not isinstance(self.demucs, BagOfModels):
                # mode chosen by AIJOCKEY_COMPILE_MODE (defaults to 'default'
                # for ROCm-safe HIP graph behavior; opt into 'reduce-overhead'
                # explicitly when validated on the target accelerator).
                self.demucs = maybe_compile(self.demucs)
            elif isinstance(self.demucs, BagOfModels):
                print(f"[analyze] skipping torch.compile (BagOfModels {demucs_model} "
                      f"incompat with apply_model.segment access)")
        except Exception as e:
            print(f"[analyze] efficiency hooks skipped ({e})")
            self._compute_dtype = torch.float32
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
            print(f"[analyze] madmom unavailable ({e.__class__.__name__}); checking beat_this")
        # Beat-This! check: only when madmom missing (madmom is more accurate
        # on Western popular music when it loads, but it's Python-3.10 broken
        # in our env). beat_this is GPU-friendly + joint beats+downbeats.
        self._use_beat_this = False
        if not self.use_madmom:
            try:
                from beat_this_wrapper import available as _bt_avail, _load as _bt_load
                if _bt_avail():
                    _bt_load(device=self.device)
                    self._use_beat_this = True
                    print(f"[analyze] beats: beat_this ({self.device})")
                else:
                    print("[analyze] beats: librosa fallback (beat_this not installed)")
            except Exception as e:
                print(f"[analyze] beats: librosa fallback (beat_this load: {e})")

    def load(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.size(0) == 1:
            wav = wav.repeat(2, 1)
        elif wav.size(0) > 2:
            wav = wav[:2]
        return wav

    def _maybe_swap_vocals(self, wav: torch.Tensor,
                            stems: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Replace the demucs vocals stem with BS-Roformer output when enabled.

        BS-Roformer (or Mel-Band Roformer) yields ~+2 dB SDR on vocals vs
        htdemucs_ft. drums/bass/other still come from demucs (BS-Roformer is
        a vocals-only architecture in the published checkpoints).
        """
        try:
            from bs_roformer_wrapper import enabled as _bsr_enabled, vocals_from_wav
        except Exception:
            return stems
        if not _bsr_enabled():
            return stems
        new_vox = vocals_from_wav(wav, sr=SR, device=self.device)
        if new_vox is None:
            return stems
        # Crop/pad to match other stems' length (rounding may differ ±few samples).
        ref_len = min(s.shape[-1] for s in stems.values())
        v = new_vox[:, :ref_len] if new_vox.size(-1) >= ref_len else new_vox
        stems['vocals'] = v
        return stems

    def stems(self, wav: torch.Tensor) -> dict[str, torch.Tensor]:
        from demucs.apply import apply_model
        # overlap=0.25 was the demucs default tuned for hi-fi mastering;
        # for offline DJ-set analysis 0.10 is inaudible and ~15% faster.
        # Override via AIJOCKEY_DEMUCS_OVERLAP for A/B.
        ov = float(os.environ.get('AIJOCKEY_DEMUCS_OVERLAP', '0.10'))
        with torch.inference_mode():
            x = wav.unsqueeze(0).to(self.device)
            if self.device == 'cuda' and self._compute_dtype != torch.float32:
                # Mixed-precision demucs forward — bf16 on MI300X gives
                # ~1.5–2x throughput vs fp32 with no audible quality loss.
                with torch.amp.autocast(device_type='cuda',
                                         dtype=self._compute_dtype):
                    sources = apply_model(self.demucs, x, split=True,
                                          overlap=ov)[0]
            else:
                sources = apply_model(self.demucs, x, split=True, overlap=ov)[0]
        out = {n: sources[i].cpu() for i, n in enumerate(self.demucs.sources)}
        return self._maybe_swap_vocals(wav, out)

    def stems_batch(self, wavs: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Batched Demucs across N clips.

        Pads to longest, runs ONE apply_model call, slices back per-clip. On a
        192GB MI300X this lifts GPU util from ~4% (sequential 1-clip) to >50%
        on 8-clip batches with htdemucs_ft. Variable lengths handled by zero-pad
        + per-clip length crop on output.

        Falls back to per-clip stems() loop if any single batched forward
        fails (e.g. OOM on extreme outliers — pad to longest hurts when one
        clip is 10× others).
        """
        from demucs.apply import apply_model
        if not wavs:
            return []
        if len(wavs) == 1:
            return [self.stems(wavs[0])]
        # Pad to longest
        ch = wavs[0].size(0)
        T = max(w.size(1) for w in wavs)
        lengths = [w.size(1) for w in wavs]
        batch = torch.zeros(len(wavs), ch, T, dtype=wavs[0].dtype)
        for i, w in enumerate(wavs):
            batch[i, :, :w.size(1)] = w
        ov = float(os.environ.get('AIJOCKEY_DEMUCS_OVERLAP', '0.10'))
        try:
            with torch.inference_mode():
                x = batch.to(self.device)
                if self.device == 'cuda' and self._compute_dtype != torch.float32:
                    with torch.amp.autocast(device_type='cuda',
                                             dtype=self._compute_dtype):
                        sources = apply_model(self.demucs, x, split=True, overlap=ov)
                else:
                    sources = apply_model(self.demucs, x, split=True, overlap=ov)
            # sources: (B, S, C, T)
            names = list(self.demucs.sources)
            out: list[dict[str, torch.Tensor]] = []
            for b, n_samples in enumerate(lengths):
                d: dict[str, torch.Tensor] = {}
                for s, name in enumerate(names):
                    d[name] = sources[b, s, :, :n_samples].cpu()
                out.append(self._maybe_swap_vocals(wavs[b], d))
            return out
        except Exception as e:
            print(f"[analyze] batched stems failed ({e}); per-clip fallback")
            return [self.stems(w) for w in wavs]

    def beats_and_downbeats(self, wav: torch.Tensor) -> tuple[float, list[float], list[float]]:
        mono = wav.mean(0).numpy().astype(np.float32)
        # Beat-This! (P0): joint beats + downbeats from a single GPU transformer
        # pass. Replaces the librosa fallback `downbeats = beats[::4]` heuristic
        # which corrupts 3/4 of downbeats on swung / non-4/4 / pickup-bar material
        # and is the upstream cause of phrase-grid drift documented in HANDOFF.
        if not self.use_madmom and getattr(self, '_use_beat_this', False):
            try:
                from beat_this_wrapper import beats_from_array
                return beats_from_array(mono, SR, device=self.device)
            except Exception as e:
                # Fall through to librosa on first-call failure; cache flag so
                # we don't repeatedly retry per clip.
                print(f"[analyze] beat_this fallback ({e.__class__.__name__}: {e})")
                self._use_beat_this = False
        if self.use_madmom:
            beat_act = self.beat_proc(mono)
            beats = [float(t) for t in self.beat_track(beat_act)]
            db_act = self.dbn_proc(mono)
            db_out = self.dbn_track(db_act)
            downbeats = [float(t) for t, b in db_out if int(b) == 1]
        else:
            tempo_lr, beat_frames = librosa.beat.beat_track(y=mono, sr=SR, units='time')
            beats = [float(t) for t in beat_frames]
            # downbeat heuristic: every 4th beat (assume 4/4) — last-resort fallback
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
        # Vocal activity reference (per-section vocals stem RMS / total)
        vox_rms_ref = None
        if stems and 'vocals' in stems:
            vox_mono = stems['vocals'].mean(0).numpy().astype(np.float32)
            vox_rms_ref = librosa.feature.rms(y=vox_mono)[0]
            inst_signal = sum(stems[k].mean(0).numpy().astype(np.float32)
                              for k in ('drums', 'bass', 'other') if k in stems)
            inst_rms_ref = (librosa.feature.rms(y=inst_signal)[0]
                            if hasattr(inst_signal, 'shape') else None)
        else:
            inst_rms_ref = None
        out: list[dict] = []
        for i in range(len(bound_times) - 1):
            s, e = bound_times[i], bound_times[i + 1]
            mask = (rms_times >= s) & (rms_times < e)
            energy = float(rms[mask].mean()) if mask.any() else 0.0
            section_dict = {'start': float(s), 'end': float(e), 'energy': energy}
            if vox_rms_ref is not None and inst_rms_ref is not None and mask.any():
                v = float(vox_rms_ref[mask].mean())
                i_ = float(inst_rms_ref[mask].mean()) + 1e-8
                section_dict['vocal_activity'] = round(v / (v + i_ + 1e-8), 4)
            out.append(section_dict)
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

    def analyze(self, path: str, clip_id: str, cache_dir: Path,
                precomputed_clap: np.ndarray | None = None) -> ClipAnalysis:
        wav = self.load(path)
        # Stems (slow). If caller pre-batched stems via stems_batch() and
        # parked them on `_pre_stems`, reuse — saves the per-clip GPU forward
        # which is the largest single cost on stage1.
        pre = getattr(self, '_pre_stems', None)
        if pre is not None and isinstance(pre, dict) and pre:
            stems = pre
        else:
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
        # CLAP embedding — accept precomputed value to enable cross-clip
        # batched CLAP at the caller layer (stage1_analyze process_batch).
        if precomputed_clap is not None:
            clap = precomputed_clap.astype(np.float32)
        else:
            clap = self.clap_embedding(wav)

        return ClipAnalysis(
            clip_id=clip_id, path=str(path),
            duration=wav.size(1) / SR, sample_rate=SR,
            tempo=tempo, beats=beats, downbeats=downbeats,
            phrase_bars=phrase_bars,
            key=key, key_confidence=key_conf,
            sections=sections, hooks=hook_list,
        ), clap, energy


def _analyze_one(args: tuple) -> str:
    """Worker function: analyze a single clip. Loads Analyzer per-process."""
    clip_path, cache_dir, device, worker_id = args
    cache_path = Path(cache_dir)
    p = Path(clip_path)
    clip_id = p.stem
    json_path = cache_path / f'{clip_id}.json'
    npz_path = cache_path / f'{clip_id}.npz'
    if json_path.exists() and npz_path.exists():
        return f"[w{worker_id}] skip {clip_id} (cached)"
    # Lazy-init per-process analyzer (model load ~5s)
    global _WORKER_ANALYZER
    if '_WORKER_ANALYZER' not in globals() or _WORKER_ANALYZER is None:
        _WORKER_ANALYZER = Analyzer(device=device)
    result, clap, energy = _WORKER_ANALYZER.analyze(str(p), clip_id, cache_path)
    np.savez_compressed(str(npz_path), clap=clap, energy=energy)
    with open(json_path, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    return f"[w{worker_id}] saved {clip_id}"


def analyze_pool(clip_dir: str, cache_dir: str, device: str = 'cuda',
                 force: bool = False, workers: int = 1) -> None:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    audio_exts = ('*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg')
    paths: list[Path] = []
    for ext in audio_exts:
        paths.extend(sorted(Path(clip_dir).glob(ext)))
    if not paths:
        print(f"no audio in {clip_dir}")
        return

    # filter cached if not forcing
    pending: list[Path] = []
    for p in paths:
        cid = p.stem
        if not force and (cache_path / f'{cid}.json').exists() and (cache_path / f'{cid}.npz').exists():
            print(f"skip {cid} (cached)")
            continue
        pending.append(p)
    if not pending:
        print("nothing to analyze")
        return

    if workers <= 1:
        # sequential path (single Analyzer reused)
        analyzer = Analyzer(device=device)
        for p in pending:
            cid = p.stem
            print(f"analyzing {cid}")
            result, clap, energy = analyzer.analyze(str(p), cid, cache_path)
            np.savez_compressed(str(cache_path / f'{cid}.npz'), clap=clap, energy=energy)
            with open(cache_path / f'{cid}.json', 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"  saved {cid}")
        return

    # parallel path
    import multiprocessing as mp
    ctx = mp.get_context('spawn')  # safe with CUDA/ROCm
    n = min(workers, len(pending))
    print(f"parallel analyze: {len(pending)} clips, {n} workers")
    args_list = [(str(p), str(cache_path), device, i % n) for i, p in enumerate(pending)]
    with ctx.Pool(processes=n) as pool:
        for msg in pool.imap_unordered(_analyze_one, args_list):
            print(msg)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', required=True)
    ap.add_argument('--cache', default='cache')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    analyze_pool(args.clips, args.cache, args.device, args.force)
