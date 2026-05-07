"""
Quantitative eval metrics for generated mixes.

Usage:
    python src/eval.py --mix output/final_mix.wav --timeline output/timeline.json
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import librosa
import torchaudio
import pyloudnorm as pyln


def _detect_beats(mix: np.ndarray, sr: int) -> np.ndarray:
    mono = mix.mean(axis=0) if mix.ndim > 1 else mix
    _, beat_frames = librosa.beat.beat_track(y=mono, sr=sr, units='frames')
    return librosa.frames_to_time(beat_frames, sr=sr)


def beat_continuity_score(mix: np.ndarray, sr: int,
                          transition_times: list[float],
                          tol_sec: float = 0.05) -> float:
    """% of transitions where a beat is within tol_sec of transition boundary."""
    beat_times = _detect_beats(mix, sr)
    if len(beat_times) == 0 or not transition_times:
        return 0.0
    hits = sum(1 for t in transition_times
               if float(np.min(np.abs(beat_times - t))) <= tol_sec)
    return hits / len(transition_times)


def beat_alignment_error_ms(mix: np.ndarray, sr: int,
                            transition_times: list[float]) -> dict:
    """
    Raw ms offset from each transition boundary to nearest detected beat.
    Returns mean, max, p50, p90 in milliseconds.
    """
    beat_times = _detect_beats(mix, sr)
    if len(beat_times) == 0 or not transition_times:
        return {'mean_ms': float('nan'), 'max_ms': float('nan'),
                'p50_ms': float('nan'), 'p90_ms': float('nan'), 'n': 0}
    offsets = np.array([float(np.min(np.abs(beat_times - t)))
                        for t in transition_times]) * 1000.0
    return {
        'mean_ms': float(offsets.mean()),
        'max_ms': float(offsets.max()),
        'p50_ms': float(np.percentile(offsets, 50)),
        'p90_ms': float(np.percentile(offsets, 90)),
        'n': int(len(offsets)),
    }


def energy_arc_correlation(mix: np.ndarray, sr: int,
                           target_arc: list[float]) -> float:
    """Pearson correlation between RMS over time and target arc shape."""
    if mix.ndim > 1:
        mono = mix.mean(axis=0)
    else:
        mono = mix
    rms = librosa.feature.rms(y=mono, frame_length=sr, hop_length=sr // 2)[0]
    if rms.size == 0 or len(target_arc) == 0:
        return 0.0
    target = np.interp(np.linspace(0, 1, len(rms)),
                       np.linspace(0, 1, len(target_arc)),
                       np.asarray(target_arc, dtype=np.float32))
    if rms.std() == 0 or target.std() == 0:
        return 0.0
    return float(np.corrcoef(rms, target)[0, 1])


def measure_lufs(mix: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(mix.T if mix.ndim > 1 else mix))


def true_peak_db(mix: np.ndarray) -> float:
    peak = float(np.abs(mix).max())
    if peak <= 0:
        return -np.inf
    return 20.0 * float(np.log10(peak))


def _embed_chunks_clap(mix: np.ndarray, sr: int, chunk_sec: float = 10.0,
                       clap_module=None) -> np.ndarray:
    """Embed audio in chunks via CLAP. Returns (N, 512). Lazy-loads CLAP."""
    if clap_module is None:
        from clap_wrapper import CLAP_Module
        clap_module = CLAP_Module(enable_fusion=False)
        clap_module.load_ckpt()
    mono = mix.mean(axis=0) if mix.ndim > 1 else mix
    if sr != 48000:
        mono = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=48000)
    chunk_n = int(chunk_sec * 48000)
    embs = []
    for s in range(0, len(mono) - chunk_n // 2, chunk_n):
        chunk = mono[s:s + chunk_n]
        if len(chunk) < chunk_n // 2:
            continue
        if len(chunk) < chunk_n:
            chunk = np.pad(chunk, (0, chunk_n - len(chunk)))
        emb = clap_module.get_audio_embedding_from_data(chunk[None, :], use_tensor=False)
        embs.append(emb[0])
    if not embs:
        return np.zeros((0, 512), dtype=np.float32)
    return np.asarray(embs, dtype=np.float32)


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                      mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """FAD = ||mu1 - mu2||^2 + tr(sigma1 + sigma2 - 2 * sqrt(sigma1 @ sigma2))."""
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def fad_against_reference(mix: np.ndarray, sr: int, reference_clap_emb: np.ndarray,
                          chunk_sec: float = 10.0) -> float:
    """
    FAD between generated mix and reference CLAP embedding distribution.
    reference_clap_emb: (N, 512) — typically loaded from cache/*.npz clap arrays
    of curated reference clips. Returns NaN if too few chunks.
    """
    gen_emb = _embed_chunks_clap(mix, sr, chunk_sec)
    if gen_emb.shape[0] < 2 or reference_clap_emb.shape[0] < 2:
        return float('nan')
    mu_g, mu_r = gen_emb.mean(axis=0), reference_clap_emb.mean(axis=0)
    sig_g = np.cov(gen_emb, rowvar=False) + np.eye(512) * 1e-6
    sig_r = np.cov(reference_clap_emb, rowvar=False) + np.eye(512) * 1e-6
    return _frechet_distance(mu_g, sig_g, mu_r, sig_r)


def load_reference_clap(cache_dir: str) -> np.ndarray:
    """Load all CLAP embeddings from cache/*.npz as reference set."""
    from pathlib import Path as _P
    embs = []
    for p in sorted(_P(cache_dir).glob('*.npz')):
        d = np.load(str(p))
        if 'clap' in d:
            embs.append(d['clap'])
    if not embs:
        return np.zeros((0, 512), dtype=np.float32)
    return np.asarray(embs, dtype=np.float32)


def evaluate(mix_path: str, timeline_path: str,
             energy_arc: list[float] | None = None,
             lufs_low: float = -10.0, lufs_high: float = -7.0,
             reference_cache_dir: str | None = None) -> dict:
    wav, sr = torchaudio.load(mix_path)
    mix = wav.numpy().astype(np.float32)
    with open(timeline_path) as f:
        tl = json.load(f)['timeline']

    transition_times = [float(e['play_at']) for e in tl[1:]]
    arc = energy_arc or [0.3, 0.5, 0.7, 0.9, 1.0, 0.85, 0.6, 0.3]

    lufs = measure_lufs(mix, sr)
    result = {
        'mix_path': str(mix_path),
        'timeline_path': str(timeline_path),
        'duration_sec': mix.shape[-1] / sr,
        'n_transitions': len(transition_times),
        'beat_continuity': beat_continuity_score(mix, sr, transition_times),
        'beat_alignment_error': beat_alignment_error_ms(mix, sr, transition_times),
        'energy_arc_correlation': energy_arc_correlation(mix, sr, arc),
        'lufs': lufs,
        'lufs_in_range': bool(lufs_low <= lufs <= lufs_high),
        'true_peak_db': true_peak_db(mix),
    }
    if reference_cache_dir:
        ref = load_reference_clap(reference_cache_dir)
        if ref.shape[0] >= 2:
            try:
                result['fad'] = fad_against_reference(mix, sr, ref)
            except Exception as e:
                result['fad'] = f"error: {e}"
        else:
            result['fad'] = 'insufficient reference embeddings (need >= 2)'
    return result


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mix', required=True)
    ap.add_argument('--timeline', required=True)
    ap.add_argument('--reference_cache', default=None,
                    help='cache dir of reference clips (for FAD)')
    args = ap.parse_args()
    result = evaluate(args.mix, args.timeline,
                      reference_cache_dir=args.reference_cache)
    print(json.dumps(result, indent=2))
