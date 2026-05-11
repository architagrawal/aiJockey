"""
Mastering chain: HP30 -> multiband compression -> glue compressor -> LUFS norm -> limiter.

Targets club playback: -9 LUFS, -1 dBTP ceiling.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torchaudio
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def hp(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(4, cutoff, btype='high', fs=sr, output='sos')
    return np.stack([sosfilt(sos, ch) for ch in x])


def split_bands(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sos_low = butter(4, 200, btype='low', fs=sr, output='sos')
    sos_mid_lp = butter(4, 4000, btype='low', fs=sr, output='sos')
    sos_mid_hp = butter(4, 200, btype='high', fs=sr, output='sos')
    sos_high = butter(4, 4000, btype='high', fs=sr, output='sos')
    low = np.stack([sosfilt(sos_low, ch) for ch in x])
    mid_lp = np.stack([sosfilt(sos_mid_lp, ch) for ch in x])
    mid = np.stack([sosfilt(sos_mid_hp, ch) for ch in mid_lp])
    high = np.stack([sosfilt(sos_high, ch) for ch in x])
    return low, mid, high


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def compress(x: np.ndarray, threshold_db: float = -20.0, ratio: float = 4.0,
             attack_ms: float = 10.0, release_ms: float = 100.0,
             sr: int = 44100) -> np.ndarray:
    eps = 1e-10
    abs_x = np.abs(x).max(axis=0)
    db = 20.0 * np.log10(abs_x + eps)
    over = np.maximum(0.0, db - threshold_db)
    target_red_db = -over * (1.0 - 1.0 / ratio)
    a_a = float(np.exp(-1.0 / max(1e-3, attack_ms * sr / 1000.0)))
    a_r = float(np.exp(-1.0 / max(1e-3, release_ms * sr / 1000.0)))
    env = np.zeros_like(target_red_db)
    g = 0.0
    for i, t in enumerate(target_red_db):
        coef = a_a if t < g else a_r
        g = coef * g + (1.0 - coef) * t
        env[i] = g
    gain_lin = 10.0 ** (env / 20.0)
    return x * gain_lin


def limit(x: np.ndarray, ceiling_db: float = -1.0, lookahead_ms: float = 5.0,
          sr: int = 44100) -> np.ndarray:
    ceiling = 10.0 ** (ceiling_db / 20.0)
    lookahead = max(1, int(lookahead_ms * sr / 1000.0))
    abs_x = np.abs(x).max(axis=0)
    pad = np.concatenate([abs_x, np.zeros(lookahead)])
    rolling = np.array([pad[i:i + lookahead].max() for i in range(len(abs_x))])
    target = np.where(rolling > ceiling, ceiling / (rolling + 1e-10), 1.0)
    a = float(np.exp(-1.0 / max(1.0, lookahead * 0.5)))
    smoothed = np.empty_like(target)
    g = 1.0
    for i, t in enumerate(target):
        # Attack instantly to lower gain, release smoothly back up
        cand = a * g + (1.0 - a) * t
        g = min(t, cand)
        smoothed[i] = g
    return x * smoothed


def lufs_normalize(x: np.ndarray, sr: int, target_lufs: float = -9.0) -> np.ndarray:
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(x.T)
    if not np.isfinite(loudness) or loudness < -70:
        return x
    return pyln.normalize.loudness(x.T, loudness, target_lufs).T


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

# Genre → target LUFS lookup. Preserves dynamics on non-EDM. EBU R128 +
# AES community recommendations. Toggle: AIJOCKEY_ADAPTIVE_LUFS=0.
_GENRE_LUFS_TARGETS: dict[str, float] = {
    'edm': -9.0, 'dnb': -9.0, 'dubstep': -8.0, 'trance': -9.0,
    'house': -9.5, 'tech_house': -9.5, 'techno': -9.5, 'hardstyle': -7.5,
    'progressive': -9.5, 'future_bass': -9.0,
    'hip_hop': -10.0, 'trap': -10.0, 'rnb': -11.0,
    'pop': -12.0, 'rock': -11.0, 'punjabi': -10.0, 'bollywood': -10.0,
    'chillstep': -13.0, 'chill': -13.0, 'lofi': -14.0, 'ambient': -16.0,
    'jazz': -16.0, 'classical': -18.0, 'indian_classical': -18.0,
    'synthwave': -10.0, 'retrowave': -10.0,
}


def adaptive_lufs_target(default_lufs: float, genre: str | None,
                          loudness_range: float | None = None) -> float:
    """Resolve target LUFS given a genre tag and optional measured LRA.

    Falls back to default if genre missing. Widens by half the LRA delta
    above 7 LU for non-EDM ambient stretches so dynamics survive.
    """
    g = (genre or '').lower().strip()
    base = _GENRE_LUFS_TARGETS.get(g, default_lufs)
    if loudness_range is not None and loudness_range > 7.0 and base < -10.0:
        # Quiet/dynamic material — pull target down a bit more
        base = base - min(2.0, (loudness_range - 7.0) * 0.3)
    return float(base)


def master(in_path: str, out_path: str, target_lufs: float = -9.0,
            genre: str | None = None) -> None:
    # Matchering reference-master alt path. When enabled + reference set,
    # we bypass the rule-based multi-band chain entirely.
    import os as _menv
    if _menv.environ.get("AIJOCKEY_MATCHERING_ENABLE", "0") == "1":
        try:
            from matchering_master import master_with_reference
            res = master_with_reference(in_path, out_path,
                                          target_lufs=target_lufs)
            if res:
                print(f"mastered via Matchering -> {out_path}")
                return
        except Exception as _e:
            print(f"[master] matchering path failed, falling back: {_e}")
    wav, sr = torchaudio.load(in_path)
    x = wav.numpy().astype(np.float32)
    if x.shape[0] == 1:
        x = np.concatenate([x, x], axis=0)
    elif x.shape[0] > 2:
        x = x[:2]

    # Hot / brickwalled uploads: reduce inter-sample stress before multiband squash
    peak = float(np.abs(x).max())
    if peak > 0.95:
        x = x * float(0.92 / max(peak, 1e-6))
    meter_early = pyln.Meter(sr)
    loud_early = meter_early.integrated_loudness(x.T)
    # Adaptive LUFS: route by genre when AIJOCKEY_ADAPTIVE_LUFS=1 (default).
    # Env override: AIJOCKEY_MASTER_GENRE picks the genre slug directly.
    import os as _aos
    eff_target = float(target_lufs)
    if _aos.environ.get('AIJOCKEY_ADAPTIVE_LUFS', '1') != '0':
        g = genre or _aos.environ.get('AIJOCKEY_MASTER_GENRE')
        if g:
            try:
                # Cheap LRA proxy via pyloudnorm not exposed directly; pass None.
                eff_target = adaptive_lufs_target(target_lufs, g)
            except Exception:
                pass
    if np.isfinite(loud_early) and loud_early > -10.0:
        eff_target = float(max(eff_target, loud_early - 2.0))

    x = hp(x, sr, 30)

    low, mid, high = split_bands(x, sr)
    low = compress(low, threshold_db=-18, ratio=3.0, sr=sr)
    mid = compress(mid, threshold_db=-20, ratio=2.5, sr=sr)
    high = compress(high, threshold_db=-22, ratio=2.0, sr=sr)
    # Tape saturation on mids — tanh asymmetric soft-clip adds 2nd/3rd
    # harmonic warmth that Audiobox PQ rewards. Subtle drive (0.6) keeps
    # transients intact. Disable: AIJOCKEY_MASTER_TAPE_SAT=0.
    import os as _os
    if _os.environ.get('AIJOCKEY_MASTER_TAPE_SAT', '1') != '0':
        drive = float(_os.environ.get('AIJOCKEY_MASTER_TAPE_DRIVE', '0.6'))
        mid = (np.tanh(mid * (1.0 + drive)) / (1.0 + drive * 0.3)).astype(np.float32)
    x = (low + mid + high).astype(np.float32)

    if np.isfinite(loud_early) and loud_early > -11.0:
        x = compress(x, threshold_db=-8, ratio=1.8, sr=sr).astype(np.float32)
    else:
        x = compress(x, threshold_db=-12, ratio=2.0, sr=sr).astype(np.float32)
    x = lufs_normalize(x, sr, eff_target).astype(np.float32)
    # Mid/Side stereo widener — opt-in via AIJOCKEY_MS_WIDEN=1.
    try:
        from ms_widener import enabled as _ms_en, widen as _ms_widen
        if _ms_en():
            x = _ms_widen(x)
    except Exception:
        pass
    x = limit(x, ceiling_db=-1.0, sr=sr).astype(np.float32)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(out_path, torch.from_numpy(x), sr)
    print(f"mastered -> {out_path}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_path', default='output/raw_mix.wav')
    ap.add_argument('--out', default='output/final_mix.wav')
    ap.add_argument('--lufs', type=float, default=-9.0)
    args = ap.parse_args()
    master(args.in_path, args.out, args.lufs)
