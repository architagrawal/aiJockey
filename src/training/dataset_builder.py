"""
Build transition dataset from downloaded DJ mixes + tracklist annotations.

Pipeline:
  1. For each tracklist JSON:
     a. Download mix audio (yt-dlp) if not cached
     b. For each marked transition timestamp:
        - Extract pre-window (8 bars before)
        - Extract transition window (4 bars centered on timestamp)
        - Extract post-window (8 bars after)
        - Compute features: tempo, beats, CLAP embeddings of pre/post
        - Optionally classify technique via heuristic
  2. Output: datasets/transitions_real/<mix_id>/<idx>.npz with
     {pre_audio, post_audio, transition_audio, features, technique_guess}

Each output sample becomes training input for fine-tuning a generative model
on (context -> transition_audio) pairs.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from dataclasses import asdict, dataclass
import numpy as np
import torch
import torchaudio

from scrape.youtube_dl import download_audio
from scrape.tracklists import MixTracklist, load_all
from clap_wrapper import get_audio_embedding


SR = 44100  # standard rate


@dataclass
class TransitionSample:
    mix_id: str
    idx: int
    at_sec: float
    pre_path: str
    transition_path: str
    post_path: str
    tempo_pre: float
    tempo_post: float
    technique_guess: str


def _detect_tempo_window(audio_mono: np.ndarray, sr: int) -> float:
    """Estimate tempo of an audio window. Returns 0 on failure."""
    try:
        import librosa
        tempo, _ = librosa.beat.beat_track(y=audio_mono.astype(np.float32), sr=sr)
        return float(tempo) if tempo else 0.0
    except Exception:
        return 0.0


def _classify_transition_heuristic(pre: np.ndarray, transition: np.ndarray,
                                   post: np.ndarray, sr: int) -> str:
    """
    Cheap auto-label of which technique was used at this transition.
    Imperfect — used as weak supervision. Refine via manual review or
    cross-check with a learned model later.
    """
    import librosa
    # 1. Detect silence in transition window -> silence_drop
    rms_t = librosa.feature.rms(y=transition.mean(axis=0).astype(np.float32))[0]
    if float(rms_t.min()) < 0.005 and float(rms_t.max()) > 0.05:
        return 'silence_drop'
    # 2. RMS continuity vs sudden jump -> cut vs crossfade
    rms_pre = float(np.sqrt(np.mean(pre[:, -sr:].astype(np.float32) ** 2)))
    rms_post = float(np.sqrt(np.mean(post[:, :sr].astype(np.float32) ** 2)))
    rms_jump = abs(rms_pre - rms_post) / max(rms_pre + rms_post, 1e-6)
    # 3. Spectral centroid shift -> filter_fade
    sc_pre = float(librosa.feature.spectral_centroid(
        y=pre[:, -sr:].mean(axis=0).astype(np.float32), sr=sr).mean())
    sc_post = float(librosa.feature.spectral_centroid(
        y=post[:, :sr].mean(axis=0).astype(np.float32), sr=sr).mean())
    sc_drop = (sc_pre - sc_post) / max(sc_pre, 1.0)
    if sc_drop > 0.4:
        return 'filter_fade'
    # 4. Drum-only window? Energy drop in mid+high but kept in low band
    # Simpler: high RMS jump with sustained transition energy -> drum_break
    # Otherwise: smooth blend = eq_swap or crossfade
    if rms_jump > 0.4:
        return 'cut'
    if rms_jump < 0.15 and sc_drop < 0.1:
        return 'eq_swap'
    return 'crossfade'


def extract_transitions_for_mix(audio_path: str, tracklist: MixTracklist,
                                out_dir: str, mix_id: str,
                                pre_sec: float = 16.0,
                                window_sec: float = 8.0,
                                post_sec: float = 16.0) -> list[dict]:
    """Extract transition windows from one mix. Returns list of metadata dicts."""
    wav, sr = torchaudio.load(audio_path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.size(0) == 1:
        wav = wav.repeat(2, 1)
    elif wav.size(0) > 2:
        wav = wav[:2]
    audio = wav.numpy().astype(np.float32)

    out_root = Path(out_dir) / mix_id
    out_root.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    for idx, trans in enumerate(tracklist.transitions):
        t = trans.at_sec
        pre_start = int(max(0, t - pre_sec - window_sec / 2) * SR)
        pre_end = int(max(0, t - window_sec / 2) * SR)
        trans_start = int(max(0, t - window_sec / 2) * SR)
        trans_end = int(min(audio.shape[1] / SR, t + window_sec / 2) * SR)
        post_start = int(min(audio.shape[1] / SR, t + window_sec / 2) * SR)
        post_end = int(min(audio.shape[1] / SR, t + window_sec / 2 + post_sec) * SR)
        if pre_end - pre_start < SR or trans_end - trans_start < SR:
            print(f"  skip transition {idx}: too close to mix edge")
            continue
        pre = audio[:, pre_start:pre_end]
        trans_aud = audio[:, trans_start:trans_end]
        post = audio[:, post_start:post_end]

        # Save audio per region (16-bit wav for size)
        pre_path = out_root / f"{idx:03d}_pre.wav"
        trans_path = out_root / f"{idx:03d}_transition.wav"
        post_path = out_root / f"{idx:03d}_post.wav"
        torchaudio.save(str(pre_path), torch.from_numpy(pre), SR, encoding='PCM_S', bits_per_sample=16)
        torchaudio.save(str(trans_path), torch.from_numpy(trans_aud), SR, encoding='PCM_S', bits_per_sample=16)
        torchaudio.save(str(post_path), torch.from_numpy(post), SR, encoding='PCM_S', bits_per_sample=16)

        # Features
        pre_mono = pre.mean(axis=0)
        post_mono = post.mean(axis=0)
        tempo_pre = _detect_tempo_window(pre_mono, SR)
        tempo_post = _detect_tempo_window(post_mono, SR)
        tech_guess = _classify_transition_heuristic(pre, trans_aud, post, SR)

        # CLAP embeddings (pre + post)
        try:
            import librosa
            pre48 = librosa.resample(pre_mono, orig_sr=SR, target_sr=48000)
            post48 = librosa.resample(post_mono, orig_sr=SR, target_sr=48000)
            clap_pre = get_audio_embedding(pre48)[0].astype(np.float32)
            clap_post = get_audio_embedding(post48)[0].astype(np.float32)
        except Exception as e:
            print(f"  CLAP failed for transition {idx}: {e}")
            clap_pre = np.zeros(512, dtype=np.float32)
            clap_post = np.zeros(512, dtype=np.float32)

        # Save feature npz
        feat_path = out_root / f"{idx:03d}_features.npz"
        np.savez_compressed(str(feat_path),
                            clap_pre=clap_pre, clap_post=clap_post,
                            tempo_pre=tempo_pre, tempo_post=tempo_post)

        sample = TransitionSample(
            mix_id=mix_id, idx=idx, at_sec=t,
            pre_path=str(pre_path),
            transition_path=str(trans_path),
            post_path=str(post_path),
            tempo_pre=tempo_pre, tempo_post=tempo_post,
            technique_guess=tech_guess,
        )
        samples.append(asdict(sample))
        print(f"  extracted [{idx}] @ {t:.1f}s, tempo {tempo_pre:.0f}->{tempo_post:.0f}, "
              f"guess={tech_guess}")

    # Index file per mix
    with open(out_root / 'index.json', 'w') as f:
        json.dump(samples, f, indent=2)
    return samples


def build(tracklists_dir: str, raw_dir: str, dataset_dir: str,
          download: bool = True) -> None:
    tls = load_all(tracklists_dir)
    if not tls:
        print(f"no tracklists found in {tracklists_dir}")
        return
    print(f"loaded {len(tls)} tracklists")
    all_samples: list[dict] = []
    for tl in tls:
        # Mix id = sanitized title, fallback to URL hash
        mix_id = ''.join(c if c.isalnum() else '_' for c in tl.title)[:80] or 'unknown'
        # Download
        audio_path = None
        if download and tl.url:
            try:
                audio_path = download_audio(tl.url, raw_dir)
            except Exception as e:
                print(f"download failed for {mix_id}: {e}")
                continue
        if audio_path is None:
            print(f"skip {mix_id}: no audio")
            continue
        try:
            samples = extract_transitions_for_mix(
                str(audio_path), tl, dataset_dir, mix_id,
            )
            all_samples.extend(samples)
        except Exception as e:
            print(f"extraction failed for {mix_id}: {e}")

    # Master index across mixes
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(dataset_dir) / 'master_index.json', 'w') as f:
        json.dump(all_samples, f, indent=2)
    print(f"\n=== built dataset: {len(all_samples)} transition samples across "
          f"{len(tls)} mixes ===")
    print(f"=== output: {dataset_dir}/master_index.json ===")
    # Class distribution
    import collections
    techs = collections.Counter(s['technique_guess'] for s in all_samples)
    for tech, count in techs.most_common():
        print(f"  {tech:20s} {count}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracklists', default='datasets/tracklists')
    ap.add_argument('--raw', default='datasets/raw_mixes')
    ap.add_argument('--out', default='datasets/transitions_real')
    ap.add_argument('--no_download', action='store_true',
                    help='use already-downloaded audio (mix_id must exist in --raw)')
    args = ap.parse_args()
    build(args.tracklists, args.raw, args.out, download=not args.no_download)
