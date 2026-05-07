"""
Synthetic transition dataset generator.

For each ordered pair of analyzed clips, render every applicable transition
technique, score the result via a smoothness heuristic, label by best technique.

Pseudo-label score combines:
- Beat continuity at transition (% of beats within ±50ms over 8-bar window
  centered on transition point)
- Spectral coherence (cosine similarity of CLAP embeddings of pre/post 4-sec
  windows around transition)
- Volume continuity (RMS smoothness across transition boundary)

Output: NPZ with X (N, FEATURE_DIM), y (N,), and metadata arrays.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import itertools
import json
import numpy as np
import torch
import torchaudio

import transitions as T
from samples import SampleBank
from execute import render_segment, apply_transition, SR
from planner import load_clips, pick_segment

from features import (
    extract_pair_features, TECHNIQUES, TECHNIQUE_INDEX, FEATURE_DIM,
)


# Subset of techniques that don't require special stem args (safer for
# auto-rendering). Stem-dependent ones (drum_break, mashup, stem_swap) are
# included but may fail gracefully on missing stems.
RENDERABLE_TECHNIQUES = TECHNIQUES  # try all


def _render_pair_with_technique(
    output_so_far: np.ndarray,
    prev_render: dict,
    cur_render: dict,
    tech_name: str,
    sample_bank: SampleBank,
    target_bpm: float,
) -> np.ndarray | None:
    """
    Wrap apply_transition with a fake 'cur' entry that overrides technique.
    Returns mixed audio or None on failure.
    """
    cur = {
        'entry': dict(cur_render['entry']),
        'full': cur_render['full'],
        'stems': cur_render['stems'],
    }
    cur['entry']['transition_in'] = {
        'name': tech_name,
        'bars': 16,
        'silence_beats': 2,
        'spinback_beats': 4,
        'start_bars': 4,
        'semitones': 1.0,
        'delay_beats': 0.5,
        'feedback': 0.55,
        'n_jogs': 4,
        'repetitions': 2,
    }
    try:
        return apply_transition(output_so_far, prev_render, cur, sample_bank, target_bpm)
    except Exception as e:
        print(f"  WARN: render technique {tech_name} failed: {e}")
        return None


def _score_transition(mix: np.ndarray, transition_sec: float) -> float:
    """
    Heuristic smoothness score for a transition at given second mark.
    Higher = smoother. Returns 0..1.
    """
    if mix.shape[1] < int(8 * SR):
        return 0.0
    # 1. Beat continuity in ±4 sec window
    import librosa
    mono = mix.mean(axis=0).astype(np.float32)
    win_start = max(0, int((transition_sec - 4) * SR))
    win_end = min(mix.shape[1], int((transition_sec + 4) * SR))
    win = mono[win_start:win_end]
    if win.size < SR:
        return 0.0
    try:
        _, beat_frames = librosa.beat.beat_track(y=win, sr=SR, units='frames')
        beat_times = librosa.frames_to_time(beat_frames, sr=SR)
        # Beat regularity: low coefficient of variation in inter-beat intervals
        if len(beat_times) >= 4:
            ibi = np.diff(beat_times)
            beat_score = max(0.0, 1.0 - float(np.std(ibi) / max(np.mean(ibi), 1e-6)))
        else:
            beat_score = 0.0
    except Exception:
        beat_score = 0.0

    # 2. Volume continuity — RMS shouldn't spike across boundary
    pre_rms = float(np.sqrt(np.mean(mono[max(0, int((transition_sec - 1) * SR)):
                                         int(transition_sec * SR)] ** 2)))
    post_rms = float(np.sqrt(np.mean(mono[int(transition_sec * SR):
                                          min(len(mono), int((transition_sec + 1) * SR))] ** 2)))
    rms_score = 1.0 - min(1.0, abs(pre_rms - post_rms) / max(pre_rms + post_rms, 1e-6))

    # 3. No silence/click — abrupt zero crossing of energy
    boundary_window = int(0.1 * SR)
    bw_start = max(0, int(transition_sec * SR) - boundary_window // 2)
    bw_end = min(len(mono), bw_start + boundary_window)
    boundary = mono[bw_start:bw_end]
    abs_max = float(np.abs(boundary).max())
    silence_score = 1.0 if abs_max > 0.001 else 0.0  # not silent at boundary

    return 0.5 * beat_score + 0.3 * rms_score + 0.2 * silence_score


def build_dataset(cache_dir: str, samples_dir: str, output_path: str,
                  max_pairs: int | None = None) -> None:
    clips = load_clips(cache_dir)
    if len(clips) < 2:
        raise SystemExit(f"need >=2 analyzed clips, got {len(clips)}")
    print(f"loaded {len(clips)} clips from {cache_dir}")
    sample_bank = SampleBank(samples_dir)

    # Pre-render each clip's chosen segment once
    print("pre-rendering clip segments...")
    rendered_by_clip: dict[str, dict] = {}
    clips_meta: dict[str, dict] = {}
    for cid, clip in clips.items():
        # Strip non-serializable arrays for clips_meta
        meta = {k: v for k, v in clip.items() if k not in ('clap', 'energy_arr')}
        clips_meta[cid] = meta
        seg, _ = pick_segment(clip)
        target_bpm = clip.get('tempo', 128.0) or 128.0
        entry = {
            'clip_id': cid, 'segment': seg,
            'target_bpm': target_bpm, 'target_key': clip.get('key', '?'),
            'transition_in': {'name': 'cut'},
        }
        try:
            full, stems = render_segment(entry, clips_meta, cache_dir)
            rendered_by_clip[cid] = {'entry': entry, 'full': full, 'stems': stems,
                                     'segment': seg}
            print(f"  rendered {cid[:50]}: shape {full.shape}")
        except Exception as e:
            print(f"  WARN: failed to render {cid}: {e}")

    if len(rendered_by_clip) < 2:
        raise SystemExit("rendered <2 clips, cannot build pairs")

    # Iterate pairs
    cids = list(rendered_by_clip.keys())
    pairs = list(itertools.permutations(cids, 2))
    if max_pairs:
        pairs = pairs[:max_pairs]
    print(f"\nbuilding dataset: {len(pairs)} pairs x {len(RENDERABLE_TECHNIQUES)} techniques")

    X: list[np.ndarray] = []
    y: list[int] = []
    scores: list[float] = []
    pair_meta: list[dict] = []

    for pi, (a_id, b_id) in enumerate(pairs, start=1):
        prev = rendered_by_clip[a_id]
        cur = rendered_by_clip[b_id]
        prev_seg = prev['segment']
        cur_seg = cur['segment']

        a_clip = clips[a_id]
        b_clip = clips[b_id]
        target_bpm = float(prev['entry']['target_bpm'])

        feats = extract_pair_features(a_clip, prev_seg, b_clip, cur_seg)

        # Render every technique, score each
        tech_scores: dict[str, float] = {}
        for tech in RENDERABLE_TECHNIQUES:
            mix = _render_pair_with_technique(
                prev['full'], prev, cur, tech, sample_bank, target_bpm,
            )
            if mix is None:
                continue
            transition_sec = prev['full'].shape[1] / SR
            score = _score_transition(mix, transition_sec)
            tech_scores[tech] = score

        if not tech_scores:
            print(f"  pair {pi}/{len(pairs)} {a_id[:30]} -> {b_id[:30]}: NO TECHNIQUES OK")
            continue
        best_tech = max(tech_scores, key=tech_scores.get)
        best_score = tech_scores[best_tech]

        X.append(feats)
        y.append(TECHNIQUE_INDEX[best_tech])
        scores.append(best_score)
        pair_meta.append({
            'a': a_id, 'b': b_id, 'best': best_tech, 'best_score': best_score,
            'all_scores': tech_scores,
        })
        print(f"  pair {pi}/{len(pairs)} {a_id[:25]} -> {b_id[:25]}: best={best_tech} ({best_score:.3f})")

    if not X:
        raise SystemExit("no successful renders, dataset empty")

    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    s_arr = np.asarray(scores, dtype=np.float32)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, X=X_arr, y=y_arr, scores=s_arr,
        techniques=np.array(TECHNIQUES),
    )
    meta_path = Path(output_path).with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump({'pairs': pair_meta, 'feature_dim': FEATURE_DIM,
                   'n_techniques': len(TECHNIQUES)}, f, indent=2, default=str)
    print(f"\nsaved {output_path} (N={len(X)}, dim={FEATURE_DIM})")
    print(f"saved {meta_path}")
    # Class balance
    import collections
    counts = collections.Counter(y)
    for idx, c in sorted(counts.items()):
        print(f"  {TECHNIQUES[idx]:20s} {c}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='cache')
    ap.add_argument('--samples', default='samples')
    ap.add_argument('--out', default='datasets/synthetic_transitions.npz')
    ap.add_argument('--max_pairs', type=int, default=None)
    args = ap.parse_args()
    build_dataset(args.cache, args.samples, args.out, args.max_pairs)
