"""S2 — segment DJ-set caches into (pre, transition, post) triplets.

Watch /scratch/cache/ for clips tagged is_dj_set=true. Use novelty curve +
onset jumps to detect transition points. Write triplet metadata to
/scratch/transitions/{set_id}/t{n}.json.

Triplet format:
    {
      "set_id": "...", "n": 3,
      "pre":   {"clip_id": "...", "start": 121.4, "end": 137.4},
      "trans": {"clip_id": "...", "start": 137.4, "end": 153.4},
      "post":  {"clip_id": "...", "start": 153.4, "end": 169.4},
      "tech_label": "filter_fade",   # heuristic
      "downbeats": [...]              # within trans window
    }

Used downstream by S4 (critic v2) + S8 (MusicGen-Small bridge fine-tune).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.common import scratch_dir, atomic_write, watch


TRANS_WINDOW_SEC = 16.0


def _detect_transitions(meta: dict) -> list[float]:
    """Heuristic: large novelty curve peaks separated by >= 30s.

    Returns timestamps (sec) at transition centers. Real production will
    train a transition-detector classifier; this is a strong baseline.
    """
    novelty = meta.get('novelty_curve') or meta.get('energy_curve') or []
    if not novelty:
        return []
    import numpy as np
    arr = np.asarray(novelty, dtype=np.float32)
    if arr.size < 30:
        return []
    hop_hz = float(meta.get('curve_hop_hz', 10.0))
    threshold = float(arr.mean()) + 1.5 * float(arr.std())
    peaks: list[int] = []
    last_peak = -1000
    min_gap = int(30 * hop_hz)
    for i in range(1, arr.size - 1):
        if arr[i] > threshold and arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
            if i - last_peak >= min_gap:
                peaks.append(i)
                last_peak = i
    return [p / hop_hz for p in peaks]


def _technique_label(meta: dict, t_center: float) -> str:
    """Heuristic technique tag from local features. Replaces with classifier later."""
    return 'filter_fade'  # placeholder


def process_set(cache_path: Path, out_root: Path) -> int:
    meta = json.loads(cache_path.read_text())
    if not meta.get('is_dj_set'):
        return 0
    cid = meta.get('clip_id') or cache_path.stem
    out_dir = out_root / cid
    if out_dir.exists() and any(out_dir.glob('t*.json')):
        return 0  # already segmented
    centers = _detect_transitions(meta)
    if not centers:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    half = TRANS_WINDOW_SEC / 2.0
    n_written = 0
    for n, c in enumerate(centers):
        triplet = {
            'set_id': cid,
            'n': n,
            'pre':   {'clip_id': cid, 'start': max(0.0, c - 3 * half), 'end': c - half},
            'trans': {'clip_id': cid, 'start': c - half, 'end': c + half},
            'post':  {'clip_id': cid, 'start': c + half, 'end': c + 3 * half},
            'tech_label': _technique_label(meta, c),
        }
        with atomic_write(out_dir / f't{n:03d}.json') as f:
            json.dump(triplet, f, indent=2)
        n_written += 1
    print(f"S2 segmented {cid}: {n_written} triplets")
    return n_written


def watch_loop(cache_root: Path, out_root: Path,
               interval: float = 60.0) -> None:
    print(f"S2 watching {cache_root} every {interval}s")
    seen: set[str] = set()
    while True:
        if cache_root.exists():
            for fp in sorted(cache_root.glob('*.json')):
                k = str(fp)
                if k in seen:
                    continue
                process_set(fp, out_root)
                seen.add(k)
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', default=str(scratch_dir('cache')))
    ap.add_argument('--out', default=str(scratch_dir('transitions')))
    ap.add_argument('--interval', type=float, default=60.0)
    args = ap.parse_args()
    watch_loop(Path(args.watch), Path(args.out), interval=args.interval)


if __name__ == '__main__':
    main()
