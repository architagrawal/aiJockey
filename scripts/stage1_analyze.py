"""S1 — analyze stage.

Watch /scratch/raw/ for new audio. Run Demucs stems + beats + key + CLAP +
phrase-length detection. Write per-clip cache to /scratch/cache/{id}.json
plus stems to /scratch/cache/stems/{id}/.

Idempotent: skips clips with existing cache + matching mtime.

Performance hooks (from src/training/efficiency.py):
    AIJOCKEY_DTYPE=bfloat16     mixed precision
    AIJOCKEY_COMPILE=1          torch.compile on hot paths
    AIJOCKEY_BATCH_SIZE=4       Demucs/CLAP batch size

Usage:
    python scripts/stage1_analyze.py --watch /scratch/raw --workers 4
"""
from __future__ import annotations
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pipeline.common import scratch_dir, watch, atomic_write


def _clip_id(path: Path) -> str:
    """Deterministic clip ID from absolute path."""
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]


def _cache_path(clip_id: str) -> Path:
    return scratch_dir('cache') / f'{clip_id}.json'


def _stems_dir(clip_id: str) -> Path:
    return scratch_dir('cache', 'stems', clip_id)


def _is_done(audio_path: Path) -> bool:
    cid = _clip_id(audio_path)
    cp = _cache_path(cid)
    return cp.exists() and cp.stat().st_mtime >= audio_path.stat().st_mtime


def process_one(audio_path: Path) -> bool:
    """Run analyze pipeline on one file. Returns True on success."""
    cid = _clip_id(audio_path)
    if _is_done(audio_path):
        return True
    try:
        # Lazy import — heavy deps only loaded once worker actually runs
        from analyze import Analyzer
    except ImportError as e:
        print(f"err: cannot import analyze ({e})")
        return False
    print(f"S1 analyze {audio_path.name} -> {cid}")
    try:
        analyzer = Analyzer(stems_dir=str(scratch_dir('cache', 'stems')))
        feat = analyzer.analyze(str(audio_path), clip_id=cid)
    except Exception as e:
        print(f"warn: analyze failed for {audio_path}: {e}")
        return False
    with atomic_write(_cache_path(cid)) as f:
        import json
        json.dump(feat.to_dict() if hasattr(feat, 'to_dict') else feat, f,
                  indent=2, default=str)
    return True


def watch_loop(raw_root: Path, interval: float = 30.0,
               extensions: tuple[str, ...] = ('.mp3', '.wav', '.flac', '.m4a', '.ogg')
               ) -> None:
    print(f"S1 watching {raw_root} for new audio every {interval}s")
    while True:
        seen_any = False
        if raw_root.exists():
            for fp in sorted(raw_root.rglob('*')):
                if not fp.is_file():
                    continue
                if fp.suffix.lower() not in extensions:
                    continue
                if _is_done(fp):
                    continue
                process_one(fp)
                seen_any = True
        if not seen_any:
            time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', default=str(scratch_dir('raw')))
    ap.add_argument('--interval', type=float, default=30.0)
    ap.add_argument('--workers', type=int, default=1,
                    help='intra-process worker count (defer to multiprocessing)')
    args = ap.parse_args()
    raw_root = Path(args.watch)
    watch_loop(raw_root, interval=args.interval)


if __name__ == '__main__':
    main()
