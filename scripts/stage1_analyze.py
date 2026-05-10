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


_ANALYZER_SINGLETON = None


def _get_analyzer():
    """Module-singleton Analyzer so we don't reload Demucs/CLAP per clip."""
    global _ANALYZER_SINGLETON
    if _ANALYZER_SINGLETON is None:
        from analyze import Analyzer
        _ANALYZER_SINGLETON = Analyzer(device=os.environ.get('AI_DEVICE', 'cuda'))
    return _ANALYZER_SINGLETON


def _run_single(analyzer, audio_path: Path, cid: str,
                 precomputed_clap=None) -> bool:
    import json
    import numpy as np
    cache_path = scratch_dir('cache')
    try:
        result = analyzer.analyze(str(audio_path), cid, cache_path,
                                   precomputed_clap=precomputed_clap)
    except TypeError:
        # Older Analyzer.analyze signature without precomputed_clap kwarg.
        result = analyzer.analyze(str(audio_path), cid, cache_path)
    except Exception as e:
        print(f"warn: analyze failed for {audio_path}: {e}")
        return False
    # Analyzer.analyze returns (ClipAnalysis, clap, energy).
    if isinstance(result, tuple) and len(result) == 3:
        ca, clap, energy = result
        with atomic_write(_cache_path(cid)) as f:
            from dataclasses import asdict
            d = asdict(ca) if hasattr(ca, '__dataclass_fields__') else \
                (ca.to_dict() if hasattr(ca, 'to_dict') else ca)
            d['clap_embedding'] = list(map(float, clap.tolist()))
            d['audio_path'] = str(audio_path)
            json.dump(d, f, indent=2, default=str)
        np.savez_compressed(str(cache_path / f'{cid}.npz'),
                             clap=clap, energy=energy)
    return True


def process_one(audio_path: Path) -> bool:
    """Run analyze pipeline on one file. Returns True on success."""
    cid = _clip_id(audio_path)
    if _is_done(audio_path):
        return True
    try:
        analyzer = _get_analyzer()
    except Exception as e:
        print(f"err: cannot init Analyzer ({e})")
        return False
    print(f"S1 analyze {audio_path.name} -> {cid}")
    return _run_single(analyzer, audio_path, cid)


def _batch_clap(audio_paths: list[Path]):
    """Pre-load 48 kHz mono audio for each clip, run a single batched
    CLAP forward, return list of 512-d embeddings (None entries for
    paths that failed to load).
    """
    import numpy as np
    try:
        import librosa
        from clap_wrapper import get_audio_embedding_batch
    except ImportError:
        return [None] * len(audio_paths)
    audios: list = []
    valid_idx: list[int] = []
    for i, p in enumerate(audio_paths):
        try:
            wav, _sr = librosa.load(str(p), sr=48000, mono=True)
            audios.append(wav.astype(np.float32))
            valid_idx.append(i)
        except Exception as e:
            print(f"warn: clap preload {p} failed ({e})")
    out: list = [None] * len(audio_paths)
    if not audios:
        return out
    try:
        embs = get_audio_embedding_batch(audios)
    except Exception as e:
        print(f"warn: batched CLAP failed ({e}); falling back to per-clip")
        return [None] * len(audio_paths)
    for row, orig_i in enumerate(valid_idx):
        out[orig_i] = embs[row]
    return out


def process_batch(audio_paths: list[Path]) -> int:
    """Batched entry point.

    Strategy: CLAP is the only HF-model step that can be safely
    cross-clip batched (variable-length OK via processor padding).
    Demucs/madmom/librosa beats remain per-clip — they have intrinsic
    variable-length pipelines and small batches don't help on GPU.
    Pre-batching CLAP eliminates the per-clip CLAP forward (~1-2s) and
    is the largest single win for stage1 throughput.

    Disable with AIJOCKEY_BATCH_CLAP=0 to fall back to per-clip path.
    """
    if not audio_paths:
        return 0
    try:
        analyzer = _get_analyzer()
    except Exception as e:
        print(f"err: cannot init Analyzer ({e})")
        return 0

    # Skip already-done clips up front so CLAP batch matches actual work.
    work: list[Path] = [p for p in audio_paths if not _is_done(p)]
    if not work:
        return len(audio_paths)
    cids = [_clip_id(p) for p in work]

    use_batch = os.environ.get('AIJOCKEY_BATCH_CLAP', '1') != '0'
    claps = _batch_clap(work) if use_batch else [None] * len(work)

    ok = 0
    for p, cid, c in zip(work, cids, claps):
        print(f"S1 analyze {p.name} -> {cid}")
        if _run_single(analyzer, p, cid, precomputed_clap=c):
            ok += 1
    return ok


def watch_loop(raw_root: Path, interval: float = 30.0,
               extensions: tuple[str, ...] = ('.mp3', '.wav', '.flac', '.m4a', '.ogg'),
               batch_size: int = 16,
               ) -> None:
    print(f"S1 watching {raw_root}, batch_size={batch_size}, every {interval}s")
    while True:
        pending: list[Path] = []
        if raw_root.exists():
            for fp in sorted(raw_root.rglob('*')):
                if not fp.is_file():
                    continue
                if fp.suffix.lower() not in extensions:
                    continue
                if _is_done(fp):
                    continue
                pending.append(fp)
                if len(pending) >= batch_size:
                    break
        if pending:
            process_batch(pending)
        else:
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
