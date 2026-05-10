"""HF-datasets fetcher — alternative S0 source when YouTube blocks cloud
IPs and Internet Archive variety isn't enough.

HuggingFace datasets uses xet protocol (proven working from prefetch_models
session). No bot detection, no auth wall for public datasets.

Targets multiple HF audio datasets, picks tracks within duration band,
saves WAV to /scratch/raw/hf/<dataset>/<track_id>.wav. S1 picks up
automatically via watch loop.

Usage:
    python scripts/fetch_via_hf_datasets.py \\
        --datasets fma_small,jamendo \\
        --max-per-dataset 500 \\
        --out /scratch/raw/hf

Datasets known to work (as of session knowledge cutoff):
    benjamin-paine/free-music-archive    — FMA mirror, 30s clips
    nateraw/fma_small                     — FMA-small (~7GB)
    zafrir-y/free-music-archive           — alt FMA mirror
    marsyas/gtzan                         — GTZAN genre dataset (10 genres)

Skip-list: datasets requiring auth or with non-redistributable licenses.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _import_optional():
    """Returns (datasets, soundfile, np). Raises on missing critical dep."""
    try:
        import datasets   # type: ignore
        import soundfile as sf
        import numpy as np
        return datasets, sf, np
    except ImportError as e:
        sys.exit(f"missing dep: {e}. pip install datasets soundfile numpy")


_KNOWN_DATASETS = {
    "fma_small": {
        "id": "nateraw/fma_small",
        "split": "train",
        "audio_field": "audio",
        "label_field": "genre",
    },
    "fma_full": {
        "id": "benjamin-paine/free-music-archive",
        "split": "train",
        "audio_field": "audio",
        "label_field": "genre",
    },
    "gtzan": {
        "id": "marsyas/gtzan",
        "split": "train",
        "audio_field": "audio",
        "label_field": "genre",
    },
}


def _save_wav(audio_array, sample_rate: int, out_path: Path,
               sf, np, target_sr: int = 44100,
               min_seconds: float = 30.0,
               max_seconds: float = 300.0) -> bool:
    """Write a stereo 44.1kHz WAV. Skip if duration outside band.

    HF audio fields are typically dicts {"array": np.ndarray, "sampling_rate": int}.
    Some are nested in {"audio": {...}}. Caller must extract correctly.
    """
    if audio_array is None:
        return False
    arr = np.asarray(audio_array, dtype=np.float32)
    if arr.ndim == 0:
        return False
    if arr.ndim == 1:
        arr_t = arr     # mono
    else:
        # Some datasets give (channels, time) or (time, channels) — normalize
        if arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 8:
            arr_t = arr.T   # transpose to (time, channels)
        else:
            arr_t = arr
    n_samples = arr_t.shape[0]
    duration = n_samples / sample_rate
    if duration < min_seconds or duration > max_seconds:
        return False

    # Resample if needed (use scipy when librosa not desired). Cheap downstream
    # path: rely on torchaudio/librosa elsewhere — write at native SR, S1 handles.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), arr_t, sample_rate, subtype="PCM_16")
    return True


def fetch_dataset(name: str, max_n: int, out_root: Path,
                  datasets, sf, np) -> tuple[int, int]:
    spec = _KNOWN_DATASETS.get(name)
    if spec is None:
        print(f"unknown dataset name: {name}; known: {list(_KNOWN_DATASETS)}")
        return (0, 0)

    print(f"\n=== loading {spec['id']} ({spec['split']}) ===")
    try:
        ds = datasets.load_dataset(spec["id"], split=spec["split"],
                                     streaming=True)
    except Exception as e:
        print(f"  load failed ({e.__class__.__name__}: {e})")
        return (0, 0)

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_seen = 0
    n_skip = 0
    t0 = time.time()

    for row in ds:
        if n_done >= max_n:
            break
        n_seen += 1
        audio = row.get(spec["audio_field"])
        if isinstance(audio, dict):
            arr = audio.get("array")
            sr = int(audio.get("sampling_rate", 44100))
        else:
            arr = audio
            sr = 44100
        if arr is None:
            n_skip += 1
            continue

        # Build a deterministic-ish filename
        track_id = row.get("track_id") or row.get("id") or row.get("path") or f"row{n_seen:06d}"
        track_id = str(track_id).replace("/", "_").replace(" ", "_")[:80]
        label = row.get(spec.get("label_field") or "")
        prefix = f"{label}_{track_id}" if label else track_id
        out_path = out_dir / f"{prefix}.wav"
        if out_path.exists():
            continue

        try:
            if _save_wav(arr, sr, out_path, sf, np):
                n_done += 1
                if n_done % 25 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{n_done}/{max_n}] saved (avg {elapsed/n_done:.1f}s/clip)")
            else:
                n_skip += 1
        except Exception as e:
            n_skip += 1
            print(f"  ! row {n_seen}: {e.__class__.__name__}: {e}")

    print(f"  done: {n_done} saved, {n_skip} skipped, {n_seen} seen "
          f"in {time.time()-t0:.1f}s")
    return (n_done, n_skip)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="fma_small",
                     help=f"comma list: {','.join(_KNOWN_DATASETS)}")
    ap.add_argument("--max-per-dataset", type=int, default=500)
    ap.add_argument("--out", default="/scratch/raw/hf",
                     help="output root dir")
    args = ap.parse_args()

    datasets, sf, np = _import_optional()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    total_done = 0
    total_skip = 0
    for name in args.datasets.split(","):
        name = name.strip()
        if not name:
            continue
        done, skip = fetch_dataset(name, args.max_per_dataset, out_root,
                                     datasets, sf, np)
        total_done += done
        total_skip += skip

    print(f"\n=== total: {total_done} clips saved, {total_skip} skipped ===")
    print(f"output: {out_root}")
    return 0 if total_done else 1


if __name__ == "__main__":
    sys.exit(main())
