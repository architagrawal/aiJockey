"""Walk cache/*.json + cache/stems/<id>/vocals.wav. Compute vocal RMS per
section, write vocal_activity field back. Run once to upgrade existing cache.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import soundfile as sf


def vocal_activity(vox_path: Path, drums_path: Path | None,
                   bass_path: Path | None, other_path: Path | None,
                   start: float, end: float) -> float:
    if not vox_path.exists() or end <= start:
        return 0.5
    info = sf.info(str(vox_path))
    sr = info.samplerate
    s_f = max(0, int(start * sr))
    e_f = min(info.frames, int(end * sr))
    try:
        vox, _ = sf.read(str(vox_path), start=s_f, stop=e_f, always_2d=False)
    except Exception:
        return 0.5
    if vox.ndim > 1:
        vox = vox.mean(axis=-1)
    vox_rms = float(np.sqrt(np.mean(vox ** 2)) + 1e-8)
    inst_rms = 1e-8
    for p in (drums_path, bass_path, other_path):
        if p is None or not p.exists():
            continue
        try:
            arr, _ = sf.read(str(p), start=s_f, stop=e_f, always_2d=False)
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            inst_rms += float(np.sqrt(np.mean(arr ** 2)))
        except Exception:
            pass
    return vox_rms / (vox_rms + inst_rms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    args = ap.parse_args()
    cache = Path(args.cache)
    if not cache.exists():
        print(f"no cache at {cache}")
        sys.exit(1)
    n_done = n_skip = 0
    for jp in sorted(cache.glob("*.json")):
        cid = jp.stem
        stems = cache / "stems" / cid
        vox = stems / "vocals.wav"
        drums = stems / "drums.wav"
        bass = stems / "bass.wav"
        other = stems / "other.wav"
        if not vox.exists():
            n_skip += 1
            continue
        with open(jp) as f:
            data = json.load(f)
        secs = data.get("sections", [])
        any_changed = False
        for s in secs:
            if "vocal_activity" in s:
                continue
            va = vocal_activity(vox, drums, bass, other,
                                float(s.get("start", 0)),
                                float(s.get("end", 0)))
            s["vocal_activity"] = round(va, 4)
            any_changed = True
        if any_changed:
            with open(jp, "w") as f:
                json.dump(data, f, indent=2)
            n_done += 1
        else:
            n_skip += 1
    print(f"backfilled {n_done} clips; skipped {n_skip}")


if __name__ == "__main__":
    main()
