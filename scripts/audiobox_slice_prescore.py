"""Pre-score every cached library clip's sections with Audiobox Aesthetics.

Writes sidecar `<clip_id>.audiobox_slices.json` next to existing
`<clip_id>.json` in the cache directory. Format:

    {
      "audiobox_axes": ["PQ", "PC", "CE", "CU"],
      "sections": [
        {"start": 12.0, "end": 36.0, "PQ": 7.21, "PC": 5.40, ...},
        ...
      ]
    }

The planner / candidate_picker reads this to bias selection toward
high-PQ sections instead of guessing from RMS + tempo alone.

Run once per library refresh. Idempotent: skips clips whose sidecar
already covers all cached sections.

Usage (on MI300X, inside container):
    export AIJOCKEY_AUDIOBOX_AESTHETICS=1
    /opt/venv/bin/python scripts/audiobox_slice_prescore.py \\
        --cache /cache --min-section-seconds 8 --workers 1

Toggle `--force` to recompute. `--limit N` to dry-run on N clips.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Make src/ importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_audio_slice(audio_path: str, start: float, end: float) -> tuple:
    """Load [start, end] of audio_path. Returns (np.ndarray mono, sr)."""
    import librosa
    duration = max(0.0, float(end) - float(start))
    if duration <= 0:
        return None, None
    wav, sr = librosa.load(audio_path, sr=None, mono=True,
                            offset=float(start), duration=duration)
    return wav, sr


def _write_temp_wav(wav, sr) -> str:
    import soundfile as sf
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="aae_slice_")
    os.close(fd)
    sf.write(path, wav, sr)
    return path


def _score_sections_batch(audio_path: str, sections: list[dict]) -> list[dict]:
    """Score every section of one clip with one batched Audiobox call.
    Returns list of {start, end, PQ, ...}."""
    from audiobox_critic import enabled, score_batch
    if not enabled():
        raise RuntimeError(
            "audiobox_critic.enabled() returned False — "
            "set AIJOCKEY_AUDIOBOX_AESTHETICS=1")
    valid: list[tuple[dict, str]] = []  # (section, tmp_path)
    try:
        for sec in sections:
            s = float(sec.get("start", 0.0))
            e = float(sec.get("end", 0.0))
            if e - s < 1.0:
                continue
            wav, sr = _load_audio_slice(audio_path, s, e)
            if wav is None or len(wav) < int(sr or 16000):
                continue
            tmp = _write_temp_wav(wav, sr)
            valid.append(({"start": s, "end": e}, tmp))
        if not valid:
            return []
        paths = [tmp for _, tmp in valid]
        results = score_batch(paths)
    finally:
        for _, tmp in valid:
            try:
                os.unlink(tmp)
            except Exception:
                pass
    out: list[dict] = []
    for (sec, _), scores in zip(valid, results):
        if not scores:
            continue
        out.append({
            "start": round(sec["start"], 3),
            "end": round(sec["end"], 3),
            "PQ": round(float(scores.get("PQ", 0.0)), 3),
            "PC": round(float(scores.get("PC", 0.0)), 3),
            "CE": round(float(scores.get("CE", 0.0)), 3),
            "CU": round(float(scores.get("CU", 0.0)), 3),
        })
    return out


def _process_clip(cache: Path, clip_id: str, *, min_seconds: float,
                  force: bool) -> str:
    json_path = cache / f"{clip_id}.json"
    sidecar = cache / f"{clip_id}.audiobox_slices.json"
    if not json_path.exists():
        return f"[skip] {clip_id} (no metadata)"
    meta = json.loads(json_path.read_text())
    audio_path = meta.get("path")
    if not audio_path or not Path(audio_path).exists():
        return f"[skip] {clip_id} (audio missing: {audio_path})"
    sections = [s for s in (meta.get("sections") or [])
                 if float(s.get("end", 0) - s.get("start", 0)) >= min_seconds]
    if not sections:
        return f"[skip] {clip_id} (no sections ≥{min_seconds}s)"

    if sidecar.exists() and not force:
        prev = json.loads(sidecar.read_text())
        prev_n = len(prev.get("sections") or [])
        if prev_n >= len(sections):
            return f"[skip-cached] {clip_id} ({prev_n} slices)"

    t0 = time.perf_counter()
    scored = _score_sections_batch(audio_path, sections)
    dt = time.perf_counter() - t0
    blob = {
        "audiobox_axes": ["PQ", "PC", "CE", "CU"],
        "sections": scored,
        "ms": int(dt * 1000),
    }
    sidecar.write_text(json.dumps(blob, indent=2))
    return f"[ok] {clip_id} {len(scored)} slices in {dt:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/cache",
                    help="Cache dir holding <clip_id>.json metadata files")
    ap.add_argument("--min-section-seconds", type=float, default=8.0)
    ap.add_argument("--force", action="store_true",
                    help="Recompute sidecars that already exist")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N clips (0 = all)")
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.exists():
        sys.exit(f"cache dir missing: {cache}")
    clips = sorted(p.stem for p in cache.glob("*.json")
                   if not p.name.endswith(".audiobox_slices.json"))
    if args.limit:
        clips = clips[: args.limit]
    print(f"audiobox slice prescore: {len(clips)} clips, cache={cache}")
    t_all = time.perf_counter()
    for i, cid in enumerate(clips):
        msg = _process_clip(cache, cid,
                             min_seconds=args.min_section_seconds,
                             force=args.force)
        print(f"[{i+1}/{len(clips)}] {msg}")
    total = time.perf_counter() - t_all
    print(f"done in {total/60:.1f} min ({total:.0f}s)")


if __name__ == "__main__":
    main()
