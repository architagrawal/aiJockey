"""Recover missing library clip audio by re-downloading from YouTube.

Cached clip JSONs reference audio at paths like:
    /workspace/aiJockey/clips_trimmed/<clip_id>.wav

When stems/audio were purged but metadata kept, this script re-pulls the
audio using the YouTube ID embedded in the filename (last 11-char token
after `__`), trims to the cached duration, and writes WAV at 44.1 kHz.

Filename pattern (from original ingest):
    <genre>__<title>__<11-char-youtube-id>

Idempotent: skips clips whose audio already exists at the target path.
Logs per-clip success/failure.

Usage:
    /opt/venv/bin/python scripts/recover_clip_audio.py --cache /cache
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _extract_yt_id(stem: str) -> str | None:
    # Tail token after last '__'
    if "__" not in stem:
        return None
    tail = stem.rsplit("__", 1)[-1].strip()
    if YT_ID_RE.match(tail):
        return tail
    return None


def _yt_dlp(yt_id: str, work_dir: Path) -> Path | None:
    """Download best audio for a YouTube ID into work_dir, return path."""
    out_tmpl = str(work_dir / f"{yt_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestaudio[ext=m4a]/bestaudio/best",
        "-x", "--audio-format", "wav", "--audio-quality", "0",
        "--no-playlist",
        "--quiet", "--no-warnings",
        "-o", out_tmpl,
        f"https://www.youtube.com/watch?v={yt_id}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    wavs = list(work_dir.glob(f"{yt_id}.wav"))
    if wavs:
        return wavs[0]
    # fallback: any audio
    for ext in ("m4a", "webm", "opus", "mp3"):
        cands = list(work_dir.glob(f"{yt_id}.{ext}"))
        if cands:
            return cands[0]
    return None


def _trim_to_wav(src: Path, dst: Path, duration_s: float, sr: int = 44100) -> bool:
    """Re-encode src to 44.1k stereo WAV of length `duration_s` at dst."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-t", f"{duration_s:.3f}",
        "-ar", str(sr), "-ac", "2",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return proc.returncode == 0 and dst.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/cache")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.exists():
        sys.exit(f"cache dir missing: {cache}")
    jsons = sorted(cache.glob("*.json"))
    if args.limit:
        jsons = jsons[: args.limit]

    summary = {"total": 0, "skip_exists": 0, "skip_no_yt": 0,
                "downloaded": 0, "dl_fail": 0, "trim_fail": 0}
    t_all = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="aijockey_recover_") as wd:
        wd = Path(wd)
        for jp in jsons:
            if jp.name.endswith(".audiobox_slices.json"):
                continue
            summary["total"] += 1
            try:
                meta = json.loads(jp.read_text())
            except Exception:
                continue
            target = meta.get("path")
            duration = float(meta.get("duration") or 0.0)
            if not target:
                continue
            target_p = Path(target)
            if target_p.exists():
                summary["skip_exists"] += 1
                continue
            stem = jp.stem
            yt_id = _extract_yt_id(stem)
            if not yt_id:
                summary["skip_no_yt"] += 1
                print(f"[skip-no-yt] {stem}")
                continue
            if args.dry_run:
                print(f"[dry] would recover {stem} (yt={yt_id}, dur={duration:.1f}s)")
                summary["downloaded"] += 1
                continue
            target_p.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            dl = _yt_dlp(yt_id, wd)
            if not dl:
                summary["dl_fail"] += 1
                print(f"[dl-fail] {stem} (yt={yt_id})")
                continue
            ok = _trim_to_wav(dl, target_p,
                               duration if duration > 0 else 90.0)
            try:
                dl.unlink()
            except Exception:
                pass
            if ok:
                summary["downloaded"] += 1
                dt = time.perf_counter() - t0
                print(f"[ok] {stem} ({dt:.1f}s)")
            else:
                summary["trim_fail"] += 1
                print(f"[trim-fail] {stem}")

    total = time.perf_counter() - t_all
    print(f"\n[recover] done in {total/60:.1f} min: {summary}")


if __name__ == "__main__":
    main()
