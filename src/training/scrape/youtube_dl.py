"""
yt-dlp wrapper. Downloads audio of public DJ mixes for personal research use.

USE FOR PERSONAL/RESEARCH ONLY. Do not redistribute downloaded audio.
Models trained on this data are AGPL-3.0 — forks must remain open.
"""
from __future__ import annotations
from pathlib import Path
import subprocess
import json


def download_audio(url: str, out_dir: str, audio_format: str = 'wav',
                   sample_rate: int = 44100, force: bool = False) -> Path:
    """
    Download audio from a YouTube/SoundCloud URL via yt-dlp.
    Returns the path to the downloaded audio file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info_cmd = ['yt-dlp', '--no-warnings', '--print',
                '%(id)s\t%(title)s\t%(duration)s', url]
    info = subprocess.run(info_cmd, capture_output=True, text=True)
    if info.returncode != 0:
        raise RuntimeError(f"yt-dlp info failed: {info.stderr}")
    vid_id, title, duration_s = info.stdout.strip().split('\t', 2)
    target = out_dir / f"{vid_id}.{audio_format}"
    if target.exists() and not force:
        print(f"[skip] {target.name} already exists")
        return target

    cmd = [
        'yt-dlp', '-x',
        '--audio-format', audio_format,
        '--audio-quality', '0',
        '-o', str(out_dir / '%(id)s.%(ext)s'),
        '--postprocessor-args', f'-ar {sample_rate}',
        url,
    ]
    print(f"[dl] {title} ({duration_s}s) -> {target.name}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {r.stderr}")
    if not target.exists():
        # yt-dlp may have produced different extension; locate by id
        for ext in ('wav', 'mp3', 'flac', 'm4a', 'opus'):
            cand = out_dir / f"{vid_id}.{ext}"
            if cand.exists():
                return cand
        raise RuntimeError(f"download succeeded but no audio file at {target}")
    return target


def get_metadata(url: str) -> dict:
    """Return video metadata as dict (id, title, duration, etc)."""
    cmd = ['yt-dlp', '--dump-single-json', '--no-warnings', url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed: {r.stderr}")
    return json.loads(r.stdout)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('url')
    ap.add_argument('--out', default='datasets/raw_mixes')
    args = ap.parse_args()
    p = download_audio(args.url, args.out)
    print(f"saved: {p}")
