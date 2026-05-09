"""Pre-bake 5 wildly different demo mixes from the full clip pool.

Run on droplet after analyze cache is warm. Idempotent.

Steps per mix:
  1. plan with arc + prompt
  2. execute -> raw_mix.wav
  3. master -> final_mix.wav
  4. encode -> mp3 for HF Space upload

Auto-trims long source clips (>120s) to first 90s before analyze.
"""
from __future__ import annotations
import os, sys, subprocess, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CLIPS_DIR = ROOT / "clips"
CLIPS_TRIMMED = ROOT / "clips_trimmed"
CACHE_DIR = ROOT / "cache"
OUT_BASE = ROOT / "output" / "demos"
DEMO_MP3 = ROOT / "demo_mp3"

DEMOS = [
    dict(slug="festival_inferno",       arc="peak",          prompt="festival main stage, euphoric drops, big bass, screaming crowd, anthemic",                          duration=180),
    dict(slug="midnight_noir",          arc="flat_low",      prompt="after-hours noir, smoky melancholy, lo-fi nostalgia, dimly lit, slow burn",                         duration=180),
    dict(slug="neon_retrowave",         arc="rollercoaster", prompt="80s synthwave neon arcade, driving arpeggios, retro nostalgia, vintage warmth",                     duration=180),
    dict(slug="east_meets_bass",        arc="rollercoaster", prompt="sitar and tabla over deep bass, raga-electronic fusion, indian classical meets dubstep",            duration=180),
    dict(slug="bollywood_block_party",  arc="build",         prompt="bollywood club anthem to punjabi drill, dancefloor heat, festival fusion, 808s",                     duration=180),
]

LUFS = -9.0
MAX_CLIP_SEC = 120
TRIM_SEC = 90


def trim_long_clips() -> Path:
    """Copy clips/ to clips_trimmed/, trimming files longer than MAX_CLIP_SEC to TRIM_SEC."""
    CLIPS_TRIMMED.mkdir(exist_ok=True)
    import soundfile as sf
    for src in CLIPS_DIR.iterdir():
        if src.suffix.lower() not in {".wav", ".mp3", ".flac"}:
            continue
        dst = CLIPS_TRIMMED / src.name
        if dst.exists():
            continue
        try:
            info = sf.info(str(src))
            dur = info.frames / info.samplerate
        except Exception as e:
            print(f"  skip {src.name}: {e}")
            continue
        if dur <= MAX_CLIP_SEC:
            shutil.copy2(src, dst)
            continue
        # trim to TRIM_SEC starting at offset to avoid silent intros
        offset = min(15.0, max(0.0, dur * 0.10))
        sr = info.samplerate
        start = int(offset * sr)
        stop = start + int(TRIM_SEC * sr)
        data, _ = sf.read(str(src), start=start, stop=stop, always_2d=False)
        sf.write(str(dst), data, sr)
        print(f"  trimmed {src.name}: {dur:.0f}s -> {TRIM_SEC}s")
    return CLIPS_TRIMMED


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def analyze_pool(clips_dir: Path) -> None:
    run([sys.executable, str(ROOT / "src" / "main.py"),
         "analyze", "--clips", str(clips_dir), "--cache", str(CACHE_DIR)])


def render_mix(slug: str, arc: str, prompt: str, duration: int) -> Path:
    out_dir = OUT_BASE / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    timeline = out_dir / "timeline.json"
    raw = out_dir / "raw_mix.wav"
    final = out_dir / "final_mix.wav"

    run([sys.executable, str(ROOT / "src" / "main.py"),
         "plan", "--cache", str(CACHE_DIR), "--out", str(timeline),
         "--duration", str(duration), "--arc", arc, "--prompt", prompt,
         "--min_unique_clips", "5"])
    run([sys.executable, str(ROOT / "src" / "main.py"),
         "execute", "--timeline", str(timeline),
         "--cache", str(CACHE_DIR), "--out", str(raw)])
    run([sys.executable, str(ROOT / "src" / "main.py"),
         "master", "--in_path", str(raw), "--out", str(final),
         "--lufs", str(LUFS)])
    return final


def encode_mp3(wav_path: Path, slug: str) -> Path:
    DEMO_MP3.mkdir(exist_ok=True)
    mp3_path = DEMO_MP3 / f"{slug}.mp3"
    run(["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame",
         "-b:a", "192k", str(mp3_path)])
    return mp3_path


def main() -> None:
    print("=== trim long clips ===")
    trimmed_dir = trim_long_clips()
    print("=== analyze pool ===")
    analyze_pool(trimmed_dir)
    print("=== render demos ===")
    results = []
    for d in DEMOS:
        try:
            final = render_mix(d["slug"], d["arc"], d["prompt"], d["duration"])
            mp3 = encode_mp3(final, d["slug"])
            results.append((d["slug"], mp3, "ok"))
            print(f"  -> {mp3}")
        except subprocess.CalledProcessError as e:
            results.append((d["slug"], None, f"fail: {e}"))
            print(f"  FAIL {d['slug']}: {e}")
    print("\n=== results ===")
    for slug, path, status in results:
        print(f"  {slug:25} {status} {path or ''}")


if __name__ == "__main__":
    main()
