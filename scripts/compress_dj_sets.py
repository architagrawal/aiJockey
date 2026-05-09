"""Convert dj_sets/*.wav -> dj_sets_mp3/*.mp3 (192 kbps) for fast upload to droplet."""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

FFMPEG = r"C:\Users\msi-laptop\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "dj_sets"
DST = ROOT / "datasets" / "dj_sets_mp3"
DST.mkdir(parents=True, exist_ok=True)


def convert(wav: Path) -> tuple[Path, bool, str]:
    out = DST / f"{wav.stem}.mp3"
    if out.exists():
        return out, True, "skip-existing"
    try:
        r = subprocess.run(
            [FFMPEG, "-y", "-i", str(wav), "-codec:a", "libmp3lame",
             "-b:a", "192k", "-ar", "44100", str(out)],
            capture_output=True, timeout=600,
        )
        return out, r.returncode == 0, r.stderr.decode("utf-8", "ignore")[-300:]
    except Exception as e:
        return out, False, str(e)


def main() -> None:
    wavs = sorted(SRC.glob("*.wav"))
    if not wavs:
        print(f"no wavs in {SRC}")
        return
    print(f"converting {len(wavs)} wavs in parallel (4 threads)...")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(convert, w): w for w in wavs}
        for fut in as_completed(futs):
            out, ok, msg = fut.result()
            print(f"  {'OK' if ok else 'FAIL'} {out.name}  {msg if not ok else ''}")
    total_mb = sum(p.stat().st_size for p in DST.glob("*.mp3")) / (1024 * 1024)
    print(f"\nTotal mp3 size: {total_mb:.0f} MB ({len(list(DST.glob('*.mp3')))} files)")


if __name__ == "__main__":
    main()
