"""Hindustani classical instrumental (no vocals). Allow longer durations."""
import subprocess, sys, os
from pathlib import Path

FFMPEG_DIR = r"C:\Users\msi-laptop\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

OUT = Path(__file__).resolve().parent.parent / "clips"
OUT.mkdir(exist_ok=True)

TRACKS = [
    ("hindustani_sitar", "Ravi Shankar Raga Yaman sitar instrumental"),
    ("hindustani_sitar", "Nikhil Banerjee sitar Raga Bageshri"),
    ("hindustani_bansuri", "Hariprasad Chaurasia bansuri Raga Bhairavi instrumental"),
    ("hindustani_bansuri", "Pandit Hariprasad Chaurasia flute Raga Hansadhwani"),
    ("hindustani_sarod", "Amjad Ali Khan sarod Raga Kafi"),
    ("hindustani_santoor", "Shivkumar Sharma santoor Raga Pahadi"),
    ("hindustani_tabla", "Zakir Hussain tabla solo Teentaal"),
    ("hindustani_tabla", "Ustad Alla Rakha tabla solo"),
    ("hindustani_violin", "Dr L Subramaniam violin classical instrumental"),
    ("hindustani_sarangi", "Ustad Sultan Khan sarangi instrumental"),
    ("indian_instrumental", "Niladri Kumar Zitar fusion instrumental"),
    ("indian_instrumental", "Rakesh Chaurasia bansuri instrumental"),
]

def run(genre, title):
    out_template = str(OUT / f"{genre}__%(title).80B__%(id)s.%(ext)s")
    print(f"\n=== {genre} / {title} ===", flush=True)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--default-search", "ytsearch1",
        "-x", "--audio-format", "wav",
        "--no-playlist",
        "--match-filter", "duration<1500",  # allow up to 25 min
        "-o", out_template,
        f"ytsearch1:{title}",
    ]
    try:
        subprocess.run(cmd, check=False, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"timeout: {title}", flush=True)

for g, t in TRACKS:
    run(g, t)

print("\nDone.", OUT)
