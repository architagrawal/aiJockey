"""Indian music gap-filler: classical, Bollywood, fusion, Punjabi."""
import subprocess, sys, os
from pathlib import Path

FFMPEG_DIR = r"C:\Users\msi-laptop\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

OUT = Path(__file__).resolve().parent.parent / "clips"
OUT.mkdir(exist_ok=True)

TRACKS = [
    ("hindustani", "Ravi Shankar Raga Jog"),
    ("hindustani", "Hariprasad Chaurasia bansuri"),
    ("carnatic", "Bombay Jayashri Vatapi Ganapatim"),
    ("indian_fusion", "Anoushka Shankar Burn"),
    ("indian_fusion", "Niladri Kumar Zitar"),
    ("bollywood", "Pritam Kalank title track"),
    ("bollywood", "AR Rahman Jai Ho"),
    ("bollywood", "Vishal Shekhar Senorita Zindagi Na Milegi Dobara"),
    ("punjabi", "Diljit Dosanjh GOAT"),
    ("punjabi", "Sidhu Moose Wala 295"),
    ("indian_edm", "Nucleya Bass Rani Laung Laachi"),
    ("sufi", "Nusrat Fateh Ali Khan Tumhe Dillagi"),
    ("indian_classical_fusion", "Shakti John McLaughlin Joy"),
    ("ghazal", "Jagjit Singh Hothon Se Chhulo Tum"),
]

def run(genre, title):
    out_template = str(OUT / f"{genre}__%(title).80B__%(id)s.%(ext)s")
    print(f"\n=== {genre} / {title} ===", flush=True)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--default-search", "ytsearch1",
        "-x", "--audio-format", "wav",
        "--no-playlist",
        "--match-filter", "duration<480",
        "-o", out_template,
        f"ytsearch1:{title}",
    ]
    try:
        subprocess.run(cmd, check=False, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"timeout: {title}", flush=True)

for g, t in TRACKS:
    run(g, t)

print("\nDone.", OUT)
