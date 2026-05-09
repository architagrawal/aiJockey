"""Gap-filling clip downloader for genre variety.

Existing pool covers EDM/house/techno/trance/dnb/dubstep/future_bass/chillstep/trap/hip_hop.
Gaps: synthwave, lo-fi, ambient/cinematic, jazz-house, drill.
"""
import subprocess, sys, os, shutil
from pathlib import Path

FFMPEG_DIR = r"C:\Users\msi-laptop\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

OUT = Path(__file__).resolve().parent.parent / "clips"
OUT.mkdir(exist_ok=True)

TRACKS = [
    ("synthwave", "Kavinsky Nightcall", "https://www.youtube.com/results?search_query=Kavinsky+Nightcall"),
    ("synthwave", "The Midnight Sunset", "https://www.youtube.com/results?search_query=The+Midnight+Sunset"),
    ("synthwave", "FM-84 Running in the Night", "https://www.youtube.com/results?search_query=FM-84+Running+in+the+Night"),
    ("retrowave", "Mitch Murder Then Again", "https://www.youtube.com/results?search_query=Mitch+Murder+Then+Again"),
    ("lofi", "Nujabes Aruarian Dance", "https://www.youtube.com/results?search_query=Nujabes+Aruarian+Dance"),
    ("lofi", "Idealism Sunset Lover", "https://www.youtube.com/results?search_query=Idealism+Sunset+Lover"),
    ("ambient", "Tycho Awake", "https://www.youtube.com/results?search_query=Tycho+Awake"),
    ("ambient", "Hammock Mono No Aware", "https://www.youtube.com/results?search_query=Hammock+Mono+No+Aware"),
    ("cinematic", "Hans Zimmer Time Inception", "https://www.youtube.com/results?search_query=Hans+Zimmer+Time+Inception"),
    ("jazz_house", "Kerri Chandler Bar A Thym", "https://www.youtube.com/results?search_query=Kerri+Chandler+Bar+A+Thym"),
    ("disco", "Chic Le Freak", "https://www.youtube.com/results?search_query=Chic+Le+Freak"),
    ("drill", "Pop Smoke Dior", "https://www.youtube.com/results?search_query=Pop+Smoke+Dior"),
]

def slug(s):
    return "".join(c if c.isalnum() else "_" for c in s)

for genre, title, url in TRACKS:
    out_template = str(OUT / f"{genre}__{slug(title)}__%(id)s.%(ext)s")
    print(f"\n=== {genre} / {title} ===", flush=True)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--default-search", "ytsearch1",
        "-x", "--audio-format", "wav",
        "--no-playlist",
        "--match-filter", "duration<420",
        "-o", out_template,
        f"ytsearch1:{title}",
    ]
    try:
        subprocess.run(cmd, check=False, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"timeout on {title}", flush=True)

print("\nDone. New clips in:", OUT)
