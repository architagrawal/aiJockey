"""Scrape long-form DJ mix sets for Tier 2/3 training.

Used by:
- Path C: real-vs-splice discriminator (positive distribution)
- Path B: continuation LM training corpus
- Tier 3: future MusicGen fine-tune

Stored in datasets/dj_sets/<artist_slug>__<id>.wav
Per-set duration: 30-120 min. Audio only.

NOT pushed to git (huge). Lives on laptop, optionally synced to droplet later.
"""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

FFMPEG_DIR = r"C:\Users\msi-laptop\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

OUT = Path(__file__).resolve().parent.parent / "datasets" / "dj_sets"
OUT.mkdir(parents=True, exist_ok=True)

# Curated long-form mixes spanning genres in our clip pool
SETS = [
    # Techno / progressive
    ("boris_brejcha_tomorrowland",     "Boris Brejcha Tomorrowland 2019 set"),
    ("charlotte_de_witte_awakenings",  "Charlotte de Witte Awakenings 2022"),
    ("adam_beyer_drumcode",            "Adam Beyer Drumcode 1 hour mix"),
    ("amelie_lens_set",                "Amelie Lens 1 hour techno mix"),
    # Progressive / melodic
    ("anyma_afterlife",                "Anyma Afterlife 1 hour mix"),
    ("eric_prydz_holosphere",          "Eric Prydz Holosphere set"),
    ("deadmau5_live_set",              "deadmau5 live set 1 hour"),
    # House
    ("fisher_set",                     "Fisher house mix 1 hour"),
    ("camelphat_set",                  "Camelphat 1 hour mix"),
    # Trance
    ("above_beyond_set",               "Above and Beyond 1 hour trance set"),
    ("armin_van_buuren_asot",          "Armin van Buuren ASOT 1 hour"),
    # Future bass / dubstep
    ("flume_set",                      "Flume 1 hour mix"),
    ("skrillex_set",                   "Skrillex 1 hour mix"),
    ("excision_set",                   "Excision 1 hour mix"),
    # DnB
    ("pendulum_set",                   "Pendulum 1 hour DnB set"),
    ("netsky_set",                     "Netsky 1 hour DnB mix"),
    ("andy_c_set",                     "Andy C 1 hour DnB set"),
    # Hip-hop / trap
    ("travis_scott_radio",             "Travis Scott Cactus Jack radio mix 1 hour"),
    ("metro_boomin_set",               "Metro Boomin 1 hour mix"),
    # Bollywood / Indian
    ("nucleya_set",                    "Nucleya 1 hour mix"),
    ("dj_chetas_bollywood",            "DJ Chetas Bollywood mashup 1 hour"),
    ("bollywood_party_mix",            "Bollywood party mix 1 hour 2024"),
    # Synthwave / retrowave
    ("synthwave_mix_neon",             "Synthwave 80s neon mix 1 hour"),
    ("retrowave_outrun",               "Retrowave outrun mix 1 hour"),
    # Lo-fi / chill
    ("lofi_hip_hop_mix",               "Lofi hip hop 1 hour mix"),
    ("chillstep_mix",                  "Chillstep 1 hour mix"),
    # Dubstep / classics
    ("dubstep_classic_mix",            "Classic dubstep 1 hour mix"),
    # Multi-genre festival
    ("tomorrowland_set",               "Tomorrowland 2023 main stage 1 hour"),
    ("ultra_festival_mix",             "Ultra Festival 2023 1 hour mix"),
    ("edc_main_stage",                 "EDC Las Vegas main stage 2023 1 hour"),
]


def run(slug, query):
    out = OUT / f"{slug}__%(id)s.%(ext)s"
    print(f"\n=== {slug} / {query} ===", flush=True)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--default-search", "ytsearch1",
        "-x", "--audio-format", "wav",
        "--no-playlist",
        # 30-120 min sets
        "--match-filter", "duration>1500 & duration<7800",
        "-o", str(out),
        f"ytsearch1:{query}",
    ]
    try:
        subprocess.run(cmd, check=False, timeout=1800)  # 30 min cap per dl
    except subprocess.TimeoutExpired:
        print(f"timeout: {query}", flush=True)


if __name__ == "__main__":
    for slug, q in SETS:
        run(slug, q)
    print("\nDone.", OUT)
    print("Files:", len(list(OUT.glob("*.wav"))))
    total_mb = sum(p.stat().st_size for p in OUT.glob("*.wav")) / (1024 * 1024)
    print(f"Total: {total_mb:.0f} MB")
