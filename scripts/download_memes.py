"""Download short meme/SFX one-shots (1-10s) for optional overlay during mixing.

Stored in samples/memes/. Planner can choose to use 0-2 in any mix.
"""
from __future__ import annotations
import subprocess, sys, os
from pathlib import Path

FFMPEG_DIR = r"C:\Users\msi-laptop\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

OUT = Path(__file__).resolve().parent.parent / "samples" / "memes"
OUT.mkdir(parents=True, exist_ok=True)

# Each: (slug, search query). Aim for short SFX/one-shot recordings.
MEMES = [
    # DJ/drop SFX
    ("airhorn",         "MLG air horn sound effect"),
    ("airhorn_2",       "DJ airhorn sample"),
    ("vinyl_rewind",    "vinyl rewind sound effect"),
    ("scratch",         "DJ scratch sound effect short"),
    ("scratch_2",       "turntable scratch sample"),
    ("riser_short",     "riser sound effect short"),
    ("subdrop",         "sub drop bass sound effect"),
    ("impact_boom",     "cinematic boom impact"),
    ("siren",           "MLG siren sound effect"),
    ("snare_roll",      "snare roll buildup short"),

    # Meme classics (short)
    ("bruh",            "bruh sound effect short"),
    ("oof",             "minecraft oof sound effect"),
    ("vine_boom",       "vine boom sound effect"),
    ("damn",            "damn meme sound short"),
    ("wow",             "anime wow sound meme"),
    ("yeet",            "yeet sound effect short"),
    ("error_windows",   "windows xp error sound"),
    ("sad_violin",      "sad violin meme short"),
    ("inception",       "inception horn bwaaa short"),
    ("nice",            "nice meme sound effect short"),

    # Crowd/atmospheric
    ("crowd_cheer",     "stadium crowd cheer short"),
    ("crowd_aw",        "crowd booing aw short"),
    ("dj_drop_lets_go", "DJ lets go sample"),
    ("mlg_hitmarker",   "MLG hitmarker sound"),
    ("808_kick",        "808 kick one shot sample"),

    # Bollywood/Indian flavor
    ("dhol_hit",        "dhol drum hit one shot"),
    ("tabla_hit",       "tabla bol hit short"),
    ("sitar_run",       "sitar lick short sample"),
    ("dialog_balle",    "balle balle short sample"),

    # Synthwave/retro
    ("arcade_coin",     "arcade coin sound effect"),
    ("retro_zap",       "8-bit zap sound effect"),
    ("vhs_glitch",      "VHS glitch sound effect"),

    # Random spice
    ("dramatic_chord",  "dramatic chipmunk chord short"),
    ("record_scratch",  "record scratch freeze frame"),
    ("metal_pipe",      "metal pipe falling sound effect"),
    ("party_horn",      "party horn sound effect"),
    ("gun_cock",        "gun cock sound effect"),
    ("typewriter_ding", "typewriter ding bell short"),
    ("notification",    "discord notification short"),
    ("thunder",         "thunder clap short"),
]

def run(slug, query):
    out = OUT / f"{slug}__%(id)s.%(ext)s"
    print(f"\n=== {slug} / {query} ===", flush=True)
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--default-search", "ytsearch1",
        "-x", "--audio-format", "wav",
        "--no-playlist",
        # constrain to short clips
        "--match-filter", "duration<60",
        "-o", str(out),
        f"ytsearch1:{query}",
    ]
    try:
        subprocess.run(cmd, check=False, timeout=180)
    except subprocess.TimeoutExpired:
        print(f"timeout: {query}", flush=True)


def trim_all(max_seconds=10):
    """Trim every wav in OUT to first max_seconds. Idempotent (skips already trimmed)."""
    import soundfile as sf
    for p in OUT.glob("*.wav"):
        if p.name.startswith("trimmed_"):
            continue
        try:
            info = sf.info(str(p))
            dur = info.frames / info.samplerate
        except Exception:
            continue
        if dur <= max_seconds:
            continue
        sr = info.samplerate
        data, _ = sf.read(str(p), start=0, stop=int(max_seconds * sr), always_2d=False)
        sf.write(str(p), data, sr)
        print(f"  trimmed {p.name}: {dur:.1f}s -> {max_seconds}s")


if __name__ == "__main__":
    for slug, q in MEMES:
        run(slug, q)
    print("\n=== trim to <=10s ===")
    trim_all(10)
    print("\nDone.", OUT)
    print("Files:", len(list(OUT.glob("*.wav"))))
