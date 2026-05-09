"""FastAPI server for live AiJockey generation on MI300X.

Endpoints:
  GET  /health              -> {"status":"ok",...}
  POST /generate            -> upload clips + preset, return mp3
  GET  /demos/{slug}.mp3    -> serve pre-baked demo
  GET  /library/info        -> library size + style breakdown

Auth: X-Key header must match SERVER_KEY env var.
"""
from __future__ import annotations
import os, sys, time, hmac, hashlib, shutil, subprocess, uuid, json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SERVER_KEY = os.environ.get("SERVER_KEY", "")
IDLE_FILE = Path(os.environ.get("IDLE_FILE", "/tmp/aijockey-last"))
DEMO_DIR = ROOT / "demo_mp3"
LIBRARY_CLIPS_DIR = ROOT / "clips"          # curated audio
LIBRARY_CACHE_DIR = ROOT / "cache"          # curated analyses
RESULTS_DIR = ROOT / "output" / "live"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRESETS = {
    "festival_inferno":      dict(arc="peak",          prompt="festival main stage, euphoric drops, big bass, screaming crowd, anthemic"),
    "midnight_noir":         dict(arc="flat_low",      prompt="after-hours noir, smoky melancholy, lo-fi nostalgia, dimly lit, slow burn"),
    "neon_retrowave":        dict(arc="rollercoaster", prompt="80s synthwave neon arcade, driving arpeggios, retro nostalgia, vintage warmth"),
    "east_meets_bass":       dict(arc="rollercoaster", prompt="sitar and tabla over deep bass, raga-electronic fusion, indian classical meets dubstep"),
    "bollywood_block_party": dict(arc="build",         prompt="bollywood club anthem to punjabi drill, dancefloor heat, festival fusion, 808s"),
}
ARCS = ["build", "peak", "rollercoaster", "descend", "flat_high", "flat_low", "custom"]

MIN_CLIPS, MAX_CLIPS = 2, 8                  # widened: 2-8
MIN_DURATION = 30                            # was 90
MAX_DURATION_HARD = 600                      # absolute ceiling regardless of input length
MAX_FILE_BYTES = 25 * 1024 * 1024            # 25 MB / file
LIBRARY_MAX_PICK = 12                        # cap library clips merged in
ALLOWED_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

app = FastAPI(title="AiJockey")


def touch_idle() -> None:
    try:
        IDLE_FILE.write_text(str(int(time.time())))
    except Exception:
        pass


def check_key(x_key: str | None) -> None:
    if not SERVER_KEY:
        raise HTTPException(500, "server key not configured")
    if not x_key or not hmac.compare_digest(x_key, SERVER_KEY):
        raise HTTPException(401, "unauthorized")


def cache_key(file_hashes: list[str], preset: str, duration: int,
              prompt: str | None, arc: str | None, seed: int | None,
              lufs: float, use_library: bool, library_size: int) -> str:
    h = hashlib.sha256()
    for fh in sorted(file_hashes):
        h.update(fh.encode())
    h.update(f"|{preset}|{duration}|{prompt or ''}|{arc or ''}|{seed}|{lufs}|{use_library}|{library_size}".encode())
    return h.hexdigest()[:16]


def audio_duration_seconds(path: Path) -> float:
    """Cheap probe via soundfile; falls back to ffprobe."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=10)
            return float(r.stdout.strip() or 0.0)
        except Exception:
            return 0.0


def library_clip_paths(limit: int = LIBRARY_MAX_PICK) -> list[Path]:
    if not LIBRARY_CLIPS_DIR.exists() or not LIBRARY_CACHE_DIR.exists():
        return []
    out = []
    for p in sorted(LIBRARY_CLIPS_DIR.iterdir()):
        if p.suffix.lower() not in ALLOWED_EXTS:
            continue
        if (LIBRARY_CACHE_DIR / f"{p.stem}.json").exists() and (LIBRARY_CACHE_DIR / f"{p.stem}.npz").exists():
            out.append(p)
        if len(out) >= limit:
            break
    return out


def link_or_copy(src: Path, dst: Path) -> None:
    """Symlink if possible, else copy. Avoids duplicating GBs of cache per request."""
    if dst.exists():
        return
    try:
        os.symlink(str(src), str(dst))
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


@app.get("/health")
def health():
    touch_idle()
    return {
        "status": "ok",
        "presets": list(PRESETS),
        "arcs": ARCS,
        "demos_available": [p.name for p in DEMO_DIR.glob("*.mp3")] if DEMO_DIR.exists() else [],
        "library_size": len(library_clip_paths(limit=10**6)),
        "limits": {
            "clips_min": MIN_CLIPS, "clips_max": MAX_CLIPS,
            "duration_min": MIN_DURATION, "duration_max_hard": MAX_DURATION_HARD,
            "file_max_mb": MAX_FILE_BYTES // (1024 * 1024),
        },
    }


@app.get("/library/info")
def library_info():
    paths = library_clip_paths(limit=10**6)
    by_genre: dict[str, int] = {}
    for p in paths:
        genre = p.stem.split("__")[0] if "__" in p.stem else "other"
        by_genre[genre] = by_genre.get(genre, 0) + 1
    return {"size": len(paths), "by_genre": by_genre}


@app.get("/demos/{slug}.mp3")
def get_demo(slug: str):
    touch_idle()
    path = DEMO_DIR / f"{slug}.mp3"
    if not path.exists():
        raise HTTPException(404, "demo not found")
    return FileResponse(str(path), media_type="audio/mpeg", filename=path.name)


@app.post("/generate")
async def generate(
    background: BackgroundTasks,
    preset: str = Form("festival_inferno"),
    duration: int = Form(180),
    files: list[UploadFile] = File(...),
    use_library: bool = Form(False),
    prompt: str | None = Form(None),                 # advanced override
    arc: str | None = Form(None),                    # advanced override
    seed: int | None = Form(None),                   # advanced reproducibility
    lufs: float = Form(-9.0),                        # advanced loudness target
    x_key: str | None = Header(default=None, alias="X-Key"),
):
    check_key(x_key)
    touch_idle()

    # validate preset/arc
    if preset not in PRESETS and not (prompt and arc):
        raise HTTPException(400, f"unknown preset; valid: {list(PRESETS)} or supply prompt+arc")
    if arc is not None and arc not in ARCS:
        raise HTTPException(400, f"invalid arc; valid: {ARCS}")

    # validate clip count
    if not (MIN_CLIPS <= len(files) <= MAX_CLIPS):
        raise HTTPException(400, f"need {MIN_CLIPS}-{MAX_CLIPS} clips, got {len(files)}")

    # save uploads + compute durations + hashes
    job_id = uuid.uuid4().hex[:12]
    pool_dir = RESULTS_DIR / job_id / "clips"
    pool_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = RESULTS_DIR / job_id / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS_DIR / job_id

    file_hashes: list[str] = []
    saved_paths: list[Path] = []
    for f in files:
        data = await f.read()
        if not data:
            raise HTTPException(400, f"empty file: {f.filename}")
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(400, f"file too large: {f.filename} ({len(data)} > {MAX_FILE_BYTES})")
        ext = Path(f.filename or "").suffix.lower() or ".wav"
        if ext not in ALLOWED_EXTS:
            raise HTTPException(400, f"unsupported format: {ext}; allow {sorted(ALLOWED_EXTS)}")
        digest = hashlib.sha256(data).hexdigest()[:16]
        file_hashes.append(digest)
        sp = pool_dir / f"{digest}{ext}"
        sp.write_bytes(data)
        saved_paths.append(sp)

    # compute total duration from uploads
    durations = [audio_duration_seconds(p) for p in saved_paths]
    total_user_duration = sum(durations)
    if total_user_duration < MIN_DURATION:
        raise HTTPException(400, f"uploaded clips total {total_user_duration:.1f}s, need at least {MIN_DURATION}s")

    # determine effective max duration
    if use_library:
        max_dur = MAX_DURATION_HARD
    else:
        max_dur = min(MAX_DURATION_HARD, max(MIN_DURATION, int(total_user_duration)))

    if not (MIN_DURATION <= duration <= max_dur):
        raise HTTPException(400, f"duration must be {MIN_DURATION}-{max_dur}s (sum of clips={total_user_duration:.1f}s, library={use_library})")

    # cache hit check
    library_size_for_key = len(library_clip_paths(limit=10**6)) if use_library else 0
    ck = cache_key(file_hashes, preset, duration, prompt, arc, seed, lufs, use_library, library_size_for_key)
    cached_mp3 = RESULTS_DIR / "cache" / f"{ck}.mp3"
    if cached_mp3.exists():
        return FileResponse(str(cached_mp3), media_type="audio/mpeg", filename=f"{preset}_{ck}.mp3")

    # determine plan args (advanced override > preset)
    cfg = PRESETS.get(preset, {"arc": "build", "prompt": ""})
    final_prompt = prompt if prompt else cfg["prompt"]
    final_arc = arc if arc else cfg["arc"]

    # analyze user uploads
    main_py = ROOT / "src" / "main.py"

    def run(cmd, label=""):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise HTTPException(500, f"pipeline error in {label or cmd[-1]}: {r.stderr[-2000:]}")

    run([sys.executable, str(main_py), "analyze",
         "--clips", str(pool_dir), "--cache", str(cache_dir),
         "--workers", str(min(4, len(saved_paths)))], label="analyze user")

    # if use_library: symlink library audio + cache into job dirs
    if use_library:
        for lib_clip in library_clip_paths():
            link_or_copy(lib_clip, pool_dir / lib_clip.name)
            stem = lib_clip.stem
            for ext in ("json", "npz"):
                src = LIBRARY_CACHE_DIR / f"{stem}.{ext}"
                if src.exists():
                    link_or_copy(src, cache_dir / f"{stem}.{ext}")
            stems_src = LIBRARY_CACHE_DIR / "stems" / stem
            stems_dst = cache_dir / "stems" / stem
            if stems_src.exists() and not stems_dst.exists():
                stems_dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.symlink(str(stems_src), str(stems_dst))
                except (OSError, NotImplementedError):
                    shutil.copytree(str(stems_src), str(stems_dst))

    timeline = out_dir / "timeline.json"
    raw = out_dir / "raw_mix.wav"
    final = out_dir / "final_mix.wav"
    mp3 = out_dir / "final_mix.mp3"

    plan_cmd = [sys.executable, str(main_py), "plan",
                "--cache", str(cache_dir), "--out", str(timeline),
                "--duration", str(duration), "--arc", final_arc,
                "--prompt", final_prompt,
                "--min_unique_clips", str(min(3, max(2, len(saved_paths))))]
    run(plan_cmd, label="plan")
    run([sys.executable, str(main_py), "execute",
         "--timeline", str(timeline), "--cache", str(cache_dir), "--out", str(raw)],
        label="execute")
    run([sys.executable, str(main_py), "master",
         "--in_path", str(raw), "--out", str(final), "--lufs", str(lufs)],
        label="master")
    run(["ffmpeg", "-y", "-i", str(final), "-codec:a", "libmp3lame",
         "-b:a", "192k", str(mp3)], label="encode")

    cached_mp3.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mp3, cached_mp3)

    background.add_task(_cleanup_job, out_dir, keep=mp3)
    touch_idle()
    return FileResponse(str(mp3), media_type="audio/mpeg", filename=f"{preset}_{ck}.mp3")


def _cleanup_job(job_dir: Path, keep: Path) -> None:
    try:
        for p in job_dir.rglob("*"):
            if p.is_file() and p != keep and not p.is_symlink():
                try: p.unlink()
                except Exception: pass
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.api:app", host="0.0.0.0", port=8000, log_level="info")
