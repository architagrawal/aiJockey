"""FastAPI server for live AiJockey generation on MI300X.

Endpoints:
  GET  /health              -> {"status":"ok","models_loaded":bool}
  POST /generate            -> upload clips + preset, return mp3 stream
  GET  /demos/{slug}.mp3    -> serve pre-baked demo

Auth: X-Key header must match SERVER_KEY env var.
Idle: each request writes timestamp to /var/run/aijockey-last (used by idle_destroy.sh).
"""
from __future__ import annotations
import os, sys, time, hmac, hashlib, shutil, subprocess, tempfile, uuid, json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SERVER_KEY = os.environ.get("SERVER_KEY", "")
IDLE_FILE = Path(os.environ.get("IDLE_FILE", "/tmp/aijockey-last"))
DEMO_DIR = ROOT / "demo_mp3"
RESULTS_DIR = ROOT / "output" / "live"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRESETS = {
    "festival_inferno":      dict(arc="peak",          prompt="festival main stage, euphoric drops, big bass, screaming crowd, anthemic"),
    "midnight_noir":         dict(arc="flat_low",      prompt="after-hours noir, smoky melancholy, lo-fi nostalgia, dimly lit, slow burn"),
    "neon_retrowave":        dict(arc="rollercoaster", prompt="80s synthwave neon arcade, driving arpeggios, retro nostalgia, vintage warmth"),
    "east_meets_bass":       dict(arc="rollercoaster", prompt="sitar and tabla over deep bass, raga-electronic fusion, indian classical meets dubstep"),
    "bollywood_block_party": dict(arc="build",         prompt="bollywood club anthem to punjabi drill, dancefloor heat, festival fusion, 808s"),
}

MIN_CLIPS, MAX_CLIPS = 3, 6
MIN_DURATION, MAX_DURATION = 90, 300

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


def cache_key(file_hashes: list[str], preset: str, duration: int) -> str:
    h = hashlib.sha256()
    for fh in sorted(file_hashes):
        h.update(fh.encode())
    h.update(preset.encode())
    h.update(str(duration).encode())
    return h.hexdigest()[:16]


@app.get("/health")
def health():
    touch_idle()
    return {"status": "ok", "presets": list(PRESETS), "demos_available": [p.name for p in DEMO_DIR.glob("*.mp3")] if DEMO_DIR.exists() else []}


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
    preset: str = Form(...),
    duration: int = Form(180),
    files: list[UploadFile] = File(...),
    x_key: str | None = Header(default=None, alias="X-Key"),
):
    check_key(x_key)
    touch_idle()

    if preset not in PRESETS:
        raise HTTPException(400, f"unknown preset; valid: {list(PRESETS)}")
    if not (MIN_CLIPS <= len(files) <= MAX_CLIPS):
        raise HTTPException(400, f"need {MIN_CLIPS}-{MAX_CLIPS} clips, got {len(files)}")
    if not (MIN_DURATION <= duration <= MAX_DURATION):
        raise HTTPException(400, f"duration must be {MIN_DURATION}-{MAX_DURATION}s")

    # save uploads to a temp pool dir + compute hashes
    job_id = uuid.uuid4().hex[:12]
    pool_dir = RESULTS_DIR / job_id / "clips"
    pool_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = RESULTS_DIR / job_id / "cache"
    out_dir = RESULTS_DIR / job_id
    file_hashes = []
    for f in files:
        data = await f.read()
        if not data:
            raise HTTPException(400, f"empty file: {f.filename}")
        if len(data) > 25 * 1024 * 1024:
            raise HTTPException(400, f"file too large: {f.filename}")
        digest = hashlib.sha256(data).hexdigest()[:16]
        file_hashes.append(digest)
        ext = Path(f.filename or "").suffix.lower() or ".wav"
        if ext not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
            raise HTTPException(400, f"unsupported format: {ext}")
        (pool_dir / f"{digest}{ext}").write_bytes(data)

    # cache check (same inputs -> same output)
    ck = cache_key(file_hashes, preset, duration)
    cached = RESULTS_DIR / "cache" / f"{ck}.mp3"
    if cached.exists():
        return FileResponse(str(cached), media_type="audio/mpeg", filename=f"{preset}_{ck}.mp3")

    # run pipeline
    cfg = PRESETS[preset]
    main_py = ROOT / "src" / "main.py"
    timeline = out_dir / "timeline.json"
    raw = out_dir / "raw_mix.wav"
    final = out_dir / "final_mix.wav"
    mp3 = out_dir / "final_mix.mp3"

    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise HTTPException(500, f"pipeline error: {' '.join(cmd[-3:])}\n{r.stderr[-2000:]}")

    run([sys.executable, str(main_py), "analyze", "--clips", str(pool_dir), "--cache", str(cache_dir)])
    run([sys.executable, str(main_py), "plan", "--cache", str(cache_dir), "--out", str(timeline),
         "--duration", str(duration), "--arc", cfg["arc"], "--prompt", cfg["prompt"],
         "--min_unique_clips", "3"])
    run([sys.executable, str(main_py), "execute", "--timeline", str(timeline),
         "--cache", str(cache_dir), "--out", str(raw)])
    run([sys.executable, str(main_py), "master", "--in_path", str(raw), "--out", str(final), "--lufs", "-9"])
    run(["ffmpeg", "-y", "-i", str(final), "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3)])

    # cache store
    cached.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mp3, cached)

    # cleanup raw stems async
    background.add_task(_cleanup_job, out_dir, keep=mp3)

    touch_idle()
    return FileResponse(str(mp3), media_type="audio/mpeg", filename=f"{preset}_{ck}.mp3")


def _cleanup_job(job_dir: Path, keep: Path) -> None:
    try:
        for p in job_dir.rglob("*"):
            if p.is_file() and p != keep:
                try: p.unlink()
                except Exception: pass
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.api:app", host="0.0.0.0", port=8000, log_level="info")
