"""FastAPI server for live AiJockey generation on AMD / MI300X.

Endpoints:
  GET  /health   — liveness + limits
  GET  /ready    — process up (models may lazy-load on first job)
  POST /generate — multi-file upload, Director LLM, pipeline, return audio

Auth: X-Key header must match SERVER_KEY.

Env:
  SERVER_KEY, IDLE_FILE (optional), AI_DEVICE (cuda|cpu),
  AIJOCKEY_USE_DIRECTOR_LLM (1|0), AIJOCKEY_JOB_TIMEOUT_SEC (default 1200)
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "server"))

from joblog import jlog

SERVER_KEY = os.environ.get("SERVER_KEY", "")
IDLE_FILE = Path(os.environ.get("IDLE_FILE", "/tmp/aijockey-last"))
DEMO_DIR = ROOT / "demo_mp3"
LIBRARY_CLIPS_DIR = ROOT / "clips"
LIBRARY_CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "output" / "live"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JOB_TIMEOUT_SEC = int(os.environ.get("AIJOCKEY_JOB_TIMEOUT_SEC", "1200"))
AI_DEVICE = os.environ.get("AI_DEVICE", "cuda")

_pipeline_lock = threading.Lock()
_concurrent_denied = 0

PRESETS = {
    "festival_inferno": dict(arc="peak", prompt="festival main stage, euphoric drops, big bass"),
    "midnight_noir": dict(arc="flat_low", prompt="after-hours noir, smoky melancholy, lo-fi"),
    "neon_retrowave": dict(arc="rollercoaster", prompt="80s synthwave neon arcade nostalgia"),
    "east_meets_bass": dict(arc="rollercoaster", prompt="sitar tabla deep bass fusion"),
    "bollywood_block_party": dict(arc="build", prompt="bollywood club punjabi drill dancefloor"),
}
ARCS = ["build", "peak", "rollercoaster", "descend", "flat_high", "flat_low", "custom"]

MIN_CLIPS, MAX_CLIPS = 2, 8
MIN_DURATION = 30
MAX_DURATION_HARD = 600
MAX_FILE_BYTES = 25 * 1024 * 1024
LIBRARY_MAX_PICK = 12
ALLOWED_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

EXPORT_MP3 = "mp3"
EXPORT_WAV = "wav"
EXPORT_FLAC = "flac"

app = FastAPI(title="AiJockey")


def touch_idle() -> None:
    try:
        IDLE_FILE.write_text(str(int(time.time())))
    except Exception:
        pass


def check_key(x_key: str | None) -> None:
    if not SERVER_KEY:
        raise HTTPException(500, detail="server key not configured")
    if not x_key or not hmac.compare_digest(x_key, SERVER_KEY):
        raise HTTPException(401, detail="unauthorized")


def cache_key(
    file_hashes: list[str],
    preset: str,
    duration: int,
    prompt: str | None,
    arc: str | None,
    seed: int | None,
    lufs: float,
    use_library: bool,
    library_size: int,
    export_format: str,
) -> str:
    h = hashlib.sha256()
    for fh in sorted(file_hashes):
        h.update(fh.encode())
    h.update(
        f"|{preset}|{duration}|{prompt or ''}|{arc or ''}|{seed}|{lufs}|{use_library}|{library_size}|{export_format}".encode()
    )
    return h.hexdigest()[:16]


def audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        try:
            r = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return float(r.stdout.strip() or 0.0)
        except Exception:
            return 0.0


def probe_mp3_bitrate_kbps(path: Path) -> int | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=bit_rate", "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        br = int(r.stdout.strip())
        return br // 1000 if br else None
    except Exception:
        return None


def library_clip_paths(limit: int = LIBRARY_MAX_PICK) -> list[Path]:
    if not LIBRARY_CLIPS_DIR.exists() or not LIBRARY_CACHE_DIR.exists():
        return []
    out = []
    for p in sorted(LIBRARY_CLIPS_DIR.iterdir()):
        if p.suffix.lower() not in ALLOWED_EXTS:
            continue
        if (
            LIBRARY_CACHE_DIR / f"{p.stem}.json"
        ).exists() and (LIBRARY_CACHE_DIR / f"{p.stem}.npz").exists():
            out.append(p)
        if len(out) >= limit:
            break
    return out


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(str(src), str(dst))
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def disk_free_gb(path: Path = ROOT) -> float:
    try:
        u = shutil.disk_usage(path)
        return round(u.free / (1024**3), 2)
    except Exception:
        return -1.0


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
            "clips_min": MIN_CLIPS,
            "clips_max": MAX_CLIPS,
            "duration_min": MIN_DURATION,
            "duration_max_hard": MAX_DURATION_HARD,
            "file_max_mb": MAX_FILE_BYTES // (1024 * 1024),
            "job_timeout_sec": JOB_TIMEOUT_SEC,
        },
        "disk_free_gb": disk_free_gb(),
        "concurrent_denied_total": _concurrent_denied,
        "pipeline_locked": _pipeline_lock.locked(),
    }


@app.get("/ready")
def ready():
    touch_idle()
    return {"status": "ready", "disk_free_gb": disk_free_gb()}


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
        raise HTTPException(404, detail="demo not found")
    return FileResponse(str(path), media_type="audio/mpeg", filename=path.name)


def _encode_final(in_wav: Path, out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == EXPORT_MP3:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(in_wav), "-codec:a", "libmp3lame", "-q:a", "2", str(out_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    elif fmt == EXPORT_FLAC:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(in_wav), "-c:a", "flac", str(out_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        shutil.copy2(in_wav, out_path)


def _run_generate_sync(
    *,
    job_id: str,
    preset: str,
    duration: int,
    files_payload: list[tuple[str, bytes]],
    use_library: bool,
    prompt: str | None,
    arc: str | None,
    seed: int | None,
    lufs: float,
    export_format: str,
) -> tuple[Path, Path | None]:
    """Returns (artifact_path, optional timeline_path)."""
    t0 = time.perf_counter()
    pool_dir = RESULTS_DIR / job_id / "clips"
    pool_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = RESULTS_DIR / job_id / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS_DIR / job_id

    file_hashes: list[str] = []
    saved_paths: list[Path] = []
    ingest_warnings: list[str] = []

    for fname, data in files_payload:
        if not data:
            raise ValueError(f"empty file: {fname}")
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"file too large: {fname}")
        ext = Path(fname or "").suffix.lower() or ".wav"
        if ext not in ALLOWED_EXTS:
            raise ValueError(f"unsupported format: {ext}")
        digest = hashlib.sha256(data).hexdigest()[:16]
        file_hashes.append(digest)
        sp = pool_dir / f"{digest}{ext}"
        sp.write_bytes(data)
        saved_paths.append(sp)
        if ext == ".mp3":
            kbps = probe_mp3_bitrate_kbps(sp)
            if kbps is not None and kbps < 160:
                ingest_warnings.append(f"{fname}: low MP3 bitrate ~{kbps} kbps; lossless/WAV improves mix quality.")

    durations = [audio_duration_seconds(p) for p in saved_paths]
    total_user_duration = sum(durations)
    if total_user_duration < MIN_DURATION:
        raise ValueError(f"uploaded clips total {total_user_duration:.1f}s, need ≥{MIN_DURATION}s")

    if use_library:
        max_dur = MAX_DURATION_HARD
    else:
        max_dur = min(MAX_DURATION_HARD, max(MIN_DURATION, int(total_user_duration)))
    if not (MIN_DURATION <= duration <= max_dur):
        raise ValueError(f"duration must be {MIN_DURATION}-{max_dur}s")

    library_size_for_key = len(library_clip_paths(limit=10**6)) if use_library else 0
    ck = cache_key(
        file_hashes, preset, duration, prompt, arc, seed, lufs,
        use_library, library_size_for_key, export_format,
    )

    fc_map = {"mp3": ".mp3", "wav": ".wav", "flac": ".flac"}
    ext_out = fc_map.get(export_format, ".mp3")
    cached = RESULTS_DIR / "cache" / f"{ck}{ext_out}"
    if cached.exists():
        jlog(job_id, "cache_hit", (time.perf_counter() - t0) * 1000, path=str(cached))
        return cached, None

    from analyze import analyze_pool
    from director import estimate_max_transitions_for_pool, run_director
    from execute import execute
    from master import master
    from planner import (
        PlannerConfig,
        apply_llm_transition_tiers_to_timeline,
        attach_accent_hints,
        compute_pool_coherence,
        load_clips,
        plan,
        save_timeline,
    )

    t_an = time.perf_counter()
    analyze_pool(str(pool_dir), str(cache_dir), device=AI_DEVICE, force=False, workers=min(4, len(saved_paths)))
    jlog(job_id, "analyze", (time.perf_counter() - t_an) * 1000, disk_gb=disk_free_gb())

    if use_library:
        for lib_clip in library_clip_paths():
            link_or_copy(lib_clip, pool_dir / lib_clip.name)
            stem = lib_clip.stem
            for extn in ("json", "npz"):
                src = LIBRARY_CACHE_DIR / f"{stem}.{extn}"
                if src.exists():
                    link_or_copy(src, cache_dir / f"{stem}.{extn}")
            stems_src = LIBRARY_CACHE_DIR / "stems" / stem
            stems_dst = cache_dir / "stems" / stem
            if stems_src.exists() and not stems_dst.exists():
                stems_dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.symlink(str(stems_src), str(stems_dst))
                except (OSError, NotImplementedError):
                    shutil.copytree(str(stems_src), str(stems_dst))

    clips = load_clips(str(cache_dir))
    coherence = compute_pool_coherence(clips)

    cfg_preset = PRESETS.get(preset, {"arc": "build", "prompt": ""})
    base_prompt = prompt if prompt else cfg_preset["prompt"]
    base_arc = arc if arc else cfg_preset["arc"]

    director = run_director(
        user_prompt=base_prompt,
        arc_preset=base_arc,
        clip_count_estimate=len(saved_paths),
        coherence_hint=coherence,
        max_transitions_hint=estimate_max_transitions_for_pool(len(clips), float(duration)),
        approx_duration_seconds=float(duration),
    )
    final_prompt = director.get("text_prompt") or base_prompt
    final_arc = director.get("arc") or base_arc
    tiers = director.get("transition_tiers") or []
    accents = director.get("accent_hints") or []

    n_user = len(saved_paths)
    min_u = max(2, min(6, n_user))
    if use_library:
        min_u = min(min_u + 2, len(clips))

    compat_head_path = ROOT / "checkpoints" / "clap_compat_head.pt"
    cfg = PlannerConfig(
        target_duration=float(duration),
        surprise_budget=int(director.get("surprise_budget", 10)),
        callback_budget=int(director.get("callback_budget", 1)),
        max_clips=200,
        min_unique_clips=min(min_u, len(clips)),
        arc_shape=final_arc,
        text_prompt=final_prompt,
        pool_coherence=coherence,
        same_genre_tight_mix=bool(director.get("same_genre_tight_mix")),
        compat_head_ckpt=str(compat_head_path) if compat_head_path.exists() else None,
    )

    t_pl = time.perf_counter()
    n_best = int(os.environ.get("AIJOCKEY_N_BEST", "1"))
    if n_best > 1:
        from planner import plan_n_best
        timeline_list, _ = plan_n_best(clips, cfg, cache_dir=str(cache_dir), n_candidates=n_best)
    else:
        timeline_list = plan(clips, cfg)
    if os.environ.get("AIJOCKEY_APPLY_LLM_TIERS", "0").lower() in ("1", "true", "yes"):
        apply_llm_transition_tiers_to_timeline(timeline_list, tiers)
    attach_accent_hints(timeline_list, accents)

    max_stretch = 1.14 if coherence >= 0.58 else 1.08
    meta = {
        "max_stretch_ratio": max_stretch,
        "ingest_warnings": ingest_warnings,
        "pool_coherence": round(coherence, 4),
        "job_id": job_id,
    }

    timeline_path = out_dir / "timeline.json"
    save_timeline(timeline_list, str(timeline_path), meta=meta)
    jlog(job_id, "plan", (time.perf_counter() - t_pl) * 1000, entries=len(timeline_list))

    raw = out_dir / "raw_mix.wav"
    final = out_dir / "final_mix.wav"
    art = out_dir / f"final_mix{ext_out}"

    t_ex = time.perf_counter()
    execute(str(timeline_path), str(cache_dir), str(raw))
    jlog(job_id, "execute", (time.perf_counter() - t_ex) * 1000)

    t_m = time.perf_counter()
    master(str(raw), str(final), target_lufs=lufs)
    jlog(job_id, "master", (time.perf_counter() - t_m) * 1000)

    if export_format == EXPORT_WAV:
        shutil.copy2(final, art)
    else:
        _encode_final(final, art, export_format)

    RESULTS_DIR.joinpath("cache").mkdir(parents=True, exist_ok=True)
    shutil.copy2(art, cached)

    jlog(job_id, "done", (time.perf_counter() - t0) * 1000, artifact=str(art))
    return art, timeline_path


@app.post("/generate")
async def generate(
    background: BackgroundTasks,
    preset: str = Form("festival_inferno"),
    duration: int = Form(180),
    files: list[UploadFile] = File(...),
    use_library: bool = Form(False),
    prompt: str | None = Form(None),
    arc: str | None = Form(None),
    seed: int | None = Form(None),
    lufs: float = Form(-9.0),
    export_format: str = Form(EXPORT_MP3),
    x_key: str | None = Header(default=None, alias="X-Key"),
):
    global _concurrent_denied
    check_key(x_key)
    touch_idle()

    ef = export_format.lower().strip()
    if ef not in (EXPORT_MP3, EXPORT_WAV, EXPORT_FLAC):
        raise HTTPException(400, detail=f"export_format must be {EXPORT_MP3}|{EXPORT_WAV}|{EXPORT_FLAC}")

    if preset not in PRESETS and not (prompt and arc):
        raise HTTPException(400, detail=f"unknown preset; valid {list(PRESETS)} or supply prompt+arc")
    if arc is not None and arc not in ARCS:
        raise HTTPException(400, detail=f"invalid arc")

    if not (MIN_CLIPS <= len(files) <= MAX_CLIPS):
        raise HTTPException(400, detail=f"need {MIN_CLIPS}-{MAX_CLIPS} clips")

    acquired = _pipeline_lock.acquire(blocking=False)
    if not acquired:
        _concurrent_denied += 1
        jlog("-", "busy_reject", concurrent_denied=_concurrent_denied)
        raise HTTPException(
            503,
            detail="Server is rendering another mix (one GPU job at a time). Retry in a few minutes.",
            headers={"Retry-After": "120"},
        )

    job_id = uuid.uuid4().hex[:12]
    t_wall = time.perf_counter()

    try:
        files_payload: list[tuple[str, bytes]] = []
        for f in files:
            files_payload.append((f.filename or "clip.wav", await f.read()))

        def _wrapped():
            return _run_generate_sync(
                job_id=job_id,
                preset=preset,
                duration=duration,
                files_payload=files_payload,
                use_library=use_library,
                prompt=prompt,
                arc=arc,
                seed=seed,
                lufs=lufs,
                export_format=ef,
            )

        try:
            artifact, timeline_path = await asyncio.wait_for(
                asyncio.to_thread(_wrapped), timeout=float(JOB_TIMEOUT_SEC)
            )
        except asyncio.TimeoutError:
            jlog(job_id, "timeout", None, seconds=JOB_TIMEOUT_SEC)
            raise HTTPException(
                504,
                detail=f"Job exceeded {JOB_TIMEOUT_SEC}s budget. Try fewer/shorter clips or lower duration.",
            )
        except ValueError as e:
            raise HTTPException(400, detail=str(e))
        except Exception as e:
            jlog(job_id, "error", (time.perf_counter() - t_wall) * 1000, err=str(e)[:800])
            raise HTTPException(500, detail=str(e)[:2000])

        media_types = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}
        mt = media_types.get(ef, "application/octet-stream")
        fname = f"aijockey_{job_id}{Path(artifact).suffix}"

        headers = {"X-Job-Id": job_id}
        try:
            tl_p = RESULTS_DIR / job_id / "timeline.json"
            if tl_p.exists():
                with open(tl_p) as tf:
                    tw = json.load(tf).get("meta", {}).get("ingest_warnings") or []
                if tw:
                    headers["X-Ingest-Warnings"] = "; ".join(tw)[:1900]
        except Exception:
            pass

        background.add_task(_delayed_cleanup, RESULTS_DIR / job_id, [Path(artifact)], False)

        resp = FileResponse(str(artifact), media_type=mt, filename=fname, headers=headers)
        # Optional second file: timeline only via separate endpoint suggestion — keep simple.
        touch_idle()
        jlog(job_id, "response", (time.perf_counter() - t_wall) * 1000)
        return resp
    finally:
        _pipeline_lock.release()


def _delayed_cleanup(job_root: Path, keep: list[Path], fail: bool) -> None:
    time.sleep(2)
    deadline = time.time() + (3600 if fail else 30)
    # success: quick cleanup; failures could extend — v1 purge aggressively after success per plan
    try:
        for p in job_root.rglob("*"):
            if p.is_dir():
                continue
            kp = str(p.resolve())
            if any(str(k.resolve()) == kp for k in keep if k.exists()):
                continue
            try:
                p.unlink()
            except Exception:
                pass
        # drop empty dirs
        for sub in sorted(job_root.rglob("*"), reverse=True):
            try:
                if sub.is_dir() and not any(sub.iterdir()):
                    sub.rmdir()
            except Exception:
                pass
    except Exception:
        pass
    _ = deadline  # retention hook — future longer keep on fail


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.api:app", host="0.0.0.0", port=8000, log_level="info")
