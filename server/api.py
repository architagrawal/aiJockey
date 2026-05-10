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
ARCS_FULL = ["build", "peak", "rollercoaster", "descend", "flat_high", "flat_low", "custom"]
PHASE1_ARCS = ["build", "peak", "flat_low"]


def _phase_arcs() -> list[str]:
    return PHASE1_ARCS if os.environ.get("AIJOCKEY_PHASE", "1") == "1" else ARCS_FULL


ARCS = ARCS_FULL  # back-compat for /health response

# Phase A polish input bands. Override via AIJOCKEY_BPM_MIN/MAX.
BPM_MIN_PHASE1 = float(os.environ.get("AIJOCKEY_BPM_MIN", "100"))
BPM_MAX_PHASE1 = float(os.environ.get("AIJOCKEY_BPM_MAX", "135"))

def _min_clips() -> int:
    """Phase 1 plan §1.4: min 3 clips. Below 3, planner has no room."""
    if os.environ.get("AIJOCKEY_PHASE", "1") == "1":
        return 3
    return 2


MIN_CLIPS, MAX_CLIPS = _min_clips(), 8
MIN_DURATION = 30
MAX_DURATION_HARD = 600
MAX_FILE_BYTES = 25 * 1024 * 1024
LIBRARY_MAX_PICK = 12
ALLOWED_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

EXPORT_MP3 = "mp3"
EXPORT_WAV = "wav"
EXPORT_FLAC = "flac"

app = FastAPI(title="AiJockey")


@app.on_event("startup")
def _warmup_director() -> None:
    """Pre-load HF Director model so first /generate doesn't pay download/load cost.

    Skipped if AIJOCKEY_USE_DIRECTOR_LLM=0 or AIJOCKEY_WARMUP=0.
    """
    if os.environ.get("AIJOCKEY_USE_DIRECTOR_LLM", "1").lower() in ("0", "false", "no"):
        return
    if os.environ.get("AIJOCKEY_WARMUP", "1").lower() in ("0", "false", "no"):
        return
    try:
        from director import run_director
        run_director(
            user_prompt="warmup", arc_preset="build",
            clip_count_estimate=2, max_transitions_hint=2,
            approx_duration_seconds=60.0,
        )
        jlog("-", "warmup_director_ok")
    except Exception as e:
        jlog("-", "warmup_director_failed", err=str(e)[:300])


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
    """All available library clips on disk (alphabetical fallback)."""
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


def lib_count_for_mode(mode: str, user_clip_count: int,
                       user_clip_total_sec: float, target_duration: float
                       ) -> int:
    """Compute how many library clips to add given semantic mode.

    Sweet spot moves with (user_count, user_total_dur, target_duration):
      - short user pool + long target = need many lib clips
      - long user clips already cover target = need 0 lib clips
    Mode caps the upper bound:
      - tight        = 0 (user clips only)
      - balanced     = enough to fill 30% headroom
      - exploratory  = up to LIBRARY_MAX_PICK
    """
    if mode == "tight":
        return 0
    needed_sec = max(target_duration * 1.3, 60.0)  # 30% headroom
    deficit = max(0.0, needed_sec - user_clip_total_sec)
    raw = int(deficit / 60.0 + 0.5)  # ~1 clip/min
    if mode == "balanced":
        return max(0, min(LIBRARY_MAX_PICK // 2, raw))
    # exploratory
    return max(0, min(LIBRARY_MAX_PICK, raw + user_clip_count // 2))


def _user_pool_centroid(saved_paths: list[Path], cache_dir: Path
                        ) -> "np.ndarray | None":
    """Mean of CLAP embeddings across user clips. None if cache absent."""
    try:
        import numpy as np
    except ImportError:
        return None
    arrs: list = []
    for sp in saved_paths:
        npz_path = cache_dir / f"{sp.stem}.npz"
        if not npz_path.exists():
            continue
        try:
            with np.load(npz_path) as d:
                if "clap" in d:
                    a = np.asarray(d["clap"], dtype=np.float32).reshape(-1)
                    if a.size:
                        arrs.append(a)
        except Exception:
            continue
    if not arrs:
        return None
    import numpy as np
    return np.mean(np.stack(arrs), axis=0)


def _bpm_compat(lib_bpm: float, user_bpm_mean: float, bpm_tol_pct: float) -> bool:
    """Octave-aware BPM compatibility check.

    Per dj_research.md, open-format DJs jump BPMs regularly (echo_out / cut
    transitions handle BPM jumps musically). Half-time/double-time are
    natively compatible — a 87 BPM dnb track played alongside a 174 BPM
    dnb track is the SAME pulse perceptually (every 2nd beat = downbeat
    of half-time). Octave-fold both BPMs into canonical band before
    comparing so trap-tempo half-time errors don't reject good matches.

    Default 15% widens to effective ~30% via octave equivalence.
    """
    if lib_bpm <= 0 or user_bpm_mean <= 0:
        return True  # missing data → allow, score will catch
    try:
        from tempo_octave import normalize_tempo
        u = normalize_tempo(user_bpm_mean)
        l = normalize_tempo(lib_bpm)
    except Exception:
        u, l = user_bpm_mean, lib_bpm
    return abs(l - u) / u <= bpm_tol_pct / 100.0


def _candidate_pool_for_user_clip(user_emb, user_bpm_mean: float | None,
                                    bpm_tol_pct: float, top_k: int):
    """Score the whole library by cosine to ONE user clip's CLAP embedding.

    Returns list of (sim, path, npz_path, meta_path) sorted desc by sim.
    Multi-query retrieval primitive — caller unions top-K across user
    clips so picks connect to AT LEAST ONE user clip individually
    (not the averaged centroid that loses individual character).
    """
    try:
        import numpy as np
    except ImportError:
        return []
    if not LIBRARY_CACHE_DIR.exists():
        return []
    enorm = float(np.linalg.norm(user_emb)) + 1e-9
    cands = []
    for p in sorted(LIBRARY_CLIPS_DIR.iterdir()):
        if p.suffix.lower() not in ALLOWED_EXTS:
            continue
        npz = LIBRARY_CACHE_DIR / f"{p.stem}.npz"
        meta = LIBRARY_CACHE_DIR / f"{p.stem}.json"
        if not npz.exists() or not meta.exists():
            continue
        try:
            with np.load(npz) as d:
                a = np.asarray(d["clap"], dtype=np.float32).reshape(-1) \
                    if "clap" in d else None
            if a is None or a.size == 0:
                continue
            if user_bpm_mean is not None:
                import json as _json
                m = _json.loads(meta.read_text())
                bpm = float(m.get("tempo", 0.0))
                # Octave-aware BPM gate (was hard ±15%, blocked all
                # half-time-detected library clips). Open-format DJs do
                # BPM-jumps; half/double tempo are natively compatible.
                if not _bpm_compat(bpm, user_bpm_mean, bpm_tol_pct):
                    continue
            sim = float((a @ user_emb) / (np.linalg.norm(a) * enorm + 1e-9))
            cands.append((sim, p, npz, meta))
        except Exception:
            continue
    cands.sort(key=lambda x: -x[0])
    return cands[:top_k]


def library_clip_paths_clap(centroid, count: int,
                              user_bpm_mean: float | None = None,
                              bpm_tol_pct: float = 30.0,
                              user_keys: list[str] | None = None,
                              user_clap_embs: list = None,
                              ) -> list[Path]:
    """Pick top-`count` library clips by COMPOSITE score combining:

      α · CLAP cosine to centroid     (vibe / genre / mood)
      β · key-compat bonus            (Camelot wheel adjacency to ANY user key)
      γ · BPM proximity bonus         (closer to user_bpm_mean = better)
      δ · outlier penalty             (clip distant from EVERY user clip,
                                       not just centroid → forces incoherent
                                       inclusions out of pool)

    Pure CLAP cosine to centroid is OK for vibe but blind to key clashes
    and rewards "average" picks that may sit far from every individual user
    clip. The composite catches both failure modes responsible for the
    v6_libaug 0.94 probe severity.

    Falls back to alphabetical first-N if centroid is None or library
    cache empty. Backwards compatible: extra kwargs (user_keys,
    user_clap_embs) are optional — caller can omit and get CLAP-only
    behavior with BPM filter (legacy).
    """
    if count <= 0:
        return []
    if centroid is None:
        return library_clip_paths(limit=count)
    try:
        import numpy as np
    except ImportError:
        return library_clip_paths(limit=count)
    if not LIBRARY_CACHE_DIR.exists():
        return []

    # Multi-query candidate set: instead of scoring the entire library by
    # cosine to the AVERAGED centroid (which loses individual user-clip
    # character — Spotify/YT 2-tower lesson), retrieve top-K per user clip
    # individually then take the UNION. Re-ranking below applies composite
    # score to this filtered candidate set. Guarantees each library pick
    # is close to at least ONE user clip, not just close to a meaningless
    # average. Skipped when user_clap_embs not supplied (legacy path).
    candidate_paths: set[Path] = set()
    if user_clap_embs:
        per_clip_top_k = max(2 * count, 8)
        for ue in user_clap_embs:
            for sim, p, _npz, _meta in _candidate_pool_for_user_clip(
                    ue, user_bpm_mean, bpm_tol_pct, per_clip_top_k):
                candidate_paths.add(p)
        try:
            print(f"[lib_pick] multi-query candidate set: "
                  f"{len(candidate_paths)} (top-{per_clip_top_k} × "
                  f"{len(user_clap_embs)} user clips, deduped)")
        except Exception:
            pass

    # Camelot adjacency lookup: keys are compatible if same number, ±1 number,
    # or same-letter swap. Distance 0/1 = compatible, 2+ = clash.
    def _camelot_compat_bonus(lib_key: str, user_keys_list: list[str]) -> float:
        if not user_keys_list or not lib_key or lib_key == '?':
            return 0.0
        try:
            from camelot import camelot_distance
        except Exception:
            return 0.0
        best_dist = 99
        for uk in user_keys_list:
            if not uk or uk == '?':
                continue
            try:
                d = camelot_distance(lib_key, uk)
                best_dist = min(best_dist, d)
            except Exception:
                continue
        if best_dist == 0:
            return 1.0   # same key = max bonus
        if best_dist == 1:
            return 0.6   # adjacent / relative
        if best_dist == 2:
            return 0.0
        return -0.3      # 3+ steps = mild penalty

    def _outlier_penalty(lib_emb, user_embs_list) -> float:
        """Penalty if lib_emb is far from EVERY user clip individually."""
        if not user_embs_list:
            return 0.0
        sims = []
        ln = np.linalg.norm(lib_emb) + 1e-9
        for ue in user_embs_list:
            un = np.linalg.norm(ue) + 1e-9
            sims.append(float((lib_emb @ ue) / (ln * un)))
        max_sim = max(sims) if sims else 0.0
        # if best individual sim < 0.55 → penalize (clip doesn't fit any user clip)
        if max_sim < 0.55:
            return -0.4 * (0.55 - max_sim) / 0.55
        return 0.0

    # Weights — tuned for "prefer cohesive over diverse".
    A_CLAP = 1.0
    B_KEY = 0.35
    G_BPM = 0.25
    D_OUTLIER = 1.0   # multiplier on already-negative penalty

    cnorm = float(np.linalg.norm(centroid)) + 1e-9
    candidates: list[tuple[float, Path, dict]] = []
    # Iterate the filtered candidate set when multi-query was used,
    # else fall back to whole library iteration (legacy path).
    iter_paths = (sorted(candidate_paths) if candidate_paths
                  else sorted(LIBRARY_CLIPS_DIR.iterdir()))
    for p in iter_paths:
        if p.suffix.lower() not in ALLOWED_EXTS:
            continue
        npz = LIBRARY_CACHE_DIR / f"{p.stem}.npz"
        meta = LIBRARY_CACHE_DIR / f"{p.stem}.json"
        if not npz.exists() or not meta.exists():
            continue
        try:
            with np.load(npz) as d:
                a = np.asarray(d["clap"], dtype=np.float32).reshape(-1) \
                    if "clap" in d else None
            if a is None or a.size == 0:
                continue
            sim = float((a @ centroid) / (np.linalg.norm(a) * cnorm + 1e-9))
            import json as _json
            m = _json.loads(meta.read_text())
            bpm = float(m.get("tempo", 0.0))
            lib_key = str(m.get("key", "?"))

            # Octave-aware BPM filter — half/double-time tracks are
            # treated as compatible. Open-format DJs jump BPMs musically
            # via echo_out/cut transitions per dj_research.md.
            if user_bpm_mean is not None and bpm > 0:
                if not _bpm_compat(bpm, user_bpm_mean, bpm_tol_pct):
                    continue

            # BPM proximity score (within band) — octave-fold both for fair
            # comparison so half-time-detected lib clip doesn't lose to
            # full-tempo lib clip just because of detection inconsistency.
            bpm_score = 0.0
            if user_bpm_mean is not None and bpm > 0:
                try:
                    from tempo_octave import normalize_tempo
                    u_n = normalize_tempo(user_bpm_mean)
                    l_n = normalize_tempo(bpm)
                    bpm_score = max(0.0, 1.0 - abs(l_n - u_n) / u_n / 0.15)
                except Exception:
                    bpm_score = max(0.0, 1.0 - abs(bpm - user_bpm_mean) / user_bpm_mean / 0.10)

            key_bonus = _camelot_compat_bonus(lib_key, user_keys or [])
            outlier = _outlier_penalty(a, user_clap_embs or [])

            score = (A_CLAP * sim
                     + B_KEY * key_bonus
                     + G_BPM * bpm_score
                     + D_OUTLIER * outlier)
            candidates.append((score, p,
                               {'clap': sim, 'key': key_bonus,
                                'bpm': bpm_score, 'outlier': outlier}))
        except Exception:
            continue
    candidates.sort(key=lambda x: -x[0])
    # Telemetry: log top-N picks with score breakdown for debugging.
    try:
        for i, (sc, pp, parts) in enumerate(candidates[:min(count, 5)]):
            print(f"[lib_pick] #{i} {pp.stem[:40]} score={sc:.3f} "
                  f"clap={parts['clap']:.2f} key={parts['key']:+.2f} "
                  f"bpm={parts['bpm']:.2f} outlier={parts['outlier']:+.2f}")
    except Exception:
        pass
    # Quality gate: if EVERY top candidate scores < threshold, the library
    # has nothing close to the user pool — better to return empty than
    # force incompatible clips into the mix (the v6_libaug failure mode:
    # 0.94 probe severity from 6 forced disparate-genre picks).
    # Caller logs a warning + falls back to user-only mix (mix_mode=tight).
    POOL_MISMATCH_THRESHOLD = 0.0  # composite score below this = no clip is "good"
    if candidates and all(c[0] < POOL_MISMATCH_THRESHOLD for c in candidates[:count]):
        try:
            top = candidates[0][0]
            print(f"[lib_pick] WARN: pool-library mismatch — top score {top:.3f} "
                  f"< threshold {POOL_MISMATCH_THRESHOLD}; returning EMPTY "
                  f"(library has no clip stylistically close to user pool)")
        except Exception:
            pass
        return []
    return [p for _, p, _ in candidates[:count]]


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
    mix_mode: str = "balanced",
    library_role: str | None = None,
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

    # Tag user clips with source='user' in their cache JSON. (Library
    # clips inherit source='library' on link from cache_dir below.)
    for sp in saved_paths:
        cj = cache_dir / f"{sp.stem}.json"
        if cj.exists():
            try:
                cm = json.loads(cj.read_text())
                cm['source'] = 'user'
                cj.write_text(json.dumps(cm, indent=2, default=str))
            except Exception:
                pass

    # Phase A polish §1.3: BPM band check post-analyze. User-uploaded clips
    # outside [BPM_MIN, BPM_MAX] are flagged as warnings (Phase 1 is tuned
    # for 4-on-floor 100-135). We don't hard-reject so users get a graceful
    # degraded mix instead of an error.
    if os.environ.get("AIJOCKEY_PHASE", "1") == "1":
        for sp in saved_paths:
            cache_json = cache_dir / f"{sp.stem}.json"
            if not cache_json.exists():
                continue
            try:
                with open(cache_json) as f:
                    cm = json.load(f)
                bpm = float(cm.get("tempo", 0.0))
            except Exception:
                continue
            if bpm <= 0:
                continue
            if not (BPM_MIN_PHASE1 <= bpm <= BPM_MAX_PHASE1):
                msg = (f"{sp.name}: BPM {bpm:.0f} outside Phase 1 band "
                       f"[{BPM_MIN_PHASE1:.0f}-{BPM_MAX_PHASE1:.0f}]; "
                       f"transitions may sound off. Set AIJOCKEY_PHASE=2 to allow.")
                ingest_warnings.append(msg)
                jlog(job_id, "bpm_band_warn", clip=sp.name, bpm=bpm)

    # Library augmentation: pick CLAP-similar library clips when use_library
    # and lib_count_for_mode > 0. Otherwise alphabetical fallback / no library.
    library_picked: list[Path] = []
    if use_library:
        # Compute user pool stats — BPM mean, key list, per-clip CLAP embeddings.
        # Composite library picker uses all three (was CLAP centroid only).
        user_total_sec = sum(audio_duration_seconds(p) for p in saved_paths)
        user_bpms: list[float] = []
        user_keys: list[str] = []
        user_embs: list = []
        try:
            import numpy as _np
        except Exception:
            _np = None
        for sp in saved_paths:
            cj = cache_dir / f"{sp.stem}.json"
            if cj.exists():
                try:
                    cm = json.loads(cj.read_text())
                    if cm.get("tempo"):
                        user_bpms.append(float(cm["tempo"]))
                    k = cm.get("key")
                    if k and k != "?":
                        user_keys.append(str(k))
                except Exception:
                    pass
            if _np is not None:
                npz = cache_dir / f"{sp.stem}.npz"
                if npz.exists():
                    try:
                        with _np.load(npz) as d:
                            if "clap" in d:
                                e = _np.asarray(d["clap"], dtype=_np.float32).reshape(-1)
                                if e.size:
                                    user_embs.append(e)
                    except Exception:
                        pass
        user_bpm_mean = (sum(user_bpms) / len(user_bpms)) if user_bpms else None
        n_lib = lib_count_for_mode(mix_mode, len(saved_paths),
                                    user_total_sec, float(duration))
        if n_lib > 0:
            centroid = _user_pool_centroid(saved_paths, cache_dir)
            library_picked = library_clip_paths_clap(
                centroid, n_lib, user_bpm_mean=user_bpm_mean,
                user_keys=user_keys, user_clap_embs=user_embs)
            if not library_picked:
                # Two reasons we get []: (1) centroid None — no embeddings
                # yet; fall back to alphabetical. (2) composite picker
                # rejected ALL because none matched user pool (all scores
                # < 0). In case (2) we honor that: do NOT pollute mix with
                # alphabetical picks. Telemetry distinguishes via _user_pool_centroid.
                if centroid is None:
                    library_picked = library_clip_paths(limit=n_lib)
                else:
                    ingest_warnings.append(
                        f"library augmentation skipped: no library clip "
                        f"stylistically close to user pool "
                        f"(use mix_mode=tight to silence this warning)")
        jlog(job_id, "library_pick",
             mode=mix_mode, role=library_role,
             user_clips=len(saved_paths), user_bpm_mean=user_bpm_mean,
             user_total_sec=round(user_total_sec, 1),
             lib_count=len(library_picked))
        for lib_clip in library_picked:
            link_or_copy(lib_clip, pool_dir / lib_clip.name)
            stem = lib_clip.stem
            for extn in ("json", "npz"):
                src = LIBRARY_CACHE_DIR / f"{stem}.{extn}"
                if src.exists():
                    link_or_copy(src, cache_dir / f"{stem}.{extn}")
            # Stamp source='library' on the linked cache JSON. We rewrite
            # the in-cache copy, leaving LIBRARY_CACHE_DIR untouched.
            cj = cache_dir / f"{stem}.json"
            if cj.exists():
                try:
                    # If symlink, replace with concrete file before edit
                    if cj.is_symlink():
                        real = cj.resolve()
                        cj.unlink()
                        shutil.copy2(real, cj)
                    cm = json.loads(cj.read_text())
                    cm['source'] = 'library'
                    cj.write_text(json.dumps(cm, indent=2, default=str))
                except Exception:
                    pass
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

    # Multimodal Director: pass user audio paths so an audio-capable model
    # (e.g. Qwen2-Audio when HF_DIRECTOR_MODEL contains 'audio') hears the
    # actual clips before planning. _call_qwen2audio caps at 6 clips
    # internally to bound input length. Library paths intentionally not
    # passed — they don't carry user intent.
    audio_paths_for_director = [str(p) for p in saved_paths]
    director = run_director(
        user_prompt=base_prompt,
        arc_preset=base_arc,
        clip_count_estimate=len(saved_paths),
        coherence_hint=coherence,
        max_transitions_hint=estimate_max_transitions_for_pool(len(clips), float(duration)),
        approx_duration_seconds=float(duration),
        clips_meta=clips,
        audio_clip_paths=audio_paths_for_director,
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


VALID_MIX_MODES = ("tight", "balanced", "exploratory")
VALID_LIBRARY_ROLES = ("any", "fill_gaps", "warmup_outro", "bridges_only")


@app.post("/generate")
async def generate(
    background: BackgroundTasks,
    preset: str = Form("festival_inferno"),
    duration: int = Form(180),
    files: list[UploadFile] = File(...),
    use_library: bool = Form(False),
    mix_mode: str = Form("balanced"),
    library_role: str | None = Form(None),
    prompt: str | None = Form(None),
    arc: str | None = Form(None),
    seed: int | None = Form(None),
    lufs: float = Form(-9.0),
    export_format: str = Form(EXPORT_MP3),
    instrumental_only: bool = Form(True),
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
    valid_arcs = _phase_arcs()
    if arc is not None and arc not in valid_arcs:
        raise HTTPException(
            400,
            detail=f"invalid arc {arc!r}; allowed in current phase: {valid_arcs}",
        )
    mode = (mix_mode or "balanced").lower().strip()
    if mode not in VALID_MIX_MODES:
        raise HTTPException(400, detail=f"invalid mix_mode {mix_mode!r}; "
                                         f"valid: {VALID_MIX_MODES}")
    role = (library_role or "").lower().strip() or None
    if role is not None and role not in VALID_LIBRARY_ROLES:
        raise HTTPException(400, detail=f"invalid library_role {library_role!r}; "
                                         f"valid: {VALID_LIBRARY_ROLES}")
    # tight mode forces use_library off regardless of toggle.
    if mode == "tight":
        use_library = False

    # Phase 1: min 3 clips. If user gave fewer, require library augmentation.
    min_clips = _min_clips()
    if len(files) < min_clips and not use_library:
        raise HTTPException(
            400,
            detail=f"need at least {min_clips} clips OR use_library=true to "
                   f"augment from preanalyzed pool (got {len(files)})",
        )
    if not (1 <= len(files) <= MAX_CLIPS):
        raise HTTPException(400, detail=f"max {MAX_CLIPS} user clips")

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
            # Stem-swap path already mutes vocals during overlaps; the
            # instrumental_only toggle additionally suppresses vocals
            # throughout segment bodies. Implemented as env var read by
            # execute.py via the existing AIJOCKEY_STEM_SWAP path.
            # _pipeline_lock guarantees only one job sets this env at a time;
            # try/finally restores prior value so a crash can't leave the
            # process permanently in instrumental-only mode.
            _ENV_KEY = "AIJOCKEY_INSTRUMENTAL_ONLY"
            _prev = os.environ.get(_ENV_KEY)
            if instrumental_only:
                os.environ[_ENV_KEY] = "1"
            else:
                os.environ.pop(_ENV_KEY, None)
            try:
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
                    mix_mode=mode,
                    library_role=role,
                )
            finally:
                if _prev is None:
                    os.environ.pop(_ENV_KEY, None)
                else:
                    os.environ[_ENV_KEY] = _prev

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

        headers = {"X-Job-Id": job_id, "X-Mix-Mode": mode}
        try:
            tl_p = RESULTS_DIR / job_id / "timeline.json"
            if tl_p.exists():
                with open(tl_p) as tf:
                    tl_blob = json.load(tf)
                tw = tl_blob.get("meta", {}).get("ingest_warnings") or []
                if tw:
                    headers["X-Ingest-Warnings"] = "; ".join(tw)[:1900]
                # Per-clip source breakdown for transparency.
                tl = tl_blob.get("timeline", [])
                cache_p = RESULTS_DIR / job_id / "cache"
                user_ids: set[str] = set()
                lib_ids: set[str] = set()
                for entry in tl:
                    cid = entry.get("clip_id")
                    if not cid:
                        continue
                    cj = cache_p / f"{cid}.json"
                    if cj.exists():
                        try:
                            cm = json.loads(cj.read_text())
                            src = (cm.get("source") or "").lower()
                            if src == "user":
                                user_ids.add(cid)
                            elif src == "library":
                                lib_ids.add(cid)
                        except Exception:
                            pass
                clips_used = {
                    "user_count": len(user_ids),
                    "library_count": len(lib_ids),
                    "library_ids": sorted(lib_ids)[:20],
                }
                headers["X-Clips-Used"] = json.dumps(clips_used)[:1900]

                # Cheap audio probes — auto-quality report on every render.
                # RMS env mismatch / vocal-bleed xcorr / spectral phasing per
                # junction, ~70% of artifacts at ~1% the cost of an audio LLM.
                # Adds ~100-300ms wall on a 3min mix; skipped when output WAV
                # not co-located with timeline (e.g. mp3 streamed before wav
                # finalize).
                try:
                    import sys as _sys
                    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
                    from audio_probes import probe_mix  # type: ignore
                    wav_for_probe = None
                    if Path(artifact).suffix.lower() == ".wav":
                        wav_for_probe = str(artifact)
                    else:
                        wav_alt = Path(artifact).with_suffix(".wav")
                        if wav_alt.exists():
                            wav_for_probe = str(wav_alt)
                    if wav_for_probe:
                        probe_t0 = time.perf_counter()
                        probe = probe_mix(wav_for_probe, str(tl_p))
                        probe_ms = int((time.perf_counter() - probe_t0) * 1000)
                        # Header: compact summary (full per-junction in
                        # /jobs/{id}/probe later if needed).
                        worst = max(probe.get("junctions") or [],
                                    key=lambda r: r.get("overall_severity", 0),
                                    default=None)
                        summary = {
                            "verdict": probe["verdict"],
                            "overall_severity": probe["overall_severity"],
                            "n_junctions": probe["n_junctions"],
                            "worst_junction": (
                                {"j": worst["junction_index"],
                                 "t": worst["time_sec"],
                                 "sev": worst["overall_severity"]}
                                if worst else None),
                            "probe_ms": probe_ms,
                        }
                        headers["X-Probe"] = json.dumps(summary)[:1900]
                        jlog(job_id, "probe", probe_ms,
                             verdict=probe["verdict"],
                             severity=probe["overall_severity"])

                        # CriticV2 advisory score header — works without
                        # checkpoint (returns None gracefully). Adds a
                        # second-opinion signal beyond probes; consumers
                        # may threshold-gate on a combined rule.
                        try:
                            import sys as _ss
                            _ss.path.insert(0, str(Path(__file__).resolve()
                                                    .parents[1] / "src"))
                            from critic_v2_score import score as _critic_score
                            cs_t0 = time.perf_counter()
                            cs = _critic_score(wav_for_probe)
                            cs_ms = int((time.perf_counter() - cs_t0) * 1000)
                            if cs is not None:
                                headers["X-Critic"] = json.dumps(
                                    {"score": round(float(cs), 3),
                                     "ms": cs_ms})[:200]
                                jlog(job_id, "critic", cs_ms, score=cs)
                        except Exception as _e:
                            jlog(job_id, "critic_skip", 0, err=str(_e)[:200])

                        # Per-render probe log → DPO data accumulator.
                        # Atomic JSONL append at $AIJOCKEY_PROBE_LOG (default
                        # /scratch/probes/log.jsonl). Captures Director plan,
                        # tier choices, probe scores, render time per call.
                        # Foundation for self-improving Director (S5/S7 DPO).
                        try:
                            from probe_log import log_render
                            director_blob = (tl_blob.get("meta", {})
                                             .get("director") or {})
                            log_render(
                                job_id=job_id,
                                prompt=tl_blob.get("meta", {}).get("prompt"),
                                arc=director_blob.get("arc"),
                                mix_mode=mode,
                                duration_target_s=tl_blob.get("meta", {})
                                                  .get("target_duration"),
                                duration_actual_s=probe.get("duration_sec"),
                                n_user_clips=len(user_ids),
                                n_library_clips=len(lib_ids),
                                director_used=bool(director_blob),
                                director_fallback=bool(
                                    director_blob.get("_fallback")),
                                set_narrative=director_blob.get("set_narrative"),
                                transition_tiers=director_blob.get(
                                    "transition_tiers"),
                                transition_intents=director_blob.get(
                                    "transition_intents"),
                                beat_source=tl_blob.get("meta", {})
                                            .get("beat_source"),
                                render_time_s=round(
                                    (time.perf_counter() - t_wall), 2),
                                probe=probe,
                            )
                        except Exception as _e:
                            jlog(job_id, "probe_log_skip", 0, err=str(_e)[:200])
                except Exception as e:
                    # Probes are advisory — never fail the response on probe errors.
                    jlog(job_id, "probe_skip", 0, err=str(e)[:200])
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
