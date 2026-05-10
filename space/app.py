"""HF Space: AiJockey — gradio-native UI.

Honest two-column layout. Real gradio components (DataFrame, Plot, Timer).
HF OAuth via gr.LoginButton.
"""
from __future__ import annotations
import os, hmac, json, requests, tempfile
from pathlib import Path
import gradio as gr

ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "demo_mp3"

ADMIN_PW    = os.environ.get("ADMIN_PW", "")
MI300X_URL  = os.environ.get("MI300X_URL", "")
MI300X_KEY  = os.environ.get("MI300X_KEY", "")

PRESETS = {
    "Festival Inferno":      ("festival_inferno",      "Western EDM peak. Big drops, anthemic."),
    "Midnight Noir":         ("midnight_noir",         "Dark cinematic, slow burn, after-hours."),
    "Neon Retrowave":        ("neon_retrowave",        "80s synthwave, driving nostalgia."),
    "East Meets Bass":       ("east_meets_bass",       "Sitar + tabla over deep electronic bass."),
    "Bollywood Block Party": ("bollywood_block_party", "Bollywood + Punjabi + drill mash."),
}
DEMO_MIXES = [
    ("Chillstep",   "chillstep_demo",    "Lush atmospheric chillstep · 5 min"),
    ("Drum & Bass", "dnb_demo",          "Energetic dnb · 6 min"),
    ("Dubstep",     "dubstep_demo",      "Heavy wobble dubstep · 5 min"),
    ("Future Bass", "future_bass_demo",  "Bright melodic future bass · 4 min"),
]
ARCS = ["build", "peak", "flat_low", "tomorrowland", "rollercoaster",
        "descend", "flat_high"]
LUFS_OPTIONS = {"streaming (-14)": -14, "club (-9)": -9, "competition (-6)": -6}

MIN_CLIPS, MAX_CLIPS = 2, 8
MIN_DURATION, MAX_DURATION_HARD = 30, 1800
BACKEND_TIMEOUT_SEC = 1200
MAX_FILE_MB = 75
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
ALLOWED_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

USER_PALETTE = [
    "#ff2d6f", "#22d3ee", "#facc15", "#a3e635",
    "#a78bfa", "#fb923c", "#34d399", "#f472b6",
]
LIBRARY_COLOR = "#475569"


def _probe_total_duration(file_paths):
    if not file_paths:
        return 0.0
    total = 0.0
    try:
        import soundfile as sf
        for p in file_paths:
            try:
                info = sf.info(p)
                total += info.frames / info.samplerate
            except Exception:
                pass
    except Exception:
        pass
    return total


def _validate_uploads(files) -> tuple[list[str], list[str]]:
    """Return (errors, oversized) lists. Errors halt submit; oversized = soft warn."""
    errs: list[str] = []
    oversized: list[str] = []
    if not files:
        return errs, oversized
    for f in files:
        path = f.name if hasattr(f, "name") else f
        try:
            sz = os.path.getsize(path)
        except Exception:
            errs.append(f"{Path(path).name}: cannot read")
            continue
        ext = Path(path).suffix.lower()
        if ext not in ALLOWED_EXTS:
            errs.append(f"{Path(path).name}: unsupported format {ext} "
                        f"(allowed: {', '.join(sorted(ALLOWED_EXTS))})")
            continue
        if sz > MAX_FILE_BYTES:
            oversized.append(f"{Path(path).name}: {sz/1024/1024:.1f} MB > "
                             f"{MAX_FILE_MB} MB cap")
    return errs, oversized


def update_duration_info(files, use_library):
    paths = [f.name if hasattr(f, "name") else f for f in (files or [])]
    errs, oversized = _validate_uploads(files)
    total = _probe_total_duration(paths)
    if use_library:
        max_dur = MAX_DURATION_HARD
    else:
        max_dur = max(MIN_DURATION, min(MAX_DURATION_HARD, int(total))) if total else MIN_DURATION
    n_ok = len(paths) - len(errs) - len(oversized)
    parts = [
        f"**{len(paths)}** clip(s) · {n_ok} valid · "
        f"uploaded **{total:.0f}s** · "
        f"max length **{max_dur}s** "
        f"(library {'on' if use_library else 'off'})"
    ]
    if errs:
        parts.append("⚠ " + " · ".join(errs))
    if oversized:
        parts.append("⚠ oversize: " + " · ".join(oversized))
    if len(paths) and not errs and not oversized:
        parts.append("✓ all files within limits")
    return "\n\n".join(parts)


def _fmt_time(sec):
    sec = max(0.0, sec)
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m}:{s:05.2f}"


def render_timeline_plot(timeline_json):
    """Matplotlib horizontal stacked bar of segments. Returns gr.Plot figure."""
    if not timeline_json or not timeline_json.get("segments"):
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segments = timeline_json["segments"]
    user_color = {}
    next_idx = 0
    fig, ax = plt.subplots(figsize=(10, 1.6), dpi=120)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    cursor = 0.0
    for seg in segments:
        dur = float(seg["duration_sec"])
        source = seg.get("source") or "?"
        label = seg.get("label") or "library"
        if source == "user":
            if label not in user_color:
                user_color[label] = USER_PALETTE[next_idx % len(USER_PALETTE)]
                next_idx += 1
            color = user_color[label]
        else:
            color = LIBRARY_COLOR
        ax.barh(0, dur, left=cursor, height=0.8, color=color, edgecolor="none")
        cursor += dur

    ax.set_xlim(0, max(1.0, cursor))
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("seconds", color="#94a3b8", fontsize=9)
    ax.tick_params(axis="x", colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig


def _decode_for_waveform(mp3_path: str, sr: int = 8000):
    """Decode audio to mono numpy at low sample rate via ffmpeg."""
    import subprocess
    import numpy as np
    wav_path = mp3_path + ".wf.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", str(sr),
             "-loglevel", "error", wav_path],
            check=True, capture_output=True, timeout=30)
        import soundfile as sf
        data, real_sr = sf.read(wav_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, int(real_sr)
    except Exception:
        return None, None


def render_colored_waveform(audio_path: str, timeline_json):
    """Decoded mono envelope, fill_between per-segment with source color."""
    import numpy as np
    samples, sr = _decode_for_waveform(audio_path)
    if samples is None or sr is None:
        return None
    n_target = 3000
    if len(samples) > n_target:
        bin_size = len(samples) // n_target
        env = np.abs(samples[:bin_size * n_target]).reshape(
            n_target, bin_size).max(axis=1)
    else:
        env = np.abs(samples)
    dur_sec = len(samples) / float(sr)
    t = np.linspace(0, dur_sec, len(env))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 2.4), dpi=120)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    segments = (timeline_json or {}).get("segments") or []
    user_color, idx = {}, 0
    cursor = 0.0
    span_colors = []
    for seg in segments:
        d = float(seg["duration_sec"])
        src = seg.get("source") or "?"
        lbl = seg.get("label") or "library"
        if src == "user":
            if lbl not in user_color:
                user_color[lbl] = USER_PALETTE[idx % len(USER_PALETTE)]
                idx += 1
            color = user_color[lbl]
        else:
            color = LIBRARY_COLOR
        span_colors.append((cursor, cursor + d, color))
        cursor += d

    if span_colors:
        for s, e, c in span_colors:
            mask = (t >= s) & (t <= e)
            if mask.any():
                ax.fill_between(t[mask], -env[mask], env[mask],
                                color=c, linewidth=0)
    else:
        ax.fill_between(t, -env, env, color="#ff2d6f", linewidth=0)

    ax.set_xlim(0, dur_sec)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([])
    ax.set_xlabel("seconds", color="#94a3b8", fontsize=9)
    ax.tick_params(axis="x", colors="#94a3b8", labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.tight_layout()
    return fig


def render_audiobox_plot(audiobox_json):
    """Matplotlib horizontal bar chart of 4 Audiobox aesthetics axes (0-10)."""
    if not audiobox_json:
        return None
    axes = [("PQ", "Production Quality"),
            ("PC", "Production Complexity"),
            ("CE", "Content Enjoyment"),
            ("CU", "Content Usefulness")]
    values = []
    labels = []
    for key, label in axes:
        v = audiobox_json.get(key)
        if v is None:
            continue
        values.append(float(v))
        labels.append(label)
    if not values:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=120)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    colors = ["#ff2d6f", "#22d3ee", "#facc15", "#a3e635"][:len(values)]
    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.6)
    ax.set_xlim(0, 10)
    ax.set_xlabel("score (0–10)", color="#94a3b8", fontsize=9)
    ax.tick_params(axis="y", colors="#e2e8f0", labelsize=10)
    ax.tick_params(axis="x", colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for bar, v in zip(bars, values):
        ax.text(v + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", color="#f5f6fa", fontsize=10,
                va="center", fontweight="600")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def render_segments_dataframe(timeline_json):
    """Build a list-of-rows for gr.DataFrame display."""
    if not timeline_json or not timeline_json.get("segments"):
        return []
    rows = []
    user_idx_map = {}
    next_idx = 0
    for seg in timeline_json["segments"]:
        source = seg.get("source") or "?"
        label = seg.get("label") or "library"
        if source == "user":
            if label not in user_idx_map:
                user_idx_map[label] = next_idx + 1
                next_idx += 1
            display = f"#{user_idx_map[label]} {seg.get('filename') or label}"[:48]
        else:
            display = "library"
        rows.append([
            _fmt_time(seg["start_sec"]),
            f"{seg['duration_sec']:.1f}s",
            display,
            seg.get("transition", "—"),
        ])
    return rows


# Friendly display names for non-obvious clip stems (test_clips pool).
_DISPLAY_NAME_OVERRIDES = {
    "test1": "Kanye West — Stronger",
    "test2": "The Weeknd",
}


def _shorten_sample_name(stem: str) -> str:
    """Strip YouTube IDs / verbose suffixes for cleaner UI labels."""
    import re
    if stem in _DISPLAY_NAME_OVERRIDES:
        return _DISPLAY_NAME_OVERRIDES[stem]
    s = stem
    s = re.sub(r"__[A-Za-z0-9_-]{8,}$", "", s)
    s = re.sub(r"_Official_Music_Video.*$", "", s, flags=re.IGNORECASE)
    s = s.replace("_", " ").strip()
    return s[:60]


def fetch_sample_clips() -> dict:
    """Return [(label, id)] of pre-staged clips on droplet.

    Test-clips origin gets a [research only] suffix so users / judges
    can distinguish copyrighted reference material from cleared user_set.
    """
    EMPTY = {"vocal": [], "instrumental": []}
    if not (MI300X_URL and MI300X_KEY):
        return EMPTY
    try:
        r = requests.get(f"{MI300X_URL}/sample_clips",
                         headers={"X-Key": MI300X_KEY}, timeout=8)
        if not r.ok:
            return EMPTY
        clips = r.json().get("clips", [])
        # Split into vocal-heavy (user_set + test_clips) and instrumental
        # (curated_ia). Returns dict so caller can render two CheckboxGroups.
        vocal: list[tuple[str, str]] = []
        instr: list[tuple[str, str]] = []
        for c in clips:
            origin = c.get("origin", "")
            if origin == "test_clips":
                tag = " [research only]"
            elif origin == "curated_ia":
                tag = " [Internet Archive · CC]"
            else:
                tag = ""
            dur = float(c.get("duration_sec", 0))
            mins = int(dur // 60)
            secs = int(dur - 60 * mins)
            mmss = f"{mins}:{secs:02d}"
            label = (f"{_shorten_sample_name(c['name'])} "
                     f"({mmss} · {c['size_mb']:.1f} MB){tag}")
            (instr if origin == "curated_ia" else vocal).append(
                (label, c["id"]))
        return {"vocal": vocal, "instrumental": instr}
    except Exception:
        return {"vocal": [], "instrumental": []}


def fetch_timeline(job_id):
    if not (job_id and MI300X_URL and MI300X_KEY):
        return None
    try:
        r = requests.get(f"{MI300X_URL}/jobs/{job_id}/timeline",
                         headers={"X-Key": MI300X_KEY}, timeout=15)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def call_backend(files, picks_vocal, picks_instr, preset_label, duration,
                 use_library, mix_mode_label, mode_choice, vocals_choice,
                 advanced_on, custom_prompt, custom_arc, seed, lufs_label,
                 oauth_profile: gr.OAuthProfile | None = None,
                 progress=gr.Progress()):
    # Merge both sample pools into one ID list. Backend accepts CSV.
    sample_picks = list(picks_vocal or []) + list(picks_instr or [])
    # Always clear stale outputs on error so prior render's plot/table doesn't linger.
    EMPTY = (None, "", None, None, [], None)
    progress(0.02, desc="validating")
    if not MI300X_URL or not MI300X_KEY:
        return None, "**Error:** backend not configured.", None, None, [], None
    sample_ids = list(sample_picks or [])
    n_uploads = len(files or [])
    n_total = n_uploads + len(sample_ids)
    if n_total == 0:
        return (None,
                f"**Error:** upload {MIN_CLIPS}–{MAX_CLIPS} clips OR pick "
                f"sample clips from the dropdown.",
                None, None, [], None)
    samples_only = (n_uploads == 0 and len(sample_ids) > 0)
    if samples_only:
        if len(sample_ids) < 6:
            return (None,
                    f"**Error:** with samples-only, pick at least "
                    f"**6 clips** (got {len(sample_ids)}).",
                    None, None, [], None)
        if int(duration) < 180:
            return (None,
                    f"**Error:** with samples-only, mix length must be "
                    f"≥ **3:00** (180s). Current: {int(duration)}s.",
                    None, None, [], None)
    if not (MIN_CLIPS <= n_total <= MAX_CLIPS):
        return (None,
                f"**Error:** need {MIN_CLIPS}–{MAX_CLIPS} clips total "
                f"(have {n_uploads} upload + {len(sample_ids)} sample).",
                None, None, [], None)
    errs, oversized = _validate_uploads(files)
    if errs:
        return (None, "**Error:** invalid uploads:\n\n- " + "\n- ".join(errs),
                None, None, [], None)
    if oversized:
        return (None,
                f"**Error:** files exceed {MAX_FILE_MB} MB cap:\n\n- "
                + "\n- ".join(oversized)
                + f"\n\n_Re-export at lower bitrate or trim length. "
                f"WAV at 44.1kHz stereo ≈ 10MB/min._",
                None, None, [], None)

    slug = PRESETS[preset_label][0]
    multipart = []
    for f in (files or []):
        path = f.name if hasattr(f, "name") else f
        multipart.append(("files", (Path(path).name, open(path, "rb"), "audio/wav")))

    mix_mode_resolved = (mix_mode_label or "").lower().strip() or (
        "tight" if not use_library else "balanced")
    if mix_mode_resolved not in ("tight", "balanced", "exploratory"):
        mix_mode_resolved = "balanced"
    data = {
        "preset": slug,
        "style": slug,
        "duration": str(int(duration)),
        "mix_mode": mix_mode_resolved,
        "use_library": "true" if use_library else "false",
        "lufs": str(LUFS_OPTIONS.get(lufs_label, -9)),
        "export_format": "mp3",
        "mode": (mode_choice or "dj_set").lower(),
        "vocals": (vocals_choice or "on").lower(),
    }
    if sample_ids:
        data["sample_clip_ids"] = ",".join(sample_ids)
    # When ONLY sample clips selected, requests still needs `files=` for
    # multipart parsing on FastAPI side. Add a tiny zero-length sentinel
    # only if nothing else; backend treats empty payload as skipped.
    if not multipart:
        # FastAPI's `files: list[UploadFile] = File(...)` requires at least one
        # file part in the multipart body. Send a zero-byte placeholder that
        # backend filters out via empty-file check.
        multipart.append(("files", ("__none__.wav", b"", "audio/wav")))
    if advanced_on:
        if custom_prompt and custom_prompt.strip():
            data["prompt"] = custom_prompt.strip()
        if custom_arc:
            data["arc"] = custom_arc
        if seed is not None and str(seed).strip() != "":
            try:
                data["seed"] = str(int(seed))
            except Exception:
                pass

    headers = {"X-Key": MI300X_KEY}
    if oauth_profile is not None:
        headers["X-User-Id"] = str(getattr(oauth_profile, "username", "anon"))[:64]
    progress(0.10, desc="uploading clips to GPU")
    # Progress ticker: spin background thread that updates progress bar with
    # rotating stage labels + real elapsed seconds while the blocking POST
    # is in flight. User sees activity instead of stuck-at-10%.
    import threading as _th
    import time as _t
    _done = _th.Event()
    _stages = [
        (0.12, "uploading + decoding"),
        (0.20, "demucs stem separation"),
        (0.32, "beat + key analysis"),
        (0.42, "CLAP semantic embedding"),
        (0.52, "Director LLM planning"),
        (0.62, "beam search timeline"),
        (0.72, "executing transitions"),
        (0.82, "phrase + stem alignment"),
        (0.88, "mastering + LUFS norm"),
    ]
    _est_sec = max(45.0, int(duration) * 0.55)
    _t0 = _t.time()
    def _tick():
        while not _done.is_set():
            elapsed = _t.time() - _t0
            frac = min(0.90, 0.10 + (elapsed / _est_sec) * 0.78)
            label = "rendering"
            for f, lbl in _stages:
                if frac >= f:
                    label = lbl
            try:
                progress(frac, desc=f"{label} · {int(elapsed)}s")
            except Exception:
                pass
            if _done.wait(2.5):
                break
    _ticker = _th.Thread(target=_tick, daemon=True)
    _ticker.start()
    try:
        r = requests.post(f"{MI300X_URL}/generate", data=data, files=multipart,
                          headers=headers, timeout=BACKEND_TIMEOUT_SEC)
    except requests.exceptions.RequestException as e:
        _done.set()
        return None, f"**Error:** backend unreachable. _{e}_", None, None, [], None
    _done.set()  # stop ticker now that POST returned
    if r.status_code == 503:
        return None, "**Server busy** — retry in ~2 min.", None, None, [], None
    if r.status_code == 504:
        return (None,
                f"**Job timed out** (>{BACKEND_TIMEOUT_SEC // 60} min). "
                f"Try shorter mix length or fewer clips.",
                None, None, [], None)
    if r.status_code == 401:
        try:
            detail = r.json().get("detail", "")
        except Exception:
            detail = ""
        return (None,
                f"🔒 **Sign-in required.** {detail}".strip(),
                None, None, [], None)
    if r.status_code == 429:
        try:
            detail = r.json().get("detail", "Daily limit reached.")
        except Exception:
            detail = "Daily limit reached."
        return None, f"⏳ **{detail}**", None, None, [], None
    if r.status_code == 400:
        try:
            detail = r.json().get("detail", r.text[:300])
        except Exception:
            detail = r.text[:300]
        return None, f"**Validation error 400:** {detail}", None, None, [], None
    if r.status_code != 200:
        return None, f"**Error {r.status_code}:** {r.text[:300]}", None, None, [], None

    progress(0.92, desc="rendering")
    job_ref = r.headers.get("X-Job-Id", "")
    out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    out.write(r.content)
    out.close()

    summary_lines = [
        f"**job** `{job_ref[:8]}` · **preset** {preset_label} · "
        f"**length** {int(duration)}s · **mode** {mix_mode_resolved}",
    ]
    try:
        if r.headers.get("X-Probe"):
            p = json.loads(r.headers["X-Probe"])
            sev = p.get("overall_severity", 0) or 0
            badge = "✓" if sev < 0.3 else ("~" if sev < 0.6 else "!")
            summary_lines.append(
                f"**probe** {badge} `{p.get('verdict', '?')}` severity {sev:.2f} "
                f"across {p.get('n_junctions', 0)} junctions"
            )
        if r.headers.get("X-Critic"):
            c = json.loads(r.headers["X-Critic"])
            cs = c.get("score")
            if cs is not None:
                summary_lines.append(f"**critic** {cs:.2f}")
        if r.headers.get("X-Clips-Used"):
            cu = json.loads(r.headers["X-Clips-Used"])
            summary_lines.append(
                f"**pool** {cu.get('user_count', 0)} user clips + "
                f"{cu.get('library_count', 0)} library clips"
            )
        warn = r.headers.get("X-Ingest-Warnings")
        if warn:
            summary_lines.append(f"_ingest warning: {warn}_")
    except Exception:
        pass
    summary_md = "\n\n".join(summary_lines)

    progress(0.98, desc="building timeline")
    timeline_json = fetch_timeline(job_ref)
    plot = render_timeline_plot(timeline_json)
    rows = render_segments_dataframe(timeline_json)
    audiobox_json = None
    try:
        if r.headers.get("X-Audiobox"):
            audiobox_json = json.loads(r.headers["X-Audiobox"])
    except Exception:
        pass
    audiobox_fig = render_audiobox_plot(audiobox_json)
    waveform_fig = render_colored_waveform(out.name, timeline_json)
    return (out.name, summary_md, waveform_fig, plot, rows, audiobox_fig)


def backend_status_md():
    if not MI300X_URL:
        return "○ backend not configured"
    try:
        r = requests.get(f"{MI300X_URL}/health", timeout=4)
        if not r.ok:
            return f"● **backend down** (HTTP {r.status_code})"
        h = r.json()
        inflight = h.get("inflight_count", 0)
        cap = h.get("inflight_max", 1)
        gpu_busy = h.get("gpu_busy", False)
        dot = "●" if gpu_busy else "○"
        return (f"{dot} **live** · queue {inflight}/{cap} · "
                f"gpu {'active' if gpu_busy else 'idle'} · "
                f"disk {h.get('disk_free_gb', '?')} GB free")
    except Exception:
        return "● **unreachable**"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.gradio-container { font-family: 'Inter', system-ui, sans-serif !important; }
.gradio-container code, .gradio-container pre {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}
/* Sample-clip CheckboxGroup: scrollable inside its own panel so the page
   stays anchored. Targets the inner wrap-block of the component. */
.aij-pool-scroll {max-height: 460px; overflow-y: auto !important;}
.aij-pool-scroll .wrap, .aij-pool-scroll > div {
    max-height: 460px !important; overflow-y: auto !important;
}
.aij-pick-hint {color: #94a3b8 !important; font-size: 12px !important;}
/* Wider canvas + better column distribution. Default Gradio container is
   narrow; bump max-width and force two-col layout to actually breathe. */
.gradio-container {max-width: 1500px !important;}
.aij-card-output {padding: 12px 0;}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.pink,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#0a0c10",
    body_background_fill_dark="#0a0c10",
    background_fill_primary="#0f1218",
    background_fill_primary_dark="#0f1218",
    background_fill_secondary="#0a0c10",
    background_fill_secondary_dark="#0a0c10",
    button_primary_background_fill="#ff2d6f",
    button_primary_background_fill_hover="#ff4d87",
    button_primary_text_color="white",
    block_border_width="0px",
    block_shadow="none",
    block_radius="6px",
)


with gr.Blocks(title="AiJockey", theme=THEME, css=CSS) as app:
    gr.Markdown("# AiJockey\n"
                "_open-source AI DJ — stems, beats, CLAP, planner, mastered output._  \n"
                "Running on AMD MI300X · AGPL-3.0")

    with gr.Tab("Demo"):
        gr.Markdown(
            "### Pre-baked mixes\n"
            "_Each mix below is a real render from the same backend. "
            "Same pipeline behind every Try It render._")
        # Fetch featured/ from backend at module-load time. Falls back to
        # static demo_mp3/ in Space repo if backend offline.
        featured_clips = []
        try:
            r = requests.get(f"{MI300X_URL}/featured", timeout=6)
            if r.ok:
                featured_clips = r.json().get("clips", [])
        except Exception:
            pass

        if featured_clips:
            for c in featured_clips:
                stem = c.get("name", "demo")
                pretty_name = stem.replace("_", " ").title()
                url = f"{MI300X_URL}{c['url']}"
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            f"#### {pretty_name}\n"
                            f"_{c['duration_sec']:.0f}s · {c['size_mb']:.1f} MB_")
                    with gr.Column(scale=2):
                        gr.Audio(url, label=pretty_name,
                                 interactive=False, autoplay=False,
                                 show_download_button=True)
        else:
            # Fallback to baked-into-space demo_mp3 dir
            for label, slug, desc in DEMO_MIXES:
                mp3 = DEMO_DIR / f"{slug}.mp3"
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"#### {label}\n{desc}")
                    with gr.Column(scale=2):
                        if mp3.exists():
                            gr.Audio(str(mp3), label=label,
                                     interactive=False, autoplay=False,
                                     show_download_button=True,
                                     type="filepath")
                        else:
                            gr.Markdown(
                                f"⚠ _missing demo file `{slug}.mp3`._")

    with gr.Tab("Try It"):
        gr.Markdown(
            "> **Disclaimer.** AiJockey is a research demo, provided as-is. "
            "The AI is experimental and the hackathon team accepts no "
            "warranty or liability for its output. Pre-staged sample clips "
            "marked _[research only]_ are copyrighted material from their "
            "original artists; they are included strictly for academic "
            "evaluation under fair-use principles. Do **not** redistribute "
            "or commercially use any rendered mix that includes those "
            "clips. By uploading audio you confirm you hold the necessary "
            "rights. AGPL-3.0; forks must remain open-source."
        )
        with gr.Row():
            login_btn = gr.LoginButton(value="Sign in with Hugging Face",
                                        size="sm")
            who_md = gr.Markdown(
                "**Sign-in required** — 1 free render per user per day.")
        with gr.Row():
            status_md = gr.Markdown(backend_status_md())
        # Auto-refresh status every 10 seconds (gradio 5 Timer).
        try:
            status_timer = gr.Timer(10)
            status_timer.tick(backend_status_md, None, status_md)
        except Exception:
            pass

        # Render panel — hidden until OAuth login completes.
        signin_notice = gr.Markdown(
            "🔒 **Sign in with Hugging Face above to access the render form.** "
            "Each user gets 1 free render per 24 hours.",
            visible=True)

        with gr.Group(visible=False) as render_panel:
          with gr.Row(variant="panel", equal_height=False):
            # Left column: controls
            with gr.Column(scale=5, min_width=440):
                gr.Markdown("### 1 · Clips")
                files = gr.File(file_count="multiple", file_types=["audio"],
                                label=f"upload {MIN_CLIPS}–{MAX_CLIPS} audio files (≤75 MB each)")
                duration_info = gr.Markdown(
                    f"_0 clips · upload to compute pool stats_")
                _initial_pools = fetch_sample_clips()
                gr.Markdown(
                    "**OR pick from droplet sample clips** (skip upload). "
                    "Mix freely from both pools. Lists are scrollable.",
                    elem_classes="aij-pick-hint")
                with gr.Accordion(
                        "vocal-heavy clips (Despacito, Taki Taki, "
                        "All The Stars, Waka Waka, Kanye, Weeknd)",
                        open=True):
                    sample_picks_vocal = gr.CheckboxGroup(
                        choices=_initial_pools["vocal"],
                        label="",
                        elem_classes="aij-pool-scroll")
                with gr.Accordion(
                        "instrumental clips · 40 from Internet Archive · CC",
                        open=True):
                    sample_picks_instr = gr.CheckboxGroup(
                        choices=_initial_pools["instrumental"],
                        label="",
                        elem_classes="aij-pool-scroll")
                refresh_samples = gr.Button(
                    "refresh sample list", size="sm", variant="secondary")

                def _refresh():
                    p = fetch_sample_clips()
                    return (gr.update(choices=p["vocal"]),
                            gr.update(choices=p["instrumental"]))
                refresh_samples.click(
                    _refresh, None,
                    [sample_picks_vocal, sample_picks_instr])

                gr.Markdown("### 2 · Output style")
                with gr.Row():
                    mode_choice = gr.Radio(
                        choices=["mashup", "dj_set"],
                        value="dj_set", label="mode",
                        info="mashup: polished smooth · dj_set: festival peaks + accents")
                    vocals_choice = gr.Radio(
                        choices=["on", "off"],
                        value="on", label="vocals",
                        info="on: full mix · off: instrumental")

                gr.Markdown("### 3 · Vibe")
                preset = gr.Dropdown(list(PRESETS), value="Festival Inferno",
                                     label="style preset")
                with gr.Row():
                    lufs = gr.Dropdown(list(LUFS_OPTIONS), value="club (-9)",
                                       label="loudness")
                    duration = gr.Slider(
                        minimum=MIN_DURATION, maximum=MAX_DURATION_HARD,
                        value=120, step=5,
                        label="length (seconds)")

                gr.Markdown("### 4 · Library blend")
                with gr.Row():
                    use_library = gr.Checkbox(
                        value=False, label="use sample library")
                    mix_mode = gr.Radio(
                        choices=["tight", "balanced", "exploratory"],
                        value="balanced", label="mode",
                        info="tight=user only · balanced=30% library · exploratory=max")

                with gr.Accordion("Advanced overrides", open=False):
                    advanced_on = gr.Checkbox(value=False, label="enable")
                    custom_prompt = gr.Textbox(
                        label="custom prompt",
                        placeholder="dark techno warehouse, driving kick")
                    custom_arc = gr.Dropdown(ARCS, value=None, label="arc")
                    seed = gr.Number(label="seed", precision=0)

                files.change(update_duration_info, [files, use_library],
                             duration_info)
                use_library.change(update_duration_info, [files, use_library],
                                   duration_info)

                with gr.Row():
                    load_demo = gr.Button("load demo preset",
                                           variant="secondary", size="sm")
                    generate = gr.Button("Generate Mix",
                                         variant="primary", size="lg")

            # Right column: output
            with gr.Column(scale=7, min_width=560):
                gr.Markdown("### Output")
                out_audio = gr.Audio(label="mastered mix", autoplay=True,
                                     type="filepath",
                                     show_download_button=True,
                                     interactive=False)
                summary = gr.Markdown(
                    "_Output, quality scores, and timeline appear here after render._")

                gr.Markdown("### Waveform (color = source clip)")
                colored_waveform = gr.Plot(label="")

                gr.Markdown("### Source-attributed timeline")
                timeline_plot = gr.Plot(label="segments")
                segments_df = gr.Dataframe(
                    headers=["start", "duration", "source", "transition"],
                    datatype=["str", "str", "str", "str"],
                    label="sequence",
                    interactive=False, wrap=True, row_count=(0, "dynamic"))

                gr.Markdown(
                    "### Audiobox aesthetics — reference-free quality (0–10)\n"
                    "_Meta's pretrained quality model. Renders as 4-bar chart "
                    "after generation. If empty: model not loaded on droplet "
                    "(install via `pip install audiobox-aesthetics` or set "
                    "`AIJOCKEY_AUDIOBOX_AESTHETICS=0`)._")
                audiobox_plot = gr.Plot(label="4-axis quality score")

        generate.click(call_backend,
                       [files, sample_picks_vocal, sample_picks_instr,
                        preset, duration, use_library,
                        mix_mode, mode_choice, vocals_choice,
                        advanced_on, custom_prompt, custom_arc, seed, lufs],
                       [out_audio, summary, colored_waveform,
                        timeline_plot, segments_df, audiobox_plot],
                       concurrency_id="gpu_job", concurrency_limit=4)

        def _load_demo_preset():
            """One-click demo: pick first 4 sample clips + dj_set +
            tomorrowland + festival_inferno + 180s + library balanced."""
            pools = fetch_sample_clips()
            ids = [cid for _, cid in pools["vocal"]][:4]
            return (
                gr.update(value=ids),                       # sample_picks
                gr.update(value="Festival Inferno"),        # preset
                gr.update(value=180),                        # duration
                gr.update(value=True),                       # use_library
                gr.update(value="balanced"),                # mix_mode
                gr.update(value="dj_set"),                  # mode_choice
                gr.update(value="on"),                      # vocals_choice
                gr.update(value=True),                       # advanced_on
                gr.update(value="tomorrowland"),            # custom_arc
            )
        load_demo.click(
            _load_demo_preset, None,
            [sample_picks_vocal, preset, duration, use_library, mix_mode,
             mode_choice, vocals_choice, advanced_on, custom_arc])

        # Sign-in surfacing + form gating + quota check.
        def show_who(profile: gr.OAuthProfile | None):
            if profile is None:
                return ("**Sign in required** — 1 free render / user / day.",
                        gr.update(visible=True),   # signin_notice
                        gr.update(visible=False))  # render_panel
            # Quota pre-check
            quota_msg = ""
            try:
                qr = requests.get(f"{MI300X_URL}/quota",
                                  headers={"X-Key": MI300X_KEY,
                                           "X-User-Id": profile.username},
                                  timeout=5)
                if qr.ok:
                    q = qr.json()
                    if not q.get("allowed"):
                        secs = q.get("retry_after_sec", 0)
                        h = int(secs / 3600)
                        m = int((secs % 3600) / 60)
                        quota_msg = (f"  ·  ⚠ daily limit used — try again "
                                     f"in {h}h{m}m")
            except Exception:
                pass
            return (f"signed in as **{profile.username}**{quota_msg}",
                    gr.update(visible=False),  # signin_notice hidden
                    gr.update(visible=True))   # render_panel shown
        app.load(show_who, None, [who_md, signin_notice, render_panel])

    with gr.Tab("How it works"):
        gr.HTML("""
<style>
.aij-flow {display:flex; gap:8px; margin:18px 0; flex-wrap:wrap;
           font-family:ui-monospace,monospace; font-size:12px;}
.aij-flow .step {flex:1; min-width:140px; padding:14px;
                 background:linear-gradient(180deg,#1a1d2440,#0f1218a0);
                 border:1px solid #1f2229; border-radius:6px;
                 border-top:2px solid #ff2d6f;}
.aij-flow .step h4 {margin:0 0 8px 0; color:#f1f5f9; font-size:14px;
                    font-family:'Inter',sans-serif; letter-spacing:0.04em;}
.aij-flow .step .sub {color:#94a3b8; font-size:11px; line-height:1.6;}
.aij-flow .arr {color:#ff2d6f; font-size:24px; align-self:center;
                font-weight:700;}
.aij-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
           gap:16px; margin:18px 0;}
.aij-card {padding:16px; border:1px solid #1f2229; border-radius:6px;
           background:#0f1218; }
.aij-card h4 {color:#22d3ee; font-size:12px; letter-spacing:0.16em;
              text-transform:uppercase; margin:0 0 8px 0;}
.aij-card .v {color:#f5f6fa; font-size:13px; line-height:1.7;}
.aij-card code {background:#1a1d24; padding:1px 6px; border-radius:3px;
                font-size:11px; color:#facc15;}
.aij-mode-table {width:100%; border-collapse:collapse; margin:12px 0;
                 font-size:13px;}
.aij-mode-table th {text-align:left; padding:8px 14px 8px 0;
                    color:#94a3b8; font-size:11px; letter-spacing:0.1em;
                    text-transform:uppercase; font-weight:500;
                    border-bottom:1px solid #1f2229;}
.aij-mode-table td {padding:8px 14px 8px 0; color:#e2e8f0;
                    border-bottom:1px solid #14161c;}
.aij-mode-table .modecol {color:#ff2d6f; font-weight:600;}
</style>

<h3 style='color:#f5f6fa; margin-top:0;'>The pipeline</h3>
<div class='aij-flow'>
  <div class='step'><h4>1 · Analyze</h4>
    <div class='sub'>Demucs stem separation (vocals / drums / bass / other).
    librosa beats + key + structural segmentation. CLAP 512-d semantic
    embedding per clip on MI300X.</div></div>
  <div class='arr'>→</div>
  <div class='step'><h4>2 · Plan</h4>
    <div class='sub'>Director LLM produces narrative + per-junction
    transition tier intents. Beam search over clip pool with arc-shape +
    text-prompt + Camelot key + BPM compatibility scoring.</div></div>
  <div class='arr'>→</div>
  <div class='step'><h4>3 · Execute</h4>
    <div class='sub'>15+ transition techniques: cut, crossfade, eq_swap,
    filter_fade, drum_break, mashup, stem_swap, echo_out, spinback,
    pitch_bend, loop_tighten, scratch, more. Phrase-quantized,
    stem-aware overlap.</div></div>
  <div class='arr'>→</div>
  <div class='step'><h4>4 · Master</h4>
    <div class='sub'>HP30 → multiband + glue compression → optional tape
    saturation → LUFS norm → true-peak limiter. Adaptive — eases
    glue-comp when integrated loudness already high.</div></div>
</div>

<h3 style='color:#f5f6fa;'>Modes — output style</h3>
<table class='aij-mode-table'>
<thead><tr><th>knob</th><th>mashup</th><th>dj_set</th></tr></thead>
<tbody>
<tr><td class='modecol'>arc</td><td>build</td><td>tomorrowland (multi-peak)</td></tr>
<tr><td class='modecol'>tape sat</td><td>off</td><td>on (drive 0.3)</td></tr>
<tr><td class='modecol'>min minor %</td><td>≥85% smooth</td><td>≥65% (more majors)</td></tr>
<tr><td class='modecol'>vocal guard</td><td>strict 0.30</td><td>looser 0.40</td></tr>
<tr><td class='modecol'>segment len</td><td>≥28s, no cap</td><td>18–28s rotation</td></tr>
<tr><td class='modecol'>callbacks</td><td>2</td><td>4</td></tr>
<tr><td class='modecol'>reuse cooldown</td><td>5 entries</td><td>1 entry</td></tr>
<tr><td class='modecol'>vibe</td><td>Polished. Long camps. Studio-quality background.</td>
                          <td>Festival peaks. Variety. Live-mix dramatic.</td></tr>
</tbody>
</table>

<h3 style='color:#f5f6fa;'>Library augmentation</h3>
<div class='aij-grid'>
  <div class='aij-card'><h4>tight</h4>
    <div class='v'>User clips only. Smallest pool, tightest narrative,
    fastest analyze.</div></div>
  <div class='aij-card'><h4>balanced</h4>
    <div class='v'>Adds library clips for ~30% headroom. Director can
    use them as bridges or warm-up. Default.</div></div>
  <div class='aij-card'><h4>exploratory</h4>
    <div class='v'>Aggressive blend up to 12 library clips. Best when
    user pool is small / monotone.</div></div>
</div>

<h3 style='color:#f5f6fa;'>Auto-quality signals</h3>
<div class='aij-grid'>
  <div class='aij-card'><h4>probe</h4>
    <div class='v'>Numpy probes per junction: RMS-envelope mismatch,
    vocal-bleed cross-correlation, spectral-phasing diff. Cheap, deterministic.
    Catches ~70% of audible artifacts.</div></div>
  <div class='aij-card'><h4>CriticV2</h4>
    <div class='v'>Trained classifier head over CLAP embeddings + DSP
    features. Confidence 0–1.</div></div>
  <div class='aij-card'><h4>Audiobox aesthetics</h4>
    <div class='v'>Meta's pretrained reference-free quality model. Four
    axes: <code>PQ</code> production quality, <code>PC</code> production
    complexity, <code>CE</code> content enjoyment, <code>CU</code> content
    usefulness. 0–10.</div></div>
</div>

<h3 style='color:#f5f6fa;'>Stack</h3>
<div class='aij-grid'>
  <div class='aij-card'><h4>compute</h4>
    <div class='v'>AMD MI300X (192 GB HBM, gfx942) on AMD Developer Cloud.
    ROCm 6 + PyTorch 2.3. Up to 4 jobs in parallel; GPU stages serialize
    via mutex.</div></div>
  <div class='aij-card'><h4>models</h4>
    <div class='v'>Demucs htdemucs_ft, LAION CLAP, Beat-This!,
    Qwen2.5-7B-Instruct (Director), Audiobox aesthetics.</div></div>
  <div class='aij-card'><h4>frontend</h4>
    <div class='v'>Gradio 5 on HF Spaces (free CPU tier). HF OAuth gate.
    1 render / user / 24h. ngrok stable tunnel to MI300X.</div></div>
  <div class='aij-card'><h4>license</h4>
    <div class='v'>AGPL-3.0. Forks (including SaaS) must remain open
    source under same license.</div></div>
</div>

<p style='margin-top:24px; color:#94a3b8; font-size:12px;'>
<a href='https://github.com/architagrawal/aiJockey' target='_blank'
   style='color:#22d3ee;'>github.com/architagrawal/aiJockey</a>
</p>
""")


if __name__ == "__main__":
    app.queue(max_size=20, default_concurrency_limit=4).launch()
