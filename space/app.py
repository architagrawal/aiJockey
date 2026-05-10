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


def fetch_sample_clips() -> list[tuple[str, str]]:
    """Return [(label, id)] of pre-staged user clips on the droplet.

    Used to populate the 'pick from sample set' checkbox group so users
    can render without uploading 50MB-each WAV files.
    """
    if not (MI300X_URL and MI300X_KEY):
        return []
    try:
        r = requests.get(f"{MI300X_URL}/sample_clips",
                         headers={"X-Key": MI300X_KEY}, timeout=8)
        if not r.ok:
            return []
        clips = r.json().get("clips", [])
        return [(f"{c['name']} ({c['duration_sec']:.0f}s, "
                 f"{c['size_mb']:.1f} MB)", c["id"]) for c in clips]
    except Exception:
        return []


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


def call_backend(files, sample_picks, preset_label, duration, use_library,
                 mix_mode_label, mode_choice, vocals_choice,
                 advanced_on, custom_prompt, custom_arc, seed, lufs_label,
                 oauth_profile: gr.OAuthProfile | None = None,
                 progress=gr.Progress()):
    # Always clear stale outputs on error so prior render's plot/table doesn't linger.
    EMPTY = (None, "", None, [], None)
    progress(0.02, desc="validating")
    if not MI300X_URL or not MI300X_KEY:
        return None, "**Error:** backend not configured.", None, [], None
    sample_ids = list(sample_picks or [])
    n_uploads = len(files or [])
    n_total = n_uploads + len(sample_ids)
    if n_total == 0:
        return (None,
                f"**Error:** upload {MIN_CLIPS}–{MAX_CLIPS} clips OR pick "
                f"sample clips from the dropdown.",
                None, [], None)
    if not (MIN_CLIPS <= n_total <= MAX_CLIPS):
        return (None,
                f"**Error:** need {MIN_CLIPS}–{MAX_CLIPS} clips total "
                f"(have {n_uploads} upload + {len(sample_ids)} sample).",
                None, [], None)
    errs, oversized = _validate_uploads(files)
    if errs:
        return (None, "**Error:** invalid uploads:\n\n- " + "\n- ".join(errs),
                None, [], None)
    if oversized:
        return (None,
                f"**Error:** files exceed {MAX_FILE_MB} MB cap:\n\n- "
                + "\n- ".join(oversized)
                + f"\n\n_Re-export at lower bitrate or trim length. "
                f"WAV at 44.1kHz stereo ≈ 10MB/min._",
                None, [], None)

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
    try:
        r = requests.post(f"{MI300X_URL}/generate", data=data, files=multipart,
                          headers=headers, timeout=BACKEND_TIMEOUT_SEC)
    except requests.exceptions.RequestException as e:
        return None, f"**Error:** backend unreachable. _{e}_", None, [], None
    if r.status_code == 503:
        return None, "**Server busy** — retry in ~2 min.", None, [], None
    if r.status_code == 504:
        return (None,
                f"**Job timed out** (>{BACKEND_TIMEOUT_SEC // 60} min). "
                f"Try shorter mix length or fewer clips.",
                None, [], None)
    if r.status_code == 401:
        return None, "**Auth failed** — backend key mismatch.", None, [], None
    if r.status_code == 400:
        try:
            detail = r.json().get("detail", r.text[:300])
        except Exception:
            detail = r.text[:300]
        return None, f"**Validation error 400:** {detail}", None, [], None
    if r.status_code != 200:
        return None, f"**Error {r.status_code}:** {r.text[:300]}", None, [], None

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
    return out.name, summary_md, plot, rows, audiobox_fig


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
        gr.Markdown("Pre-baked mixes from the same engine.")
        for label, slug, desc in DEMO_MIXES:
            with gr.Row(equal_height=True):
                gr.Markdown(f"### {label}\n_{desc}_")
                mp3 = DEMO_DIR / f"{slug}.mp3"
                if mp3.exists():
                    gr.Audio(str(mp3), label="", interactive=False,
                             show_download_button=True)
                else:
                    gr.Markdown(f"_pending render: `{slug}.mp3`_")

    with gr.Tab("Try It"):
        with gr.Row():
            login_btn = gr.LoginButton(value="Sign in with Hugging Face",
                                        size="sm")
            who_md = gr.Markdown("_anonymous · sign in for queue fairness_")
        with gr.Row():
            status_md = gr.Markdown(backend_status_md())
        # Auto-refresh status every 10 seconds (gradio 5 Timer).
        try:
            status_timer = gr.Timer(10)
            status_timer.tick(backend_status_md, None, status_md)
        except Exception:
            pass

        with gr.Row(variant="panel"):
            # Left column: controls
            with gr.Column(scale=2, min_width=320):
                gr.Markdown("### 1 · Clips")
                files = gr.File(file_count="multiple", file_types=["audio"],
                                label=f"upload {MIN_CLIPS}–{MAX_CLIPS} audio files (≤75 MB each)")
                duration_info = gr.Markdown(
                    f"_0 clips · upload to compute pool stats_")
                sample_picks = gr.CheckboxGroup(
                    choices=fetch_sample_clips(),
                    label="OR pick from droplet sample clips (skip upload)",
                    info="Server-side files in /workspace/user_set/. "
                         "Faster than uploading WAVs.")
                refresh_samples = gr.Button("refresh sample list", size="sm",
                                             variant="secondary")
                refresh_samples.click(
                    lambda: gr.update(choices=fetch_sample_clips()),
                    None, sample_picks)

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

                generate = gr.Button("Generate Mix", variant="primary",
                                     size="lg")

            # Right column: output
            with gr.Column(scale=3, min_width=400):
                gr.Markdown("### Output")
                out_audio = gr.Audio(label="mastered mix", autoplay=True,
                                     type="filepath",
                                     show_download_button=True,
                                     interactive=False,
                                     waveform_options={"waveform_color": "#ff2d6f",
                                                       "waveform_progress_color": "#22d3ee"})
                summary = gr.Markdown(
                    "_Output, quality scores, and timeline appear here after render._")

                gr.Markdown("### Source-attributed timeline")
                timeline_plot = gr.Plot(label="segments")
                segments_df = gr.Dataframe(
                    headers=["start", "duration", "source", "transition"],
                    datatype=["str", "str", "str", "str"],
                    label="sequence",
                    interactive=False, wrap=True, row_count=(0, "dynamic"))

                gr.Markdown("### Audiobox aesthetics — reference-free quality (0–10)")
                audiobox_plot = gr.Plot(label="4-axis quality score")

        generate.click(call_backend,
                       [files, sample_picks, preset, duration, use_library,
                        mix_mode, mode_choice, vocals_choice,
                        advanced_on, custom_prompt, custom_arc, seed, lufs],
                       [out_audio, summary, timeline_plot, segments_df,
                        audiobox_plot],
                       concurrency_id="gpu_job", concurrency_limit=4)

        # Sign-in surfacing: show "signed in as X" when user has logged in.
        # gradio auto-injects gr.OAuthProfile when fn signature requests it.
        def show_who(profile: gr.OAuthProfile | None):
            if profile is None:
                return "_anonymous · sign in for queue fairness_"
            return f"signed in as **{profile.username}**"
        app.load(show_who, None, who_md)

    with gr.Tab("How it works"):
        gr.Markdown("""
### Pipeline

1. **Analyze** — Demucs stems, librosa beats and key, structural segmentation,
   CLAP semantic embedding per clip on GPU.
2. **Plan** — Beam search over the clip pool with arc-shape and text-prompt
   bias from a Director LLM.
3. **Execute** — 15+ transition techniques: cut, eq_swap, drum_break, mashup,
   echo_out, spinback, loop_tighten, others.
4. **Master** — HP30, multiband and glue compression, LUFS normalization,
   true-peak limiter.

### Library option

When **use sample library** is checked, the planner can blend in pre-analyzed
curated clips for variety. Modes:

- **tight** — user clips only
- **balanced** — adds enough library clips to fill 30% headroom
- **exploratory** — aggressive library blend up to 12 clips

### Auto-quality signals

Each render returns a probe verdict (energy / vocal-bleed / spectral phasing
across junctions) plus an Audiobox-style critic score. Both shown in the
output summary.

[github.com/architagrawal/aiJockey](https://github.com/architagrawal/aiJockey) · AGPL-3.0
""")


if __name__ == "__main__":
    app.queue(max_size=20, default_concurrency_limit=4).launch()
