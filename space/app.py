"""HF Space: AiJockey demo + HF-OAuth-gated live generation.

Public Demo tab: pre-baked mixes.
Try It tab: HF login (or admin password) -> upload clips -> stream mix back
            with color-coded source-attributed timeline below the audio player.

Required Space secrets:
  ADMIN_PW       admin password fallback for Try It tab
  MI300X_URL     ngrok stable URL
  MI300X_KEY     shared secret matching SERVER_KEY on droplet
"""
from __future__ import annotations
import os, hmac, html, json, requests, tempfile
from pathlib import Path
import gradio as gr

ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "demo_mp3"

ADMIN_PW    = os.environ.get("ADMIN_PW", "")
MI300X_URL  = os.environ.get("MI300X_URL", "")
MI300X_KEY  = os.environ.get("MI300X_KEY", "")

PRESETS = {
    "Festival Inferno":       ("festival_inferno",      "Western EDM peak. Big drops, anthemic."),
    "Midnight Noir":          ("midnight_noir",         "Dark cinematic, slow burn, after-hours."),
    "Neon Retrowave":         ("neon_retrowave",        "80s synthwave, driving nostalgia."),
    "East Meets Bass":        ("east_meets_bass",       "Sitar + tabla over deep electronic bass."),
    "Bollywood Block Party":  ("bollywood_block_party", "Bollywood + Punjabi + drill mash."),
}
DEMO_MIXES = [
    ("Chillstep",   "chillstep_demo",    "Lush atmospheric chillstep — 5 min"),
    ("Drum & Bass", "dnb_demo",          "Energetic dnb — 6 min"),
    ("Dubstep",     "dubstep_demo",      "Heavy wobble dubstep — 5 min"),
    ("Future Bass", "future_bass_demo",  "Bright melodic future bass — 4 min"),
]
ARCS = ["build", "peak", "rollercoaster", "descend", "flat_high", "flat_low"]
LUFS_OPTIONS = {"streaming (-14)": -14, "club (-9)": -9, "competition (-6)": -6}

MIN_CLIPS, MAX_CLIPS = 2, 8
MIN_DURATION = 30
MAX_DURATION_HARD = 600
BACKEND_TIMEOUT_SEC = 1200
LIBRARY_COLOR = "#3a3f48"
USER_PALETTE = [
    "#ff5e7e", "#48d6ff", "#ffd166", "#74f6a8",
    "#c084fc", "#f97316", "#22d3ee", "#fb7185",
]


def check_pw(pw: str):
    ok = bool(ADMIN_PW) and hmac.compare_digest(pw, ADMIN_PW)
    return (
        gr.update(visible=ok),
        gr.update(visible=not ok),
        "" if ok else "Wrong password.",
    )


def _probe_total_duration(file_paths) -> float:
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


def update_duration_bounds(files, use_library):
    paths = [f.name if hasattr(f, "name") else f for f in (files or [])]
    total = _probe_total_duration(paths)
    if use_library:
        max_dur = MAX_DURATION_HARD
    else:
        max_dur = max(MIN_DURATION, min(MAX_DURATION_HARD, int(total))) if total else MIN_DURATION
    default = min(180, max_dur)
    info_text = (f"Uploaded total: {total:.1f}s · max mix length: {max_dur}s · "
                 f"library: {'on' if use_library else 'off'}")
    return gr.update(minimum=MIN_DURATION, maximum=max_dur, value=default), info_text


def _fmt_time(sec: float) -> str:
    sec = max(0.0, sec)
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m}:{s:05.2f}"


def render_timeline_html(timeline_json: dict | None) -> str:
    """Produce a horizontal color-coded segment bar + per-segment table.

    User clips: filename + unique color from palette, indexed by upload order.
    Library clips: opaque dark gray (no source leak per spec).
    """
    if not timeline_json or not timeline_json.get("segments"):
        return ""
    segments = timeline_json["segments"]
    total = max(1e-3, float(timeline_json.get("total_duration_sec", 0.0)))
    user_color: dict[str, str] = {}
    next_user_idx = 0
    bar = []
    rows = []
    for seg in segments:
        pct = 100.0 * float(seg["duration_sec"]) / total
        label = seg.get("label") or "library"
        source = seg.get("source") or "?"
        if source == "user":
            if label not in user_color:
                user_color[label] = USER_PALETTE[next_user_idx % len(USER_PALETTE)]
                next_user_idx += 1
            color = user_color[label]
            display_label = (seg.get("filename") or label)[:36]
        else:
            color = LIBRARY_COLOR
            display_label = "library clip"
        title = (f"{_fmt_time(seg['start_sec'])} → "
                 f"{_fmt_time(seg['start_sec'] + seg['duration_sec'])}  ·  "
                 f"{display_label}  ·  {seg.get('transition', '?')}")
        bar.append(
            f'<div style="flex:0 0 {pct:.3f}%;background:{color};'
            f'border-right:1px solid #0a0a0a;height:100%;" '
            f'title="{html.escape(title)}"></div>'
        )
        rows.append({
            "start": _fmt_time(seg["start_sec"]),
            "dur": f"{seg['duration_sec']:.1f}s",
            "color": color,
            "src": source,
            "label": display_label,
            "transition": seg.get("transition", "?"),
            "tier": seg.get("tier") or "—",
        })

    bar_html = (
        '<div style="display:flex;width:100%;height:34px;border-radius:6px;'
        'overflow:hidden;background:#1a1c20;margin-bottom:8px;">'
        + "".join(bar) + "</div>"
    )

    legend_items = []
    for label, color in user_color.items():
        legend_items.append(
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'margin-right:14px;font-size:12px;color:#cbd5e1;">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'border-radius:2px;background:{color};"></span>'
            f'{html.escape(label)}</span>'
        )
    legend_items.append(
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'font-size:12px;color:#94a3b8;">'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'border-radius:2px;background:{LIBRARY_COLOR};"></span>library</span>'
    )
    legend_html = (
        '<div style="margin-bottom:10px;">' + "".join(legend_items) + "</div>"
    )

    table_rows = "".join(
        f'<tr>'
        f'<td style="padding:4px 8px;color:#9aa3ad;font-variant-numeric:tabular-nums;">'
        f'{r["start"]}</td>'
        f'<td style="padding:4px 8px;color:#9aa3ad;">{r["dur"]}</td>'
        f'<td style="padding:4px 8px;"><span style="display:inline-block;'
        f'width:10px;height:10px;border-radius:2px;background:{r["color"]};'
        f'margin-right:6px;"></span>{html.escape(r["label"])}</td>'
        f'<td style="padding:4px 8px;color:#cbd5e1;">{html.escape(r["transition"])}'
        f' <span style="color:#64748b;font-size:11px;">{html.escape(str(r["tier"]))}</span></td>'
        f'</tr>'
        for r in rows
    )
    table_html = (
        '<table style="width:100%;border-collapse:collapse;font-size:13px;'
        'color:#e2e8f0;background:#0f1115;border-radius:6px;overflow:hidden;">'
        '<thead><tr style="background:#1a1c20;color:#94a3b8;">'
        '<th style="text-align:left;padding:6px 8px;">start</th>'
        '<th style="text-align:left;padding:6px 8px;">dur</th>'
        '<th style="text-align:left;padding:6px 8px;">source</th>'
        '<th style="text-align:left;padding:6px 8px;">transition</th>'
        '</tr></thead><tbody>' + table_rows + '</tbody></table>'
    )

    return bar_html + legend_html + table_html


def fetch_timeline(job_id: str) -> dict | None:
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


def call_backend(files, preset_label, duration, use_library, mix_mode_label,
                 advanced_on, custom_prompt, custom_arc, seed, lufs_label,
                 oauth_profile: gr.OAuthProfile | None = None,
                 progress=gr.Progress()):
    progress(0.02, desc="validating...")
    if not MI300X_URL or not MI300X_KEY:
        return None, "Backend not configured.", ""
    if not files:
        return None, f"Upload {MIN_CLIPS}-{MAX_CLIPS} audio clips.", ""
    if not (MIN_CLIPS <= len(files) <= MAX_CLIPS):
        return None, f"Need {MIN_CLIPS}-{MAX_CLIPS} clips, got {len(files)}.", ""

    slug = PRESETS[preset_label][0]
    multipart = []
    for f in files:
        path = f.name if hasattr(f, "name") else f
        multipart.append(("files", (Path(path).name, open(path, "rb"), "audio/wav")))

    mix_mode_resolved = (mix_mode_label or "").lower().strip() or (
        "tight" if not use_library else "balanced")
    if mix_mode_resolved not in ("tight", "balanced", "exploratory"):
        mix_mode_resolved = "balanced"
    data = {
        "preset": slug,
        "duration": str(int(duration)),
        "mix_mode": mix_mode_resolved,
        "use_library": "true" if use_library else "false",
        "lufs": str(LUFS_OPTIONS.get(lufs_label, -9)),
        "export_format": "mp3",
    }
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
    progress(0.08, desc="uploading clips to GPU...")
    try:
        r = requests.post(f"{MI300X_URL}/generate",
                          data=data, files=multipart, headers=headers,
                          timeout=BACKEND_TIMEOUT_SEC)
    except requests.exceptions.RequestException as e:
        return None, f"Backend unreachable: {e}", ""
    if r.status_code == 503:
        return None, "Server busy — retry in ~2 min.", ""
    if r.status_code == 504:
        return None, f"Job exceeded ~{BACKEND_TIMEOUT_SEC // 60} min. Try shorter mix.", ""
    if r.status_code != 200:
        return None, f"Error {r.status_code}: {r.text[:300]}", ""

    progress(0.92, desc="rendering timeline...")
    job_ref = r.headers.get("X-Job-Id", "")
    out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    out.write(r.content)
    out.close()

    badges = [
        f"`preset:{preset_label}`",
        f"`{int(duration)}s`",
        f"`mode:{mix_mode_resolved}`",
    ]
    if job_ref:
        badges.append(f"`job:{job_ref[:8]}`")

    probe_hdr = r.headers.get("X-Probe", "")
    critic_hdr = r.headers.get("X-Critic", "")
    clips_hdr = r.headers.get("X-Clips-Used", "")
    warn = r.headers.get("X-Ingest-Warnings")
    try:
        if probe_hdr:
            p = json.loads(probe_hdr)
            sev = p.get("overall_severity", 0) or 0
            emoji = "🟢" if sev < 0.3 else ("🟡" if sev < 0.6 else "🔴")
            badges.append(f"{emoji} probe `{p.get('verdict', '?')}` "
                          f"sev={sev:.2f}")
        if critic_hdr:
            c = json.loads(critic_hdr)
            cs = c.get("score")
            if cs is not None:
                emoji = "🟢" if cs >= 0.7 else ("🟡" if cs >= 0.5 else "🔴")
                badges.append(f"{emoji} critic `{cs:.2f}`")
        if clips_hdr:
            cu = json.loads(clips_hdr)
            badges.append(f"📂 {cu.get('user_count', 0)} user + "
                          f"{cu.get('library_count', 0)} library clips")
    except Exception:
        pass

    msg = "**OK** · " + " · ".join(badges)
    if warn:
        msg += f"\n\n> ⚠️ Ingest: {warn}"

    progress(0.98, desc="fetching segment map...")
    timeline_json = fetch_timeline(job_ref)
    timeline_html = render_timeline_html(timeline_json)
    return out.name, msg, timeline_html


def backend_status_banner() -> str:
    if not MI300X_URL:
        return "🔴 **Backend not configured.** Demo tab still works."
    try:
        r = requests.get(f"{MI300X_URL}/health", timeout=4)
        if not r.ok:
            return f"🔴 **Backend down** (HTTP {r.status_code})"
        h = r.json()
        inflight = h.get("inflight_count", 0)
        cap = h.get("inflight_max", 1)
        gpu_busy = h.get("gpu_busy", False)
        gpu_lbl = "GPU busy" if gpu_busy else "GPU idle"
        return (f"🟢 **Live** — {gpu_lbl}, queue {inflight}/{cap}, "
                f"disk {h.get('disk_free_gb', '?')} GB free.")
    except Exception as e:
        return f"🔴 **Unreachable**: {str(e)[:120]}"


CSS = """
.gradio-container {background: linear-gradient(180deg,#0a0c10 0%,#11141a 100%) !important;}
.aij-hero {background: linear-gradient(135deg,#1a1c25 0%,#11141a 100%);
           border: 1px solid #232733; border-radius: 10px;
           padding: 18px 22px; margin-bottom: 14px;}
.aij-hero h1 {margin:0 0 4px 0; color:#f1f5f9; font-weight:700;}
.aij-hero p {margin:0; color:#94a3b8; font-size:14px;}
"""


with gr.Blocks(title="AiJockey",
               theme=gr.themes.Monochrome(primary_hue="pink",
                                          neutral_hue="slate"),
               css=CSS) as app:
    gr.HTML(
        '<div class="aij-hero">'
        '<h1>🎧 AiJockey</h1>'
        '<p>Open-source AI DJ. Stems → beats → CLAP → planner → mastered mix. '
        'Running on AMD MI300X · AGPL-3.0</p>'
        '</div>'
    )

    with gr.Tab("🎵 Demo"):
        gr.Markdown("Pre-baked mixes from the same engine, different intent. "
                    "Same pipeline behind every render in the Try It tab.")
        for label, slug, desc in DEMO_MIXES:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"### {label}\n{desc}")
                with gr.Column(scale=2):
                    mp3 = DEMO_DIR / f"{slug}.mp3"
                    if mp3.exists():
                        gr.Audio(str(mp3), label=label, interactive=False,
                                 show_download_button=True, autoplay=False)
                    else:
                        gr.Markdown(f"_Pending render: `{slug}.mp3`_")

    with gr.Tab("🎚️ Try It"):
        status_banner = gr.Markdown(backend_status_banner())
        refresh_status = gr.Button("Refresh status", size="sm",
                                    variant="secondary")
        refresh_status.click(backend_status_banner, None, status_banner)

        with gr.Group(visible=True) as gate:
            gr.Markdown("### Sign in to render\n"
                        "Hugging Face account preferred. Admin password works "
                        "as fallback.")
            try:
                login_btn = gr.LoginButton()
            except Exception:
                login_btn = None
            pw = gr.Textbox(type="password",
                            label="Admin password (fallback)")
            unlock = gr.Button("Unlock with password",
                               variant="secondary")
            err = gr.Markdown()
        with gr.Group(visible=False) as panel:
            gr.Markdown(
                f"Drop **{MIN_CLIPS}–{MAX_CLIPS}** clips (wav/mp3/flac/m4a/ogg, "
                f"≤25 MB each). Mix length ≤{MAX_DURATION_HARD // 60} min. "
                f"Backend serves up to 4 jobs in parallel; GPU stages serialize."
            )
            files = gr.File(file_count="multiple", file_types=["audio"],
                            label=f"Your clips ({MIN_CLIPS}-{MAX_CLIPS})")

            with gr.Row():
                preset = gr.Dropdown(list(PRESETS), value="Festival Inferno",
                                     label="Vibe preset")
                lufs = gr.Dropdown(list(LUFS_OPTIONS), value="club (-9)",
                                   label="Loudness target")

            duration = gr.Slider(minimum=MIN_DURATION, maximum=MIN_DURATION,
                                 value=MIN_DURATION, step=5,
                                 label="Mix length (seconds)")
            duration_info = gr.Markdown(
                f"Uploaded total: 0s · max mix length: {MIN_DURATION}s")

            with gr.Row():
                use_library = gr.Checkbox(value=False,
                    label="Use sample library (variety)")
                mix_mode = gr.Radio(
                    choices=["tight", "balanced", "exploratory"],
                    value="balanced",
                    label="Mix mode",
                    info="tight: user only · balanced: 30% library · exploratory: max blend")

            with gr.Accordion("Advanced (optional)", open=False):
                advanced_on = gr.Checkbox(value=False,
                                          label="Enable advanced overrides")
                custom_prompt = gr.Textbox(
                    label="Custom prompt (overrides preset)",
                    placeholder="e.g. 'dark techno warehouse, driving kick'")
                custom_arc = gr.Dropdown(ARCS, value=None,
                                         label="Arc shape (overrides preset)")
                seed = gr.Number(label="Seed (optional)", precision=0)

            files.change(update_duration_bounds, [files, use_library],
                         [duration, duration_info])
            use_library.change(update_duration_bounds, [files, use_library],
                               [duration, duration_info])

            generate = gr.Button("🎛️ Generate Mix", variant="primary",
                                 size="lg")
            out_audio = gr.Audio(label="Mastered mix",
                                 autoplay=True, type="filepath",
                                 show_download_button=True,
                                 streaming=False, interactive=False)
            status = gr.Markdown()
            timeline_view = gr.HTML(label="Timeline")

            generate.click(call_backend,
                           [files, preset, duration, use_library, mix_mode,
                            advanced_on, custom_prompt, custom_arc, seed, lufs],
                           [out_audio, status, timeline_view],
                           concurrency_id="gpu_job",
                           concurrency_limit=4)

        unlock.click(check_pw, pw, [panel, gate, err])

        def _oauth_unlock(profile: gr.OAuthProfile | None):
            if profile is None:
                return gr.update(), gr.update(), ""
            return (gr.update(visible=True), gr.update(visible=False),
                    f"Signed in as **{profile.username}**.")
        try:
            app.load(_oauth_unlock, None, [panel, gate, err])
        except Exception:
            pass

    with gr.Tab("ℹ️ How it works"):
        gr.Markdown("""
- **Analyze** — Demucs stems, librosa beats/key/structure, CLAP embedding (per clip, GPU).
- **Plan** — beam search over clip pool with arc-shape + text-prompt bias.
- **Execute** — 15+ transition techniques (cut, eq_swap, drum_break, mashup, …).
- **Master** — HP30 → multiband + glue compression → LUFS norm → true-peak limiter.
- **Library option** — when enabled, planner can blend in pre-analyzed curated clips for variety.

Repo: [github.com/architagrawal/aiJockey](https://github.com/architagrawal/aiJockey) (AGPL-3.0)
""")

if __name__ == "__main__":
    app.queue(max_size=20, default_concurrency_limit=4).launch()
