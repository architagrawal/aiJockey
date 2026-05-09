"""HF Space: AiJockey demo + password-gated live generation.

Public tab: 5 pre-baked demo mixes.
Try It tab: password gate -> upload 2-8 clips, vibe preset, length, optional library, advanced.

Required Space secrets:
  ADMIN_PW       admin password for Try It tab
  MI300X_URL     ngrok stable URL, e.g. https://issue-slingshot-bobsled.ngrok-free.dev
  MI300X_KEY     shared secret matching SERVER_KEY on droplet
"""
from __future__ import annotations
import os, hmac, requests, tempfile, math
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
ARCS = ["build", "peak", "rollercoaster", "descend", "flat_high", "flat_low"]
LUFS_OPTIONS = {"streaming (-14)": -14, "club (-9)": -9, "competition (-6)": -6}

MIN_CLIPS, MAX_CLIPS = 2, 8
MIN_DURATION = 30
MAX_DURATION_HARD = 600
BACKEND_TIMEOUT_SEC = 1200  # align with API AIJOCKEY_JOB_TIMEOUT_SEC default
EXPORT_FORMATS = {"MP3 (default)": "mp3", "WAV (lossless)": "wav", "FLAC (lossless)": "flac"}


def check_pw(pw: str):
    ok = bool(ADMIN_PW) and hmac.compare_digest(pw, ADMIN_PW)
    return (
        gr.update(visible=ok),
        gr.update(visible=not ok),
        "" if ok else "Wrong password.",
    )


def _probe_total_duration(file_paths) -> float:
    """Best-effort duration sum on the Space side. Uses soundfile if available."""
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
    """Recompute duration slider max when uploads change."""
    paths = [f.name if hasattr(f, "name") else f for f in (files or [])]
    total = _probe_total_duration(paths)
    if use_library:
        max_dur = MAX_DURATION_HARD
    else:
        max_dur = max(MIN_DURATION, min(MAX_DURATION_HARD, int(total))) if total else MIN_DURATION
    default = min(180, max_dur)
    info_text = f"Uploaded total: {total:.1f}s · slider max: {max_dur}s · library: {'on' if use_library else 'off'}"
    return gr.update(minimum=MIN_DURATION, maximum=max_dur, value=default), info_text


def call_backend(files, preset_label, duration, use_library,
                 advanced_on, custom_prompt, custom_arc, seed, lufs_label,
                 export_label):
    if not MI300X_URL or not MI300X_KEY:
        return None, "Backend not configured. Set MI300X_URL + MI300X_KEY in Space secrets."
    if not files:
        return None, f"Upload {MIN_CLIPS}-{MAX_CLIPS} audio clips."
    if not (MIN_CLIPS <= len(files) <= MAX_CLIPS):
        return None, f"Need {MIN_CLIPS}-{MAX_CLIPS} clips, got {len(files)}."

    slug = PRESETS[preset_label][0]
    multipart = []
    for f in files:
        path = f.name if hasattr(f, "name") else f
        multipart.append(("files", (Path(path).name, open(path, "rb"), "audio/wav")))

    data = {
        "preset": slug,
        "duration": str(int(duration)),
        "use_library": "true" if use_library else "false",
        "lufs": str(LUFS_OPTIONS.get(lufs_label, -9)),
        "export_format": EXPORT_FORMATS.get(export_label, "mp3"),
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
    suf = {"mp3": ".mp3", "wav": ".wav", "flac": ".flac"}.get(
        EXPORT_FORMATS.get(export_label, "mp3"), ".mp3")
    try:
        r = requests.post(
            f"{MI300X_URL}/generate",
            data=data, files=multipart, headers=headers, timeout=BACKEND_TIMEOUT_SEC)
    except requests.exceptions.RequestException as e:
        return None, f"Backend unreachable: {e}. Check tunnel / AMD instance."
    if r.status_code == 503:
        return None, "Server busy (one mix at a time on GPU). Retry in ~2 min. " + r.text[:300]
    if r.status_code == 504:
        return None, f"Job exceeded ~{BACKEND_TIMEOUT_SEC // 60} min. Try shorter mix or fewer clips."
    if r.status_code != 200:
        return None, f"Error {r.status_code}: {r.text[:400]}"

    job_ref = r.headers.get("X-Job-Id", "")
    warn = r.headers.get("X-Ingest-Warnings")
    out = tempfile.NamedTemporaryFile(suffix=suf, delete=False)
    out.write(r.content)
    out.close()
    extra = f" job_id={job_ref}" if job_ref else ""
    if warn:
        extra += f"\n\nIngest note: {warn}"
    return (
        out.name,
        f"OK · preset={preset_label} · duration={int(duration)}s · export={EXPORT_FORMATS.get(export_label, 'mp3')}{extra}",
    )


def health_check():
    if not MI300X_URL:
        return "MI300X_URL not set."
    try:
        r = requests.get(f"{MI300X_URL}/health", timeout=5)
        return f"Backend: {r.json()}" if r.ok else f"Backend down ({r.status_code})"
    except Exception as e:
        return f"Backend unreachable: {e}"


with gr.Blocks(title="AiJockey", theme=gr.themes.Soft()) as app:
    gr.Markdown("# AiJockey — open-source AI DJ\n"
                "Stems → beats → CLAP → planner → mastered mix. AGPL-3.0. "
                "Running on AMD MI300X.")

    with gr.Tab("Demo"):
        gr.Markdown("Five pre-baked mixes from the same engine, different intent. "
                    "Same pool, different prompt + arc.")
        for label, (slug, desc) in PRESETS.items():
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"### {label}\n{desc}")
                with gr.Column(scale=2):
                    mp3 = DEMO_DIR / f"{slug}.mp3"
                    if mp3.exists():
                        gr.Audio(str(mp3), label=label, interactive=False)
                    else:
                        gr.Markdown(f"_Pending render: `{slug}.mp3`_")

    with gr.Tab("Try It"):
        with gr.Group(visible=True) as gate:
            gr.Markdown("### Live generation (password required)\n"
                        "Burns MI300X credits. Owner only.")
            pw = gr.Textbox(type="password", label="Admin password")
            unlock = gr.Button("Unlock")
            err = gr.Markdown()
        with gr.Group(visible=False) as panel:
            gr.Markdown(
                f"Upload **{MIN_CLIPS}–{MAX_CLIPS}** audio clips (wav/mp3/flac/m4a/ogg, **≤25 MB** each). "
                f"Target mix **≤{MAX_DURATION_HARD // 60} min** wall clock; backend may return **503** if busy (one GPU job) or **504** if generation exceeds **~{BACKEND_TIMEOUT_SEC // 60} min**."
            )
            files = gr.File(file_count="multiple", file_types=["audio"],
                            label=f"Your clips ({MIN_CLIPS}-{MAX_CLIPS})")

            with gr.Row():
                preset = gr.Dropdown(list(PRESETS), value="Festival Inferno",
                                     label="Vibe preset")
                lufs = gr.Dropdown(list(LUFS_OPTIONS), value="club (-9)",
                                   label="Loudness target")
                export_fmt = gr.Dropdown(
                    list(EXPORT_FORMATS), value="MP3 (default)", label="Download format")

            duration = gr.Slider(minimum=MIN_DURATION, maximum=MIN_DURATION,
                                 value=MIN_DURATION, step=5,
                                 label="Mix length (seconds)")
            duration_info = gr.Markdown(f"Uploaded total: 0s · slider max: {MIN_DURATION}s")

            use_library = gr.Checkbox(value=False,
                label="Use sample library (allow planner to mix in pre-curated clips for variety; optional)")

            with gr.Accordion("Advanced (optional)", open=False):
                advanced_on = gr.Checkbox(value=False, label="Enable advanced overrides")
                custom_prompt = gr.Textbox(label="Custom prompt (overrides preset)",
                                           placeholder="e.g. 'dark techno warehouse, driving kick'")
                custom_arc = gr.Dropdown(ARCS, value=None, label="Arc shape (overrides preset)")
                seed = gr.Number(label="Seed (optional, for reproducibility)", precision=0)

            files.change(update_duration_bounds, [files, use_library], [duration, duration_info])
            use_library.change(update_duration_bounds, [files, use_library], [duration, duration_info])

            with gr.Row():
                health_btn = gr.Button("Check backend")
                health_out = gr.Markdown()
            health_btn.click(health_check, None, health_out)

            generate = gr.Button("Generate (uses MI300X credits)", variant="primary")
            out_audio = gr.Audio(label="Output mix")
            status = gr.Markdown()

            generate.click(call_backend,
                           [files, preset, duration, use_library,
                            advanced_on, custom_prompt, custom_arc, seed, lufs, export_fmt],
                           [out_audio, status])

        unlock.click(check_pw, pw, [panel, gate, err])

    with gr.Tab("How it works"):
        gr.Markdown("""
- **Analyze** — Demucs stems, librosa beats/key/structure, CLAP embedding (per clip, GPU).
- **Plan** — beam search over clip pool with arc-shape + text-prompt bias.
- **Execute** — 15 transition techniques (cut, eq_swap, drum_break, mashup, …).
- **Master** — HP30 → multiband + glue compression → LUFS norm → true-peak limiter.
- **Library option** — when enabled, planner can blend in pre-analyzed curated clips for variety.

Repo: [github.com/architagrawal/aiJockey](https://github.com/architagrawal/aiJockey) (AGPL-3.0)
""")

if __name__ == "__main__":
    app.launch()
