"""HF Space: AiJockey demo + password-gated live generation.

Public tab: 5 pre-baked demo mixes (MP3 in repo).
Try It tab: password gate -> upload 3-6 clips + pick preset -> calls MI300X tunnel.

Required Space secrets:
  ADMIN_PW       admin password for Try It tab
  MI300X_URL     ngrok stable URL, e.g. https://issue-slingshot-bobsled.ngrok-free.dev
  MI300X_KEY     shared secret matching SERVER_KEY on droplet
"""
from __future__ import annotations
import os, hmac, requests, tempfile
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


def check_pw(pw: str):
    ok = bool(ADMIN_PW) and hmac.compare_digest(pw, ADMIN_PW)
    return (
        gr.update(visible=ok),
        gr.update(visible=not ok),
        "" if ok else "Wrong password.",
    )


def call_backend(files, preset_label: str, duration: int):
    if not MI300X_URL or not MI300X_KEY:
        return None, "Backend not configured. Set MI300X_URL + MI300X_KEY in Space secrets."
    if not files:
        return None, "Upload 3-6 audio clips."
    if not (3 <= len(files) <= 6):
        return None, f"Need 3-6 clips, got {len(files)}."

    slug = PRESETS[preset_label][0]
    multipart = []
    for f in files:
        path = f.name if hasattr(f, "name") else f
        multipart.append(("files", (Path(path).name, open(path, "rb"), "audio/wav")))
    data = {"preset": slug, "duration": str(duration)}
    headers = {"X-Key": MI300X_KEY}

    try:
        r = requests.post(
            f"{MI300X_URL}/generate",
            data=data, files=multipart, headers=headers, timeout=600,
        )
    except requests.exceptions.RequestException as e:
        return None, f"Backend unreachable: {e}. Droplet may be destroyed (idle)."

    if r.status_code != 200:
        return None, f"Error {r.status_code}: {r.text[:300]}"

    out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    out.write(r.content); out.close()
    return out.name, f"Generated. Preset: {preset_label}. Length: {duration}s."


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
                "Stems → beats → CLAP → planner → mastered mix. AGPL-3.0.")

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
            gr.Markdown("Upload 3-6 audio clips (mp3/wav, ≤25MB each, ≤5min each).")
            files = gr.File(file_count="multiple",
                            file_types=["audio"],
                            label="Your clips (3-6)")
            preset = gr.Dropdown(list(PRESETS), value="Festival Inferno",
                                 label="Vibe preset")
            duration = gr.Radio([90, 180, 300], value=180,
                                label="Mix length (seconds)")
            health_btn = gr.Button("Check backend")
            health_out = gr.Markdown()
            generate = gr.Button("Generate (uses MI300X credits)", variant="primary")
            out_audio = gr.Audio(label="Output mix")
            status = gr.Markdown()

            health_btn.click(health_check, None, health_out)
            generate.click(call_backend, [files, preset, duration],
                           [out_audio, status])

        unlock.click(check_pw, pw, [panel, gate, err])

    with gr.Tab("How it works"):
        gr.Markdown("""
- **Analyze** — Demucs stems, madmom beats, librosa key/structure, CLAP embedding.
- **Plan** — beam search over clip pool with arc-shape + text-prompt bias.
- **Execute** — 15 transition techniques (cut, eq_swap, drum_break, mashup, …).
- **Master** — HP30 → multiband + glue compression → LUFS norm → true-peak limiter.

Repo: [github.com/architagrawal/aiJockey](https://github.com/architagrawal/aiJockey) (AGPL-3.0)
""")

if __name__ == "__main__":
    app.launch()
