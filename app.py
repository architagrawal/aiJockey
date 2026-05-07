"""
AiJockey hackathon demo — Gradio web app.

Hosts the pipeline as a web UI. Run on MI300X (or any GPU host):

    python app.py --share

`--share` returns a temp public URL via Gradio's tunnel. Show that URL on
projector/audience devices.

Endpoints:
- Upload clips (multi-file)
- Configure: duration, lufs, classifier on/off
- Generate → progress bar → inline audio player + timeline JSON

For local-only:
    python app.py --host 0.0.0.0 --port 7860
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import shutil
import time
from pathlib import Path

# Ensure src/ on path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'src'))

import gradio as gr


CLIP_DIR = 'demo_clips'
CACHE_DIR = 'demo_cache'
OUT_DIR = 'demo_output'
SAMPLES_DIR = 'samples'


def reset_dirs() -> None:
    for d in (CLIP_DIR, CACHE_DIR, OUT_DIR):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)


def stage_uploads(files: list) -> list[str]:
    """Copy uploaded files into CLIP_DIR. Returns list of staged filenames."""
    if not files:
        return []
    os.makedirs(CLIP_DIR, exist_ok=True)
    staged: list[str] = []
    for f in files:
        # gradio passes either temp file path string or namedtuple-ish object
        src = f if isinstance(f, str) else getattr(f, 'name', None)
        if src is None:
            continue
        fname = os.path.basename(src)
        dst = os.path.join(CLIP_DIR, fname)
        shutil.copy(src, dst)
        staged.append(fname)
    return staged


def run_pipeline(files, duration_min, use_classifier, ckpt_path,
                 lufs, restricted_mode, progress=gr.Progress()) -> tuple[str, str, str, str]:
    """
    Run end-to-end pipeline.
    Returns (final_mix_wav_path, raw_mix_wav_path, timeline_pretty, status).
    """
    progress(0.0, desc='setting up...')
    reset_dirs()
    staged = stage_uploads(files)
    if len(staged) < 2:
        return None, None, '', f'ERROR: need >=2 clips, got {len(staged)}'

    progress(0.1, desc=f'analyzing {len(staged)} clip(s)...')
    from analyze import analyze_pool
    try:
        analyze_pool(CLIP_DIR, CACHE_DIR, device='cuda')
    except Exception as e:
        return None, None, '', f'analyze failed: {e}'

    progress(0.55, desc='planning timeline...')
    from planner import load_clips, plan, save_timeline, PlannerConfig
    clips = load_clips(CACHE_DIR)
    if not clips:
        return None, None, '', 'no analyzed clips after analyze (check clip format)'

    cfg_kwargs = dict(
        target_duration=float(duration_min) * 60.0,
        surprise_budget=10,
        callback_budget=1,
        max_clips=200,
        restricted=restricted_mode,
    )
    if use_classifier and ckpt_path and os.path.exists(ckpt_path):
        cfg_kwargs['classifier_ckpt'] = ckpt_path
    cfg = PlannerConfig(**cfg_kwargs)
    timeline = plan(clips, cfg)
    timeline_path = os.path.join(OUT_DIR, 'timeline.json')
    save_timeline(timeline, timeline_path)

    progress(0.7, desc='executing transitions...')
    from execute import execute
    raw_path = os.path.join(OUT_DIR, 'raw_mix.wav')
    try:
        execute(timeline_path, CACHE_DIR, raw_path, SAMPLES_DIR)
    except Exception as e:
        return None, None, json.dumps(timeline, indent=2), f'execute failed: {e}'

    progress(0.9, desc='mastering...')
    from master import master
    final_path = os.path.join(OUT_DIR, 'final_mix.wav')
    try:
        master(raw_path, final_path, target_lufs=float(lufs))
    except Exception as e:
        return raw_path, raw_path, json.dumps(timeline, indent=2), f'master failed: {e}'

    progress(1.0, desc='done')
    timeline_pretty = _pretty_timeline(timeline)
    status = (f"✓ {len(staged)} clips, {len(timeline)} segments, "
              f"{sum(e['segment']['end']-e['segment']['start'] for e in timeline):.0f}s plan, "
              f"classifier={'ON' if cfg_kwargs.get('classifier_ckpt') else 'OFF (rule tree)'}")
    return final_path, raw_path, timeline_pretty, status


def _pretty_timeline(tl: list[dict]) -> str:
    lines = [f"Timeline ({len(tl)} entries):", ""]
    for i, e in enumerate(tl):
        seg = e['segment']
        tech = e['transition_in']
        seg_dur = seg['end'] - seg['start']
        lines.append(
            f"  {i:2d}. {e['clip_id'][:38]:38s} "
            f"[{seg.get('type','?'):10s}] "
            f"{seg_dur:5.1f}s  "
            f"key={e['target_key']:>3s}  "
            f"bpm={e['target_bpm']:.1f}  "
            f"-> {tech.get('name','?')} (bars={tech.get('bars','-')})"
        )
    return '\n'.join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title='AiJockey — AI DJ Set Generator',
                   theme=gr.themes.Base()) as ui:
        gr.Markdown(
            "# AiJockey\n"
            "Open-source AI DJ set generator. Upload clips, get a mixed set with "
            "pro-DJ transitions. Hybrid agent + ML pipeline. AGPL-3.0.\n\n"
            "Repo: https://github.com/architagrawal/aiJockey"
        )

        with gr.Row():
            with gr.Column(scale=1):
                files = gr.Files(
                    label='Upload audio clips (3+ recommended, .wav/.mp3)',
                    file_types=['audio'],
                    file_count='multiple',
                )
                duration_min = gr.Slider(
                    label='Mix duration (minutes)',
                    minimum=2, maximum=30, value=5, step=1,
                )
                lufs = gr.Slider(
                    label='Mastering LUFS (club ≈ -9, streaming ≈ -14)',
                    minimum=-16, maximum=-6, value=-9, step=1,
                )
                use_classifier = gr.Checkbox(
                    label='Use trained ML classifier for technique selection',
                    value=False,
                )
                ckpt_path = gr.Textbox(
                    label='Classifier checkpoint path',
                    value='checkpoints/technique_classifier.pt',
                    visible=False,
                )
                use_classifier.change(
                    lambda v: gr.update(visible=v), use_classifier, ckpt_path,
                )
                restricted_mode = gr.Checkbox(
                    label='Restricted (demo-safe) mode — drops artifact-prone techniques',
                    value=True,
                )
                btn = gr.Button('Generate Mix', variant='primary')
            with gr.Column(scale=1):
                status = gr.Textbox(label='Status', interactive=False, lines=2)
                final_audio = gr.Audio(label='Mastered Mix (final)', type='filepath')
                raw_audio = gr.Audio(label='Raw Mix (pre-master)', type='filepath')
                timeline_box = gr.Code(label='Timeline', language='markdown',
                                       lines=20)

        gr.Markdown(
            "### How it works\n"
            "1. **Analyze** — Demucs stems + madmom beats + librosa key + CLAP "
            "embedding per clip\n"
            "2. **Plan** — beam-search builds non-sequential timeline with "
            "15-technique transition library\n"
            "3. **Execute** — rubberband stretch/pitch + per-technique DSP\n"
            "4. **Master** — multiband comp + LUFS norm + limiter\n\n"
            "**Toggle the ML checkbox** to A/B against rule-based decision tree."
        )

        btn.click(
            fn=run_pipeline,
            inputs=[files, duration_min, use_classifier, ckpt_path,
                    lufs, restricted_mode],
            outputs=[final_audio, raw_audio, timeline_box, status],
        )

    return ui


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--share', action='store_true',
                    help='create public Gradio tunnel URL')
    args = ap.parse_args()

    print('starting AiJockey demo...')
    ui = build_ui()
    ui.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
    )
