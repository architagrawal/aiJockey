# Hackathon Demo — Host on AMD MI300X

1-day demo plan. No new model training. Polish + UI + narrative.

## Hosting (AMD MI300X)

### Setup (~30 min, one-time)

```bash
# SSH to MI300X instance (AMD Developer Cloud)
ssh <amd-instance>

# Clone + install
git clone https://github.com/architagrawal/aiJockey.git
cd aiJockey

# Python env
python -m venv .venv && source .venv/bin/activate

# ROCm-enabled torch
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/rocm6.0

# numpy<2 + cython<3 first
pip install --force-reinstall --no-deps "numpy<2.0"
pip install --no-build-isolation --upgrade "cython<3"

# Audio libs
pip install demucs==4.0.1 librosa==0.10.2 soundfile==0.12.1 \
    pyrubberband pyloudnorm==0.1.1 scipy tqdm transformers gradio

# madmom from git
pip install --no-build-isolation git+https://github.com/CPJKU/madmom.git

# System binaries
sudo apt-get install -y rubberband-cli ffmpeg

# Verify
python scripts/00_imports_only.py
python scripts/00_rocm_sanity.py path/to/test_clip.wav  # need a test clip
```

### Pre-cache models (avoid first-run download during demo)

```bash
# Trigger Demucs + CLAP downloads now, not during demo
python -c "from demucs.pretrained import get_model; get_model('htdemucs')"
python -c "from transformers import ClapModel; ClapModel.from_pretrained('laion/clap-htsat-unfused')"
```

### Pre-analyze demo clips (avoid analyze latency)

Drop 5-10 known-good clips into `clips/`, then:
```bash
python src/main.py analyze --clips clips/ --device cuda
```
Cache stays in `cache/` for instant demo.

### Launch demo

```bash
# In tmux so it survives SSH drops
tmux new -s demo
python app.py --share
```

Output ends with:
```
Running on local URL: http://0.0.0.0:7860
Running on public URL: https://abc123.gradio.live
```

Open public URL on demo machine. Audience can hit it from phones too.

Detach tmux: Ctrl-B D. Reattach: `tmux a -t demo`.

### When demo done

```bash
# Stop the app
tmux a -t demo
# Ctrl-C in app, then:
exit
exit  # log out of MI300X

# CRITICAL: tear down instance via AMD console (billed hourly)
```

## Demo flow (3-minute pitch + 2-minute live)

### Slide 1 — Problem
- Existing auto-DJ tools: djay Pro auto-mix, Spotify auto-DJ, Mixmaster Robot
- All closed, all generic, none open-source
- Suno/Udio: pure generative, can't use YOUR clips
- Gap: no open-source AI DJ that mixes user-provided pool with pro techniques

### Slide 2 — What we built
- **AiJockey**: hybrid agent + ML system
- 15 transition techniques (cut, crossfade, eq_swap, drum_break, mashup,
  echo_out, spinback, loop_tighten, ...)
- Stem-aware (Demucs)
- Beat-aligned (madmom phrase enforcement)
- Camelot key compatibility
- Sample bank embellishment (impacts, risers, vinyl FX — procedurally synthesized)
- Mastered to club LUFS (-9)
- AGPL-3.0 (forks must remain open)

### Slide 3 — Architecture
[Insert ARCHITECTURE.md diagram]

```
Director → DJ LLM → Selector → Mixer → audio out
                ↑                         ↓
                └──── Critic ←────────────┘
```

Phase A (today, demoed): rule-based pipeline, real-time
Phase B (training): trained MusicGen bridge generator
Phase C (target): full DJ LLM with streaming runtime

### Slide 4 — What's "AI" in current system
- CLAP embeddings — audio→512-dim semantic vectors (LAION pretrained)
- Demucs neural stem separation
- librosa beat tracking + heuristic downbeats (madmom dropped — Python 3.10 incompat)
- Trained MLP technique classifier (Tier 1, optional toggle)
- Trained CLAP compatibility head (Tier 1.5)
- Trained mix critic (Tier 2)
- Multimodal Qwen2-Audio Director — hears clips before producing JSON plan
- Phase A polish: phrase quantization + stem-additive overlap + humanized accents + constitutional validator
- Roadmap: ORPO LoRA Director + MusicGen-Small bridge fine-tune (Phase B start)

### Slide 5 — Live demo
1. Upload 5 EDM clips
2. Click "Generate Mix"
3. ~2 min processing
4. Inline audio playback
5. Show timeline JSON
6. Toggle ML classifier checkbox → re-render → A/B

### Slide 6 — Roadmap + ask
- Tier 3 fine-tune MusicGen on transition data → unlocks generative bridges
- Tier 4 custom DJ LLM → "Suno for DJ sets"
- Need: MI300X credits ($300-500), labeled DJ mix data (~25 mixes manually annotated)

### Slide 7 — Market + open-source angle
- Mubert ($16/mo) closed
- Boomy/AIVA closed
- AiJockey: AGPL-3.0, anyone can fork, but must open-source their fork
- Network use clause = SaaS forks must open too

## Demo safety checklist

- [ ] Tested with 5 known-good clips end-to-end
- [ ] Pre-cached Demucs + CLAP weights on instance
- [ ] Pre-analyzed clips so first demo run uses cache
- [ ] Tested classifier toggle works (or hide if not trained yet)
- [ ] LUFS -9 mastering produces audible club-loud output
- [ ] Public URL works on phones
- [ ] Backup: pre-rendered `final_mix.wav` if live render fails
- [ ] Timer: total demo under 5 min

## Backup plan if MI300X fails

Have a Colab notebook (`scripts/aijockey_colab.ipynb`) ready as fallback.
Same pipeline, T4 GPU, slightly slower but works.

Or pre-render mix locally → play file from laptop.

## Cost estimate

MI300X for hackathon day: ~$30-60 of credits (instance + ~6 hours active +
overhead). Tear down right after.

## Key talking points

1. **"Your clips, your mix"** — unlike Suno, model uses YOUR audio, no
   hallucinations or copyright concerns.
2. **15 pro DJ techniques** — most auto-DJ tools do crossfade. We do drops,
   mashups, echo-outs, spinbacks.
3. **Open-source AGPL** — forks stay open, contributions accumulate.
4. **Hybrid agent + ML** — ML where it adds value (technique selection),
   rules where DSP is solved (time-stretch).
5. **Streaming-ready architecture** — can serve real-time mixes via WebSocket
   in Phase C.

## Repo + URL handout

- GitHub: https://github.com/architagrawal/aiJockey
- License: AGPL-3.0
- Live demo URL: [paste your gradio.live URL]
