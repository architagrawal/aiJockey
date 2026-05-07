# AiJockey — Getting Started

AI DJ set generator. Builds cohesive mixes from clip pool with pro-DJ transitions.

Full architecture + design + code: see [AiJockey.md](AiJockey.md).

This README = practical onboarding: how to develop, what data, how to use AMD GPU.

**License**: [AGPL-3.0-or-later](LICENSE). Forks (including network-served versions like SaaS) must remain open-source under the same license.

---

## Dev workflow (ordered)

```
Week 0     setup local repo + venv on laptop, 5 test clips, sanity scripts
Week 1     first MI300X session ~$5: ROCm sanity + 1-clip analyze, tear down
Week 2-4   build analyze.py locally (Demucs slow on CPU but works for 5 clips)
Week 5     MI300X batch analyze IF 50+ clips, else stay laptop
Week 6-9   planner + transitions + execute (CPU, fast iteration)
Week 10-12 first end-to-end mix renders, iterate weights
Week 13-16 UI (FastAPI + frontend)
Week 17+   Phase B (generative fills) if needed
```

**Golden rule**: dev locally, run heavy jobs remote. Never dev on MI300X (idle = wasted credits). Git push/pull between.

---

## Data needed

### Test clips (5-10 for dev)

Pick tracks YOU know well so you can judge output quality.

- **Best**: 5 of your own DJ sets / favorite remixes (own audio, no licensing issues)
- **Free CC0/permissive sources**:
  - [NCS](https://ncs.io) — NoCopyrightSounds, EDM-heavy
  - [Free Music Archive](https://freemusicarchive.org) — varied genres
  - [Bandcamp](https://bandcamp.com) — many artists offer free downloads
- **Variety**: 2 EDM, 2 hip-hop/trap, 1 ambient/breakdown — tests transition diversity

### Sample bank (~50 one-shots)

For spinback FX, impacts, risers, sweeps.

- [Cymatics free vault](https://cymatics.fm/pages/free-download-vault) — CC0
- [freesound.org](https://freesound.org) (filter by CC0 license)
- [99sounds](https://99sounds.org) — free packs

Categories: kicks, sub_drops, snare_rolls, white_noise_risers, downsweeps, vinyl_stops, impacts.

### Eval clips (20-30, later)

For testing planner across genre/BPM/key combos. Same sources as above.

### Phase B fine-tune data (~500 transition snippets, much later)

Clip pairs + ground-truth pro-DJ transition. Sources: rip from published DJ sets (research/personal use only) or commission DJs to record transitions.

---

## AMD GPU workflow (MI300X via AMD Developer Cloud)

### One-time setup

1. Sign in: [AMD Developer Cloud](https://dev.cloud.amd.com)
2. Provision MI300X instance — Linux + ROCm 6.x preinstalled
3. Save SSH endpoint + key
4. On instance, configure git:
   ```bash
   git config --global user.email "you@example.com"
   git config --global user.name "Your Name"
   ```

### Per-session pattern

```bash
# LOCAL: push latest code
git push

# SSH to MI300X
ssh <amd-instance>

# First time only:
git clone <your-repo>
cd aijockey
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-rocm.txt

# Subsequent sessions:
cd aijockey && git pull
source .venv/bin/activate

# Run job inside tmux (survives SSH drop)
tmux new -s job
python src/main.py analyze --clips clips/ --cache cache/ --device cuda
# Detach: Ctrl-B then D
# Reattach: tmux a -t job

# When done, pull results to laptop (LOCAL):
rsync -av <amd-instance>:aijockey/cache/ cache/
rsync -av <amd-instance>:aijockey/output/ output/

# CRITICAL: tear down instance via AMD console — billed hourly
```

### Don't burn credits

- **Always** tear down after job done
- **Never** leave instance idle "for tomorrow"
- **Never** dev/debug on remote — push code, run, pull results
- **Use tmux** so SSH drops don't kill long jobs
- **Batch jobs**: queue multiple analyses in one session, not one-at-a-time

### Storage strategy

- Small clips (<10GB): rsync each session
- Large (>10GB): mount S3-compatible bucket once, persist across sessions

### Budget table

| Activity | Where | Est. cost |
|----------|-------|-----------|
| ROCm sanity | MI300X 1hr | ~$5 |
| Phase A dev | laptop CPU | $0 |
| Phase A end-to-end test (10 clips) | laptop GPU or 1 MI300X hr | ~$5 |
| Mass analysis (1000+ clips) | MI300X batch | $20-50 |
| Phase B SAO inference experiments | MI300X 10hr | ~$50 |
| Phase B last-layer fine-tune | MI300X 3-5 days | $200-400 |

**Total Phase A**: $30-60. **Phase B**: $250-450.

---

## Concrete next 3 actions

### 1. Create project skeleton + git (30 min)

```bash
mkdir aijockey && cd aijockey
git init
mkdir -p src tests/fixtures clips samples cache output scripts
python -m venv .venv
# Linux/Mac: source .venv/bin/activate
# Windows:   .venv\Scripts\activate
```

### 2. Add requirements files (10 min)

`requirements-base.txt`:
```
demucs==4.0.1
madmom==0.16.1
librosa==0.10.2
soundfile==0.12.1
pyrubberband==0.3.0
pyloudnorm==0.1.1
scipy
numpy
tqdm
laion-clap==1.1.4
```

`requirements-cpu.txt`:
```
-r requirements-base.txt
torch==2.3.1
torchaudio==2.3.1
```

`requirements-rocm.txt`:
```
-r requirements-base.txt
--index-url https://download.pytorch.org/whl/rocm6.0
torch==2.3.1
torchaudio==2.3.1
```

`requirements-cuda.txt`:
```
-r requirements-base.txt
--index-url https://download.pytorch.org/whl/cu121
torch==2.3.1
torchaudio==2.3.1
```

System deps (Linux): `apt install rubberband-cli ffmpeg`

### 3. Drop 5 test clips + run laptop sanity (1 hr)

Place 5 .wav/.mp3 files in `clips/`.

`scripts/00_imports_only.py`:
```python
"""Verify all libs install on laptop. No GPU needed."""
import torch, torchaudio, demucs, madmom, librosa, soundfile, pyrubberband
print("torch", torch.__version__)
print("all imports OK")
```

Run: `python scripts/00_imports_only.py`

If pass → ready to start `analyze.py` (see AiJockey.md Layer 1).

If fail → fix env first. Most common issues:
- madmom install: needs `numpy<2.0` and Cython
- pyrubberband: needs `rubberband-cli` system binary
- Windows + madmom: troublesome, prefer WSL2 or Linux

---

## Repo layout

```
aijockey/
├── README.md                 # this file
├── AiJockey.md               # full architecture + code reference
├── requirements-*.txt        # pinned deps per platform
├── pyproject.toml            # package metadata (later)
├── clips/                    # input audio (gitignore'd)
├── samples/                  # one-shot bank
│   └── manifest.json
├── cache/                    # analysis features (gitignore'd)
├── output/                   # generated mixes (gitignore'd)
├── scripts/                  # one-off scripts (sanity, eval)
│   ├── 00_imports_only.py
│   └── 00_rocm_sanity.py
├── src/                      # main package
│   ├── analyze.py
│   ├── camelot.py
│   ├── hooks.py
│   ├── phrase.py
│   ├── planner.py
│   ├── transitions.py
│   ├── execute.py
│   ├── master.py
│   ├── eval.py
│   └── main.py               # CLI entry
└── tests/
    └── fixtures/             # 3-5 short test clips for unit tests
```

Suggested `.gitignore`:
```
.venv/
clips/
cache/
output/
samples/
__pycache__/
*.pyc
.pytest_cache/
```

---

## Honest expectations

- Phase A MVP: **3-4 months** part-time. Build incrementally, validate each layer.
- Output quality target: **competent pro auto-DJ**, beats djay Pro auto-mix on transitions. NOT Daft Punk.
- Most quality gains come from: **stem separation + phrase enforcement + transition library + mastering**. Not from any single magic component.
- 15 transition techniques = lots of edge cases. Expect 60% of your time on tuning + fixing artifacts.

See [AiJockey.md](AiJockey.md) sections "Risks" + "Abandon if" + "Push to Phase B if".

---

## Help

- Architecture questions → AiJockey.md
- Stuck on ROCm install → AMD docs at rocm.docs.amd.com
- audiocraft / Stable Audio Open ROCm issues → check GitHub issues, fall back to CPU for inference if needed
- Sample bank curation → start with 10 samples, expand later

Start with the **Concrete next 3 actions** above. Ship Week 0 setup before reading more.
