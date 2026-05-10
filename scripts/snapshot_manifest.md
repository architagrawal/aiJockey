# Snapshot manifest — what to back up before destroying droplet

Goal: spin up a new MI300X droplet from scratch and reach "ready to run pipeline"
in under 30 minutes by restoring exactly these artifacts. Anything not on this
list is reproducible from code + network.

Snapshots cost ~$0.06/GB/month. Keep the manifest tight; don't snapshot what's
re-derivable.

## MUST snapshot

| Path | Size | Why irreplaceable |
|------|-----:|------------------|
| `clips/` | ~5 GB | Curated user-clip pool. Re-downloading takes hours; YouTube IDs disappear. |
| `cache/` | ~2-6 GB | Analyzed CLAP/stems/beats per clip. ~1 min/clip × N to regenerate on GPU. |
| `samples/` | ~60 MB | Meme/SFX library. Curated; re-downloading paths break. |
| `checkpoints/` | ~200 MB | Trained heads (CLAP-compat, technique classifier, mix critic). Hours of training. |
| `datasets/dj_sets_mp3/` | ~2.5 GB | 25 long-form DJ sets used as critic positives. Re-scraping is slow + brittle. |
| `/scratch/embed/clap.npy` + `clap_index.json` + `captions.json` | ~10-50 MB | CLAP retrieval index. Rebuild requires whole pool re-analyze. |
| `/scratch/preferences/` | ~1-10 MB | DPO preference pairs from S5. Hours of self-play to regenerate. |
| `/scratch/models/director_dpo_e*` | ~150 MB per epoch | LoRA adapters from S7. Hours of training. |
| `~/.cache/huggingface/` | ~30 GB | Qwen2-Audio + Qwen2.5-Instruct + CLAP + Demucs + Beat-This!. Re-downloading is bandwidth + time tax. |
| `~/.cache/torch/hub/checkpoints/` | varies | demucs htdemucs_ft (~640 MB), Beat-This! ckpt. |

## DO NOT snapshot

| Path | Why skip |
|------|----------|
| `/scratch/raw/` | Re-downloadable via `scripts/stage0_download.py`. |
| `/scratch/transitions/` | Re-derived from `/scratch/cache` + S2. |
| `/scratch/output/` | Renders themselves — keep only the few you want as demos. |
| `/scratch/renders/` | Self-play intermediate output. |
| `output/` (repo) | Local renders, not load-bearing on droplet. |
| `node_modules/` (if any) | Reinstallable. |
| `__pycache__/` | Regenerate on first import. |
| `.git/` (entire repo state) | Pull from remote. |

## Optional (saves time but reproducible)

| Path | Size | Saves |
|------|-----:|-------|
| `/scratch/cache/stems/` | ~50 GB on full pool | ~30 min/100 clips Demucs re-run |
| `/scratch/transitions/*.json` | ~100 MB on full pool | ~5 min/100 clips S2 re-run |

If snapshot budget is tight, skip the `stems/` subdir — heaviest single bucket.

## Pre-snapshot cleanup

Run before snapshotting to drop transient artifacts:

```bash
bash scripts/clean_for_snapshot.sh
```

That script clears `/scratch/raw`, `/scratch/output`, `/scratch/renders`, and
local `__pycache__`. Edit if it doesn't already.

## Restore workflow

After spinning up new droplet from snapshot:

```bash
ssh aijockey-mi300x
cd /workspace/aijockey
git pull origin best-output-pipeline       # fresh code
source .venv/bin/activate
pip install -r requirements-rocm.txt        # in case deps shifted
bash scripts/preflight.sh                   # verify everything
# If preflight green:
bash scripts/pipeline_launch.sh             # resume pipeline
```

## Verify restore

`scripts/preflight.sh` checks:
- ROCm + torch.cuda
- All Python deps importable
- Demucs + CLAP + Director models cached
- /scratch subdirs ready
- Unit tests pass

If preflight green, you're back where you left off in <30 min total.

## Snapshot lifecycle policy

| Event | Action |
|-------|--------|
| End of dev session (droplet idle) | Snapshot if uncommitted training artifacts on disk; otherwise destroy droplet |
| After major checkpoint training (CLAP head v2, critic v2, DPO adapter) | Snapshot |
| After demo prep done, before destroy | Snapshot as `aijockey-demo-ready` |
| Pre-deploy to public | Snapshot as `aijockey-prod-vN` |

Keep last 2 snapshots, delete older. ~$15-20/mo for 2 × 110 GB.
