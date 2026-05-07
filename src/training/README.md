# Training

Two tracks running in parallel:

- **Tier 1**: trainable transition classifier (T4-feasible, scaffold here)
- **Tier 3**: fine-tune MusicGen / Stable Audio Open for generative bridges
  (MI300X-required, scaffold here, dataset infra ready)

Tier 2 (RLHF mix-quality scorer) deferred until Tier 1 + 3 produce listenable output.

## Quick map

| File | Tier | Purpose |
|------|------|---------|
| `features.py` | 1 | feature extraction |
| `synthetic_dataset.py` | 1 | synthetic dataset (smoothness-labeled) |
| `classifier.py` | 1 | MLP train/eval |
| `integrate.py` | 1 | plug classifier into planner |
| `scrape/youtube_dl.py` | 3 | yt-dlp wrapper |
| `scrape/tracklists.py` | 3 | manual tracklist JSON loader |
| `scrape/README.md` | 3 | legal + workflow notes |
| `dataset_builder.py` | 3 | extract real transitions from DJ mixes |
| `transitions_data.py` | 3 | PyTorch Dataset over real transitions |
| `finetune_musicgen.py` | 3 | fine-tune skeleton (MI300X) |

---

## Tier 1 — Technique Classifier

Replaces hand-coded transition decision tree in `planner.transition_score`
with a small MLP trained on synthetic data.

## Pipeline

```
analyzed clips (cache/*.json + .npz)
        |
        v
[synthetic_dataset.py]   render every technique on every clip pair,
                          score smoothness, label by best technique
        |
        v
datasets/synthetic_transitions.npz    (X, y, scores)
        |
        v
[classifier.py]          train MLP (1051 -> 512 -> 128 -> 15)
        |
        v
checkpoints/technique_classifier.pt
        |
        v
[integrate.py + planner --classifier flag]   replaces decision tree
```

## Files

- `features.py` — feature extraction `(prev_clip, prev_seg, cand_clip, cand_seg) -> (1051,)`
- `synthetic_dataset.py` — render all techniques, FAD-like smoothness scoring, labels by best
- `classifier.py` — MLP train + eval scripts
- `integrate.py` — load classifier, expose `pick_technique()` for planner

## Usage

### Build dataset

```bash
python src/training/synthetic_dataset.py \
    --cache cache/ \
    --samples samples/ \
    --out datasets/synthetic_transitions.npz
```

5 clips → ~20 ordered pairs × 15 techniques = ~300 renders, ~5-10 min on T4.
20 clips → ~380 pairs × 15 = ~5700 renders. Scale up after MVP.

### Train

```bash
python src/training/classifier.py \
    --dataset datasets/synthetic_transitions.npz \
    --ckpt checkpoints/technique_classifier.pt \
    --epochs 100
```

### Use in pipeline

```bash
python src/main.py plan \
    --cache cache/ \
    --duration 300 \
    --classifier checkpoints/technique_classifier.pt
```

Or A/B without flag = rule-based fallback.

## Honest caveats

- **Synthetic labels are weak signal**. Smoothness ≠ "good DJ choice" (sometimes you WANT a sharp cut). Use this as bootstrap, then upgrade with real DJ-set data in Tier 2.
- **Dataset size** depends on clip pool. 5 clips → ~20 examples = severe overfit risk. Need 20+ clips minimum for usable model.
- **Class imbalance** likely — some techniques rarely score best. classifier.py uses inverse-frequency loss weights to compensate.
- **Feature dim 1051** is mostly raw CLAP — could PCA-reduce to ~128 if dataset stays small (better generalization).

## Tier 1 Roadmap

- **Tier 1.1**: improve synthetic scoring (add CLAP coherence, key continuity, FAD vs published-DJ-set reference)
- **Tier 1.2**: weak labels from real DJ mixes (Tier 3 dataset doubles as Tier 1.2 source — `technique_guess` field auto-labels)

---

## Tier 3 — Real Dataset + MusicGen Fine-tune

### Step 1 — Build dataset

```bash
# Get a JSON skeleton:
python src/training/scrape/tracklists.py skeleton

# Edit datasets/tracklists/example.json:
#   set "url", "title", "duration_sec", and the "transitions" array
#   each transition: at_sec + from_track + to_track

# Build dataset (downloads via yt-dlp + extracts windows):
python src/training/dataset_builder.py \
    --tracklists datasets/tracklists \
    --raw datasets/raw_mixes \
    --out datasets/transitions_real
```

Output structure:
```
datasets/transitions_real/
  <mix_id>/
    000_pre.wav         16-second pre-window
    000_transition.wav  8-second transition window
    000_post.wav        16-second post-window
    000_features.npz    CLAP_pre, CLAP_post, tempo_pre, tempo_post
    001_*.wav, 001_features.npz
    ...
    index.json
  master_index.json     all transitions across all mixes
```

### Step 2 — Verify dataset

```python
from src.training.transitions_data import TransitionDataset
ds = TransitionDataset()
print(len(ds))       # number of transitions
print(ds[0].keys())  # pre_audio, transition_audio, post_audio, clap_pre/post, tempo_*, technique_label
```

### Step 3 — Fine-tune (MI300X)

```bash
# On MI300X instance:
python src/training/finetune_musicgen.py \
    --dataset_dir datasets/transitions_real \
    --base_model medium \
    --out_dir checkpoints/musicgen_dj \
    --epochs 5 \
    --use_qlora
```

NOTE: skeleton currently — needs audiocraft ConditionProvider integration to
plug `DJContextEncoder` into MusicGen's cross-attention. TODOs marked in
`finetune_musicgen.py`. Estimate $200-400 of MI300X credits for full fine-tune.

### License notes

- MusicGen weights: CC-BY-NC. Output is non-commercial only.
- For commercial path: train Stable Audio Open (Stability Community License) instead.
- Dataset legality: see `scrape/README.md`. Personal/research only — keep raw
  audio out of git (`.gitignore` enforces).

---

## Roadmap (consolidated)

- **Tier 1.1** — refine synthetic labels
- **Tier 1.2** — auto-label real transitions, retrain classifier on real data
- **Tier 2** — RLHF: rate generated mixes pairwise, train preference model
- **Tier 3** — MusicGen fine-tune for bridge generation
- **Tier 4** — custom audio token LM, train from scratch on aggregated data
  (months on MI300X)
