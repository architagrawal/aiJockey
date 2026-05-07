# Training — Tier 1 (Technique Classifier)

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

## Roadmap

- **Tier 1.1**: improve synthetic scoring (add CLAP coherence, key continuity, FAD vs published-DJ-set reference)
- **Tier 1.2**: weak labels from YouTube DJ mixes (1001tracklists alignment + auto-detect technique per transition)
- **Tier 2**: pairwise preference learning — rate mix-A vs mix-B
- **Tier 3**: generative transition fill via Stable Audio Open QLoRA fine-tune
