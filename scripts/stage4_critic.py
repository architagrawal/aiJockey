"""S4 — critic v2 streaming train.

Watch /scratch/transitions/ for new triplets. Build training set from
real-DJ positives (DJ-set transitions) vs synthetic negatives (random
splices of unrelated clips). Train CLAP-feature critic with multi-task
auxiliary heads (BPM regression, key classification, technique classification).

Streaming: retrains every N minutes as data grows. Resumes from latest
/scratch/models/critic_v2_e{N}.pt.

Phase A polish §16.1.A (augment), §16.1.E (multi-task), §16.2.A-G (efficiency).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pipeline.common import scratch_dir, atomic_write
from training.efficiency import (autocast_ctx, get_dtype, maybe_compile,
                                  make_optimizer)


CRITIC_DIR = lambda: scratch_dir('models')
TRANS_DIR = lambda: scratch_dir('transitions')
CACHE_DIR = lambda: scratch_dir('cache')

MIN_SAMPLES_TO_START = 200
RETRAIN_INTERVAL_SEC = 1800  # 30 min
NUM_TECH_CLASSES = 8


class CriticV2(nn.Module):
    """CLAP-feature critic with multi-task auxiliary heads.

    Inputs: pre + transition + post CLAP embeddings (3 x 512 = 1536).
    Outputs:
        is_real:    P(real DJ transition)
        bpm:        regression target
        key_class:  24-way (Camelot)
        tech_class: NUM_TECH_CLASSES technique
    """

    def __init__(self, in_dim: int = 1536, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
        )
        self.head_real = nn.Linear(hidden, 1)
        self.head_bpm = nn.Linear(hidden, 1)
        self.head_key = nn.Linear(hidden, 24)
        self.head_tech = nn.Linear(hidden, NUM_TECH_CLASSES)

    def forward(self, x):
        h = self.backbone(x)
        return {
            'real': self.head_real(h).squeeze(-1),
            'bpm': self.head_bpm(h).squeeze(-1),
            'key': self.head_key(h),
            'tech': self.head_tech(h),
        }


def _load_dataset() -> tuple[torch.Tensor, dict[str, torch.Tensor]] | None:
    """Build features tensor + label dict from current triplets on disk."""
    triplets = list(TRANS_DIR().rglob('t*.json'))
    if len(triplets) < MIN_SAMPLES_TO_START:
        return None
    embed = scratch_dir('embed') / 'clap.npy'
    idx_path = scratch_dir('embed') / 'clap_index.json'
    if not embed.exists() or not idx_path.exists():
        return None
    vecs = np.load(embed)
    idx = json.loads(idx_path.read_text())

    feats: list[np.ndarray] = []
    labels = {'real': [], 'bpm': [], 'key': [], 'tech': []}
    for tp in triplets:
        try:
            t = json.loads(tp.read_text())
        except Exception:
            continue
        cid = t['pre']['clip_id']
        if cid not in idx:
            continue
        v = vecs[idx[cid]]
        # 3x stacked (pre = trans = post for now; richer when stem-level CLAP added)
        feats.append(np.concatenate([v, v, v]).astype(np.float32))
        labels['real'].append(1.0)
        labels['bpm'].append(float(t.get('bpm', 120.0)))
        labels['key'].append(int(t.get('key_class', 0)))
        labels['tech'].append(int(t.get('tech_class', 0)))

    if not feats:
        return None
    x = torch.from_numpy(np.stack(feats))
    y = {
        'real': torch.tensor(labels['real'], dtype=torch.float32),
        'bpm': torch.tensor(labels['bpm'], dtype=torch.float32),
        'key': torch.tensor(labels['key'], dtype=torch.long),
        'tech': torch.tensor(labels['tech'], dtype=torch.long),
    }
    return x, y


def _generate_negatives(x_pos: torch.Tensor, n: int) -> torch.Tensor:
    """Random splice of three different clips' CLAP vecs = a 'fake' triplet."""
    rng = torch.randperm(len(x_pos))
    a = x_pos[rng[:n], :512]
    b = x_pos[torch.randperm(len(x_pos))[:n], 512:1024]
    c = x_pos[torch.randperm(len(x_pos))[:n], 1024:]
    return torch.cat([a, b, c], dim=1)


def _latest_checkpoint() -> Path | None:
    cps = sorted(CRITIC_DIR().glob('critic_v2_e*.pt'))
    return cps[-1] if cps else None


def train_one_pass(model: CriticV2, x: torch.Tensor, y: dict[str, torch.Tensor],
                   epoch: int) -> dict[str, float]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    opt = make_optimizer(model.parameters(), lr=3e-4)

    # Add 1:1 negatives
    n_pos = len(x)
    x_neg = _generate_negatives(x, n_pos)
    y_real = torch.cat([y['real'], torch.zeros(n_pos)])
    x_full = torch.cat([x, x_neg]).to(device)
    y_real = y_real.to(device)
    y_bpm = y['bpm'].to(device)
    y_key = y['key'].to(device)
    y_tech = y['tech'].to(device)

    bs = 64
    perm = torch.randperm(len(x_full))
    losses = {'real': 0.0, 'bpm': 0.0, 'key': 0.0, 'tech': 0.0}
    n_batches = 0
    for i in range(0, len(x_full), bs):
        b = perm[i:i + bs]
        xb = x_full[b]
        yr = y_real[b]
        # bpm/key/tech only meaningful for positives (first n_pos)
        opt.zero_grad()
        with autocast_ctx():
            out = model(xb)
            l_real = bce(out['real'], yr)
            pos_mask = b < n_pos
            if pos_mask.any():
                idx_pos = b[pos_mask]
                l_bpm = mse(out['bpm'][pos_mask], y_bpm[idx_pos] / 100.0)
                l_key = ce(out['key'][pos_mask], y_key[idx_pos])
                l_tech = ce(out['tech'][pos_mask], y_tech[idx_pos])
            else:
                l_bpm = l_key = l_tech = torch.tensor(0.0, device=device)
            loss = l_real + 0.3 * (l_bpm + l_key + l_tech)
        loss.backward()
        opt.step()
        losses['real'] += float(l_real)
        losses['bpm'] += float(l_bpm)
        losses['key'] += float(l_key)
        losses['tech'] += float(l_tech)
        n_batches += 1
    if n_batches:
        for k in losses:
            losses[k] /= n_batches
    losses['epoch'] = float(epoch)
    return losses


def watch_loop(min_samples: int, interval_sec: float) -> None:
    print(f"S4 critic-train, min={min_samples} samples, every {interval_sec}s")
    model = CriticV2()
    cp = _latest_checkpoint()
    epoch = 0
    if cp is not None:
        try:
            state = torch.load(cp, map_location='cpu')
            model.load_state_dict(state['model'])
            epoch = int(state.get('epoch', 0))
            print(f"S4 resumed from {cp.name} epoch={epoch}")
        except Exception as e:
            print(f"warn: failed to load {cp}: {e}")
    model = maybe_compile(model)

    while True:
        ds = _load_dataset()
        if ds is None:
            print("S4 not enough data yet, sleeping")
            time.sleep(interval_sec)
            continue
        x, y = ds
        if len(x) < min_samples:
            time.sleep(interval_sec)
            continue
        epoch += 1
        losses = train_one_pass(model, x, y, epoch)
        print(f"S4 e{epoch} n={len(x)} losses={ {k: round(v, 4) for k, v in losses.items()} }")
        out = CRITIC_DIR() / f'critic_v2_e{epoch:03d}.pt'
        torch.save({'model': model.state_dict(), 'epoch': epoch}, out)
        print(f"S4 saved {out}")
        time.sleep(interval_sec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', default=str(TRANS_DIR()))
    ap.add_argument('--min-samples', type=int, default=MIN_SAMPLES_TO_START)
    ap.add_argument('--interval', type=float, default=RETRAIN_INTERVAL_SEC)
    args = ap.parse_args()
    watch_loop(args.min_samples, args.interval)


if __name__ == '__main__':
    main()
