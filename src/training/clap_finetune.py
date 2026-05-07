"""
DJ-compatibility projection head: train MLP on top of frozen CLAP embeddings
via InfoNCE contrastive loss.

Replaces raw CLAP cosine in planner with embedding cosine in DJ-compat space.
After training: similar-mixability tracks have higher cosine similarity, even
if their raw audio timbres differ.

Usage:
    python src/training/clap_finetune.py \
        --triplets datasets/clap_triplets.npz \
        --ckpt checkpoints/clap_compat_head.pt \
        --epochs 50

Inference:
    from training.clap_finetune import load_compat_head, project
    head = load_compat_head('checkpoints/clap_compat_head.pt')
    z = project(head, clap_embedding)   # (128,) DJ-compat vector
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


CLAP_DIM = 512
EMBED_DIM = 128


class DJCompatibilityHead(nn.Module):
    """
    Projection head: CLAP (512) -> DJ-compat embedding (128).
    Output is L2-normalized. Cosine similarity = compatibility score.
    """

    def __init__(self, in_dim: int = CLAP_DIM, hidden: int = 256,
                 out_dim: int = EMBED_DIM, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        return F.normalize(z, dim=-1)


class TripletDataset(Dataset):
    """Loads triplets from npz produced by clap_pairs.py."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.anchor = torch.from_numpy(d['anchor']).float()
        self.positive = torch.from_numpy(d['positive']).float()
        self.negatives = torch.from_numpy(d['negatives']).float()  # (N, K, 512)

    def __len__(self) -> int:
        return self.anchor.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            'anchor': self.anchor[idx],
            'positive': self.positive[idx],
            'negatives': self.negatives[idx],
        }


def info_nce(anchor_z: torch.Tensor, positive_z: torch.Tensor,
             negatives_z: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE: pull anchor toward positive, push from K negatives.
    anchor_z, positive_z: (B, D)
    negatives_z: (B, K, D)
    Returns scalar loss.
    """
    pos_sim = (anchor_z * positive_z).sum(dim=-1, keepdim=True) / temperature  # (B, 1)
    neg_sim = torch.einsum('bd,bkd->bk', anchor_z, negatives_z) / temperature  # (B, K)
    logits = torch.cat([pos_sim, neg_sim], dim=-1)  # (B, 1+K)
    target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


def train(triplet_npz: str, ckpt_path: str,
          epochs: int = 50, batch_size: int = 64, lr: float = 1e-3,
          val_frac: float = 0.15, temperature: float = 0.07,
          device: str = 'auto') -> dict:
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = TripletDataset(triplet_npz)
    print(f"triplets: N={len(ds)}, anchor dim={ds.anchor.shape[-1]}, "
          f"K negatives={ds.negatives.shape[1]}")
    n_val = max(1, int(val_frac * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = DJCompatibilityHead().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            a = batch['anchor'].to(device)
            p = batch['positive'].to(device)
            n_ = batch['negatives'].to(device)
            B, K, _ = n_.shape
            za = model(a)
            zp = model(p)
            zn = model(n_.reshape(B * K, -1)).reshape(B, K, -1)
            loss = info_nce(za, zp, zn, temperature)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss_sum += float(loss)
            n_batches += 1
        sched.step()
        train_loss = train_loss_sum / max(n_batches, 1)

        # Val: accuracy = positive scored higher than ALL negatives?
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                a = batch['anchor'].to(device)
                p = batch['positive'].to(device)
                n_ = batch['negatives'].to(device)
                B, K, _ = n_.shape
                za = model(a)
                zp = model(p)
                zn = model(n_.reshape(B * K, -1)).reshape(B, K, -1)
                loss = info_nce(za, zp, zn, temperature)
                val_loss_sum += float(loss)
                pos_sim = (za * zp).sum(dim=-1, keepdim=True)
                neg_sim = torch.einsum('bd,bkd->bk', za, zn)
                correct += int((pos_sim > neg_sim.max(dim=-1, keepdim=True).values).sum())
                total += B
        val_loss = val_loss_sum / max(len(val_loader), 1)
        val_acc = correct / max(total, 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'in_dim': CLAP_DIM,
                'embed_dim': EMBED_DIM,
                'epoch': epoch,
                'val_acc': val_acc,
                'config': {'temperature': temperature},
            }, ckpt_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    print(f"\nbest val_acc={best_val_acc:.3f} (saved {ckpt_path})")
    history_path = Path(ckpt_path).with_suffix('.history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    return history


def load_compat_head(ckpt_path: str, device: str = 'auto') -> DJCompatibilityHead:
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    head = DJCompatibilityHead(state['in_dim'], 256, state['embed_dim']).to(device)
    head.load_state_dict(state['model_state_dict'])
    head.eval()
    print(f"loaded compat head: epoch={state.get('epoch','?')}, "
          f"val_acc={state.get('val_acc','?'):.3f}")
    return head


@torch.no_grad()
def project(head: DJCompatibilityHead, clap_emb: np.ndarray) -> np.ndarray:
    """Project a single CLAP embedding (512,) to DJ-compat space (128,)."""
    device = next(head.parameters()).device
    x = torch.from_numpy(clap_emb).float().unsqueeze(0).to(device)
    z = head(x)[0].cpu().numpy()
    return z


@torch.no_grad()
def project_batch(head: DJCompatibilityHead, claps: np.ndarray) -> np.ndarray:
    """Project a batch of CLAP embeddings (N, 512) to (N, 128)."""
    device = next(head.parameters()).device
    x = torch.from_numpy(claps).float().to(device)
    return head(x).cpu().numpy()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--triplets', default='datasets/clap_triplets.npz')
    ap.add_argument('--ckpt', default='checkpoints/clap_compat_head.pt')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--temperature', type=float, default=0.07)
    args = ap.parse_args()
    train(args.triplets, args.ckpt, args.epochs, args.batch_size, args.lr,
          temperature=args.temperature)
