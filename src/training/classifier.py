"""
Train a small MLP technique classifier on synthetic transition dataset.

Replaces the rule-based decision tree in planner.transition_score().
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
from torch.utils.data import DataLoader, TensorDataset, random_split

from features import FEATURE_DIM, N_TECHNIQUES, TECHNIQUES


class TechniqueClassifier(nn.Module):
    """MLP: features -> technique probability."""

    def __init__(self, in_dim: int = FEATURE_DIM, n_classes: int = N_TECHNIQUES,
                 hidden: tuple[int, ...] = (512, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(dataset_path: str, ckpt_path: str, epochs: int = 50,
          batch_size: int = 32, lr: float = 1e-3,
          val_frac: float = 0.2, device: str = 'auto') -> dict:
    npz = np.load(dataset_path, allow_pickle=True)
    X = torch.from_numpy(npz['X']).float()
    y = torch.from_numpy(npz['y']).long()
    print(f"dataset: X={tuple(X.shape)}, y={tuple(y.shape)}")
    if X.shape[0] < 10:
        print("WARN: very small dataset, results unreliable")

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    ds = TensorDataset(X, y)
    n_val = max(1, int(val_frac * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Class weights to combat imbalance (synthetic datasets often skew)
    class_counts = np.bincount(npz['y'], minlength=N_TECHNIQUES).astype(np.float32)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.sum() * N_TECHNIQUES
    print("class counts:", dict(zip(TECHNIQUES, class_counts.astype(int))))

    model = TechniqueClassifier().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss_sum += float(loss) * xb.size(0)
            n += xb.size(0)
        sched.step()
        train_loss = train_loss_sum / max(n, 1)

        model.eval()
        val_loss_sum = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss_sum += float(loss_fn(logits, yb)) * xb.size(0)
                preds = logits.argmax(dim=-1)
                correct += int((preds == yb).sum())
                total += xb.size(0)
        val_loss = val_loss_sum / max(total, 1)
        val_acc = correct / max(total, 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_dim': FEATURE_DIM,
                'n_classes': N_TECHNIQUES,
                'techniques': TECHNIQUES,
                'epoch': epoch,
                'val_acc': val_acc,
            }, ckpt_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    print(f"\nbest val_acc={best_val_acc:.3f} (saved {ckpt_path})")
    history_path = Path(ckpt_path).with_suffix('.history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    return history


def predict_technique(ckpt_path: str, features: np.ndarray,
                      device: str = 'auto') -> tuple[str, np.ndarray]:
    """Load checkpoint, predict technique. Returns (name, prob_distribution)."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TechniqueClassifier(state['feature_dim'], state['n_classes']).to(device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    x = torch.from_numpy(features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return state['techniques'][idx], probs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='datasets/synthetic_transitions.npz')
    ap.add_argument('--ckpt', default='checkpoints/technique_classifier.pt')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    train(args.dataset, args.ckpt, args.epochs, args.batch_size, args.lr)
