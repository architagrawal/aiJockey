"""Path C: real-DJ-mix vs random-splice discriminator (mix critic).

Positives: 8-second mel-spectrogram windows from real DJ sets (includes natural
transitions, professional mixing).
Negatives: same-length splices fabricated by concatenating two random clips
from our pool with abrupt cuts (no DJ technique applied).

Train small CNN on log-mel spectrograms. Output: P(real DJ mix) in [0, 1].

Usage:
    python src/training/mix_critic.py build-data --sets datasets/dj_sets_mp3 \\
                                                 --clips clips --out datasets/critic.npz
    python src/training/mix_critic.py train --data datasets/critic.npz \\
                                            --ckpt checkpoints/mix_critic.pt
    python src/training/mix_critic.py score --ckpt checkpoints/mix_critic.pt \\
                                            --audio output/demos/festival_inferno/final_mix.wav
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SR = 22050
N_MELS = 64
WIN_SECONDS = 8.0
WIN_SAMPLES = int(SR * WIN_SECONDS)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def to_mel(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    import librosa
    if wav.ndim > 1:
        wav = wav.mean(axis=0) if wav.shape[0] in (1, 2) else wav.mean(axis=-1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    mel = librosa.feature.melspectrogram(y=wav.astype(np.float32),
                                         sr=SR, n_fft=1024, hop_length=256,
                                         n_mels=N_MELS, power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # normalize roughly to [-1, 1]
    log_mel = (log_mel + 40.0) / 40.0
    return log_mel.astype(np.float32)


def random_window(audio: np.ndarray, sr: int) -> np.ndarray | None:
    target = int(WIN_SECONDS * sr)
    if audio.ndim > 1:
        audio = audio.mean(axis=0) if audio.shape[0] in (1, 2) else audio.mean(axis=-1)
    if audio.shape[0] < target:
        return None
    s = random.randint(0, audio.shape[0] - target)
    return audio[s:s + target]


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    wav, sr = sf.read(str(path), always_2d=False)
    return wav, sr


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def _positives_one_set(args: tuple) -> list[np.ndarray]:
    """Worker: load 1 DJ set, sample N random 8s windows, return list of mels.
    Each worker handles 1 set to avoid loading huge audio in main process.
    """
    set_path, n_samples, seed = args
    rnd = random.Random(seed)
    out: list[np.ndarray] = []
    try:
        audio, sr = load_audio(Path(set_path))
    except Exception as e:
        print(f"  skip {set_path}: {e}", flush=True)
        return out
    if audio.ndim > 1:
        audio = audio.mean(axis=0) if audio.shape[0] in (1, 2) else audio.mean(axis=-1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
        sr = SR
    target = int(WIN_SECONDS * sr)
    if audio.shape[0] < target:
        return out
    for _ in range(n_samples):
        s = rnd.randint(0, audio.shape[0] - target)
        out.append(to_mel(audio[s:s + target], sr))
    print(f"  + {Path(set_path).name}: {len(out)} windows", flush=True)
    return out


def _negative_splice_one(args: tuple) -> np.ndarray | None:
    """Worker: random splice of two clips. Each call loads only what it needs.

    Reuses a small process-local audio cache for the most-recent clip to avoid
    reloading the same file 100x.
    """
    clip_paths, seed = args
    rnd = random.Random(seed)
    half = WIN_SAMPLES // 2
    a, b = rnd.sample(clip_paths, 2)
    try:
        wa, sra = load_audio(Path(a))
        wb, srb = load_audio(Path(b))
    except Exception:
        return None
    import librosa
    def _to_mono_sr(w, sr):
        if w.ndim > 1:
            w = w.mean(axis=0) if w.shape[0] in (1, 2) else w.mean(axis=-1)
        if sr != SR:
            w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=SR)
        return w
    wa = _to_mono_sr(wa, sra)
    wb = _to_mono_sr(wb, srb)
    if wa.shape[0] < half or wb.shape[0] < half:
        return None
    sa = rnd.randint(0, wa.shape[0] - half)
    sb = rnd.randint(0, wb.shape[0] - half)
    spliced = np.concatenate([wa[sa:sa + half], wb[sb:sb + half]])
    return to_mel(spliced, SR)


def build_data(sets_dir: Path, clips_dir: Path, out_path: Path,
               n_pos: int = 1000, n_neg: int = 1000,
               workers: int = 0) -> None:
    set_paths = sorted(list(sets_dir.glob("*.mp3")) + list(sets_dir.glob("*.wav")))
    clip_paths = sorted(list(clips_dir.glob("*.wav")) + list(clips_dir.glob("*.mp3")))
    if len(set_paths) < 2:
        raise SystemExit(f"need >=2 DJ sets in {sets_dir}, got {len(set_paths)}")
    if len(clip_paths) < 4:
        raise SystemExit(f"need >=4 clips in {clips_dir}, got {len(clip_paths)}")

    if workers <= 0:
        import os as _os
        workers = max(2, min(8, (_os.cpu_count() or 4) - 1))
    print(f"sets: {len(set_paths)}, clips: {len(clip_paths)}, workers: {workers}")

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    # Positives: parallel — each worker handles 1 set, samples N windows.
    samples_per_set = max(1, n_pos // len(set_paths))
    pos_args = [(str(sp), samples_per_set, i) for i, sp in enumerate(set_paths)]
    print(f"positives: {samples_per_set} windows/set across {len(set_paths)} sets...")
    X_pos: list[np.ndarray] = []
    with ctx.Pool(processes=min(workers, len(set_paths))) as pool:
        for batch in pool.imap_unordered(_positives_one_set, pos_args):
            X_pos.extend(batch)
            if len(X_pos) >= n_pos:
                pool.terminate()
                break
    X_pos = X_pos[:n_pos]

    # Negatives: parallel — each worker does 1 random splice
    print(f"negatives: {n_neg} random splices...")
    neg_args = [([str(p) for p in clip_paths], i + 100000) for i in range(n_neg * 2)]
    X_neg: list[np.ndarray] = []
    with ctx.Pool(processes=workers) as pool:
        for r in pool.imap_unordered(_negative_splice_one, neg_args, chunksize=4):
            if r is not None:
                X_neg.append(r)
                if len(X_neg) >= n_neg:
                    pool.terminate()
                    break
    X_neg = X_neg[:n_neg]

    X_arr = np.stack(X_pos + X_neg).astype(np.float32)
    y_arr = np.concatenate([np.ones(len(X_pos), dtype=np.int64),
                            np.zeros(len(X_neg), dtype=np.int64)])
    print(f"X={X_arr.shape}, pos={len(X_pos)}, neg={len(X_neg)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), X=X_arr, y=y_arr)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MixCritic(nn.Module):
    def __init__(self, n_mels: int = N_MELS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.conv(x)
        return self.head(z).squeeze(-1)


def train(data_path: Path, ckpt: Path, epochs: int = 20,
          batch_size: int = 64, lr: float = 1e-3) -> None:
    npz = np.load(str(data_path))
    X = torch.from_numpy(npz["X"]).float()
    y = torch.from_numpy(npz["y"]).float()
    N = X.size(0)
    perm = torch.randperm(N)
    X, y = X[perm], y[perm]
    n_train = int(0.9 * N)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_va, y_va = X[n_train:], y[n_train:]
    print(f"train={n_train}, val={N - n_train}, X shape={X.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MixCritic().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size)

    best_acc = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss) * xb.size(0)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            val_loss = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += float(F.binary_cross_entropy_with_logits(logits, yb)) * xb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += int((preds == yb).sum())
                total += yb.size(0)
        train_loss = loss_sum / n_train
        val_loss /= max(1, total)
        val_acc = correct / max(1, total)
        if ep == 1 or ep % 2 == 0 or ep == epochs:
            print(f"  epoch {ep:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "n_mels": N_MELS,
                        "val_acc": val_acc, "epoch": ep}, str(ckpt))
    print(f"\nbest val_acc={best_acc:.3f} (saved {ckpt})")


# ---------------------------------------------------------------------------
# Score (inference)
# ---------------------------------------------------------------------------

def score(ckpt: Path, audio_path: Path) -> float:
    """Score an audio file. Returns mean P(real DJ mix) over all 8s windows."""
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model = MixCritic()
    model.load_state_dict(state["state_dict"])
    model.eval()
    audio, sr = load_audio(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=0) if audio.shape[0] in (1, 2) else audio.mean(axis=-1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
    target = int(WIN_SECONDS * SR)
    if audio.shape[0] < target:
        return 0.0
    # tile windows with 50% overlap
    hop = target // 2
    windows = []
    for s in range(0, audio.shape[0] - target + 1, hop):
        windows.append(to_mel(audio[s:s + target], SR))
    if not windows:
        return 0.0
    X = torch.from_numpy(np.stack(windows)).float()
    with torch.no_grad():
        probs = torch.sigmoid(model(X))
    return float(probs.mean().item())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-data")
    p.add_argument("--sets", required=True)
    p.add_argument("--clips", required=True)
    p.add_argument("--out", default="datasets/critic.npz")
    p.add_argument("--n_pos", type=int, default=1000)
    p.add_argument("--n_neg", type=int, default=1000)
    p.add_argument("--workers", type=int, default=0,
                   help="parallel CPU workers; 0=auto (cpu_count-1, max 8)")

    p = sub.add_parser("train")
    p.add_argument("--data", default="datasets/critic.npz")
    p.add_argument("--ckpt", default="checkpoints/mix_critic.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)

    p = sub.add_parser("score")
    p.add_argument("--ckpt", default="checkpoints/mix_critic.pt")
    p.add_argument("--audio", required=True)

    args = ap.parse_args()
    if args.cmd == "build-data":
        build_data(Path(args.sets), Path(args.clips), Path(args.out),
                   args.n_pos, args.n_neg, workers=args.workers)
    elif args.cmd == "train":
        train(Path(args.data), Path(args.ckpt), args.epochs,
              args.batch_size, args.lr)
    elif args.cmd == "score":
        s = score(Path(args.ckpt), Path(args.audio))
        print(json.dumps({"audio": str(args.audio), "real_mix_prob": s}, indent=2))


if __name__ == "__main__":
    main()
