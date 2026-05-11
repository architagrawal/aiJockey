"""MERT-95M embedding → 4-axis Audiobox-PQ regressor head.

Trains a tiny MLP on our existing JSONL render logs:
    input  = mean-pooled MERT-v1-95M embedding of the rendered mix
             (or a 3-second junction window if junction provided)
    output = 4 scalars predicting Audiobox PQ/PC/CE/CU

Used at plan-time as an additional picker score term — gives the
picker a "future PQ guess" before committing to a clip choice, derived
from real render outcomes.

Architecture:
    MERT-v1-95M (frozen) → [layer 6-8 mean] → MLP(768→256→128→4)

Training:
    - Read /scratch/probes/log.jsonl (or env path)
    - For each render with audiobox scores, compute MERT embedding of
      the rendered mix (resampled to 24 kHz mono).
    - Train MSE regression for ~10 epochs with weight decay.

Inference (called from picker):
    - reward_head_score(clip_id, cache_dir) -> 4-axis prediction
    - Cached per clip-id+section.

Env knobs:
    AIJOCKEY_MERT_REWARD_ENABLE   0|1  default 0
    AIJOCKEY_MERT_REWARD_CKPT     path default '/scratch/mert_reward.pt'
    AIJOCKEY_MERT_REWARD_MODEL    hf id default 'm-a-p/MERT-v1-95M'
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_PIPE: dict | None = None
_HEAD: Any = None
_LOAD_FAILED = False


def enabled() -> bool:
    if os.environ.get("AIJOCKEY_MERT_REWARD_ENABLE", "0") != "1":
        return False
    return not _LOAD_FAILED


def _load_mert(device: str | None = None):
    """Lazy-load MERT encoder once."""
    global _PIPE, _LOAD_FAILED
    if _PIPE is not None:
        return _PIPE
    if _LOAD_FAILED:
        return None
    with _LOCK:
        if _PIPE is not None:
            return _PIPE
        try:
            import torch
            from transformers import AutoModel, AutoFeatureExtractor  # type: ignore
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = os.environ.get("AIJOCKEY_MERT_REWARD_MODEL",
                                       "m-a-p/MERT-v1-95M")
            proc = AutoFeatureExtractor.from_pretrained(model_id,
                                                          trust_remote_code=True)
            model = AutoModel.from_pretrained(model_id,
                                                trust_remote_code=True)
            model.eval()
            if device == "cuda":
                model = model.cuda()
            _PIPE = {"model": model, "proc": proc, "device": device}
            return _PIPE
        except Exception as e:
            print(f"[mert_reward] MERT load failed: {e}")
            _LOAD_FAILED = True
            return None


def _embed(audio_path: str, device: str = "cuda") -> "Any | None":
    pipe = _load_mert(device=device)
    if pipe is None:
        return None
    try:
        import librosa
        import torch
        wav, _ = librosa.load(audio_path, sr=24000, mono=True, duration=30.0)
        inputs = pipe["proc"](wav, sampling_rate=24000, return_tensors="pt")
        if pipe["device"] == "cuda":
            inputs = {k: v.cuda() if hasattr(v, "cuda") else v
                       for k, v in inputs.items()}
        with torch.inference_mode():
            out = pipe["model"](**inputs, output_hidden_states=True)
        # Layer 6-8 mean pool — strongest production-quality signal per
        # MERT paper.
        hs = out.hidden_states[6:9]   # tuple of (1, T, D)
        stack = torch.stack(hs, dim=0).mean(0)   # (1, T, D)
        emb = stack.mean(dim=1).squeeze(0)       # (D,)
        return emb.detach().cpu().float().numpy()
    except Exception as e:
        print(f"[mert_reward] embed failed: {e}")
        return None


class _RewardHead:
    """Tiny MLP: input D=768 → 4-axis prediction."""

    def __init__(self, in_dim: int = 768):
        import torch.nn as nn
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.net(x)

    def __call__(self, x):
        return self.forward(x)


def _load_head(ckpt_path: str | None = None):
    global _HEAD
    if _HEAD is not None:
        return _HEAD
    import torch
    path = ckpt_path or os.environ.get("AIJOCKEY_MERT_REWARD_CKPT",
                                          "/scratch/mert_reward.pt")
    if not Path(path).exists():
        print(f"[mert_reward] ckpt missing: {path}")
        return None
    try:
        state = torch.load(path, map_location="cpu")
        head = _RewardHead(in_dim=state.get("in_dim", 768))
        head.net.load_state_dict(state["state_dict"])
        head.net.eval()
        _HEAD = head
        return head
    except Exception as e:
        print(f"[mert_reward] head load failed: {e}")
        return None


def predict(audio_path: str, device: str = "cuda") -> dict | None:
    """Return predicted {PQ, PC, CE, CU} for an audio file, or None."""
    if not enabled():
        return None
    head = _load_head()
    if head is None:
        return None
    emb = _embed(audio_path, device=device)
    if emb is None:
        return None
    try:
        import torch
        x = torch.from_numpy(emb).float().unsqueeze(0)
        with torch.no_grad():
            y = head.forward(x).squeeze(0).numpy()
        return {"PQ": float(y[0]), "PC": float(y[1]),
                "CE": float(y[2]), "CU": float(y[3])}
    except Exception as e:
        print(f"[mert_reward] predict failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Training loop — used by scripts/train_mert_reward.py
# ---------------------------------------------------------------------------

def train_from_jsonl(jsonl_path: str,
                      audio_root: str,
                      out_ckpt: str,
                      epochs: int = 10,
                      lr: float = 1e-3,
                      batch_size: int = 16,
                      device: str = "cuda") -> dict:
    """Train head from /scratch/probes/log.jsonl entries with audiobox.

    Expects each line to be a JSON dict including keys:
        job_id, audiobox_pq, audiobox_pc, audiobox_ce, audiobox_cu, output_path
    (or `audiobox` dict). Falls back to per-key flat names if dict absent.
    """
    import torch
    import torch.nn as nn
    import numpy as np
    from pathlib import Path as _P
    pipe = _load_mert(device=device)
    if pipe is None:
        return {"status": "mert_unavailable"}

    # Gather labelled samples
    rows: list[tuple[str, list[float]]] = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            aae = r.get("audiobox") or {
                "PQ": r.get("audiobox_pq") or r.get("pq"),
                "PC": r.get("audiobox_pc") or r.get("pc"),
                "CE": r.get("audiobox_ce") or r.get("ce"),
                "CU": r.get("audiobox_cu") or r.get("cu"),
            }
            try:
                pq = float(aae["PQ"]); pc = float(aae["PC"])
                ce = float(aae["CE"]); cu = float(aae["CU"])
            except Exception:
                continue
            p = r.get("output_path") or r.get("audio_path") or r.get("path")
            if not p:
                continue
            ap = _P(p)
            if not ap.is_absolute():
                ap = _P(audio_root) / ap
            if ap.exists():
                rows.append((str(ap), [pq, pc, ce, cu]))
    if len(rows) < 8:
        return {"status": "insufficient_data", "n": len(rows)}

    # Embed all (cache to disk later if needed)
    X = []
    Y = []
    for ap, lbl in rows:
        emb = _embed(ap, device=device)
        if emb is None:
            continue
        X.append(emb); Y.append(lbl)
    if len(X) < 8:
        return {"status": "embed_failed", "n_ok": len(X)}
    X = np.stack(X).astype(np.float32)
    Y = np.array(Y, dtype=np.float32)
    in_dim = X.shape[1]

    head = _RewardHead(in_dim=in_dim)
    net = head.net
    if device == "cuda":
        net = net.cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    n = len(X)
    idx = np.arange(n)
    history = []
    for ep in range(epochs):
        np.random.shuffle(idx)
        total = 0.0
        steps = 0
        for s in range(0, n, batch_size):
            sl = idx[s : s + batch_size]
            xb = torch.from_numpy(X[sl])
            yb = torch.from_numpy(Y[sl])
            if device == "cuda":
                xb = xb.cuda(); yb = yb.cuda()
            opt.zero_grad()
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        history.append(total / max(1, steps))
        print(f"[mert_reward] epoch {ep+1}/{epochs} loss={history[-1]:.4f}")

    torch.save({"in_dim": in_dim,
                 "state_dict": net.state_dict(),
                 "history": history,
                 "n_train": n},
                out_ckpt)
    return {"status": "ok", "out_ckpt": out_ckpt,
             "n_train": n, "final_loss": history[-1]}
