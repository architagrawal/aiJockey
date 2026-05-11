"""Continue training VampNet's coarse model on user pool + library.

Specializes the masked-LM to user's musical taste. Uses VampNet's own
trainer infrastructure where available, else implements a minimal
masked-prediction loop on top of the codec-encoded tokens.

Strategy:
    1. Encode all audio in --clips dir through VampNet codec → save
       token tensors at /scratch/vampnet_tokens/<id>.pt.
    2. Run masked-prediction fine-tuning of the coarse model with
       random span masking. Reuse VampNet's `vampnet.train` API when
       available; otherwise minimal AdamW + masked-CE loop.
    3. Save adapted checkpoint to AIJOCKEY_VAMPNET_CKPT_DIR/coarse.pth
       (or new dir, env-configurable).

Run on droplet:
    export AIJOCKEY_VAMPNET_ENABLE=1
    /opt/venv/bin/python scripts/vampnet_finetune.py \\
        --clips /workspace/user_set /workspace/user_genres \\
        --out  /scratch/vampnet_finetuned/coarse.pth \\
        --epochs 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _gather_audio_files(dirs: list[str]) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for ext in ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"):
            out.extend(p.rglob(ext))
    return sorted(set(out))


def _encode_tokens(audio_paths: list[Path], iface, sr: int,
                    token_dir: Path) -> list[Path]:
    import torch
    import librosa
    from audiotools import AudioSignal  # type: ignore
    token_dir.mkdir(parents=True, exist_ok=True)
    tok_paths: list[Path] = []
    for p in audio_paths:
        out_p = token_dir / f"{p.stem}.pt"
        if out_p.exists():
            tok_paths.append(out_p); continue
        try:
            wav, _ = librosa.load(str(p), sr=sr, mono=True, duration=30.0)
            sig = AudioSignal(wav[None, :], sample_rate=sr).to(iface.device)
            z = iface.encode(sig).cpu()
            torch.save(z, out_p)
            tok_paths.append(out_p)
        except Exception as e:
            print(f"[ft] encode fail {p}: {e}")
    return tok_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True,
                    help="dirs to scan for audio")
    ap.add_argument("--out", required=True,
                    help="path to write fine-tuned coarse checkpoint")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--mask-frac", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--token-dir", default="/scratch/vampnet_tokens")
    args = ap.parse_args()

    from vampnet_wrapper import enabled, _load
    if not enabled():
        sys.exit("AIJOCKEY_VAMPNET_ENABLE != 1; aborting")
    pipe = _load()
    if pipe is None:
        sys.exit("VampNet interface failed to load")
    iface = pipe["interface"]
    sr = int(pipe["sr"])

    paths = _gather_audio_files(args.clips)
    print(f"[ft] {len(paths)} audio files")
    if not paths:
        sys.exit("no audio inputs")

    tok_dir = Path(args.token_dir)
    tok_paths = _encode_tokens(paths, iface, sr, tok_dir)
    print(f"[ft] encoded {len(tok_paths)} -> {tok_dir}")
    if len(tok_paths) < 2:
        sys.exit("insufficient tokens for training")

    import torch
    import torch.nn.functional as F

    coarse = iface.coarse   # vampnet model module
    optim = torch.optim.AdamW(coarse.parameters(), lr=args.lr,
                                weight_decay=1e-4)
    history = []

    # Load all token tensors into memory (small).
    toks = [torch.load(p, map_location="cpu") for p in tok_paths]
    # Each is shape (1, n_codebooks, T). Strip batch dim.
    toks = [t.squeeze(0) for t in toks]

    n_cb = toks[0].shape[0]
    mask_token = int(getattr(coarse, "mask_token", 1024))   # default 1024

    for ep in range(args.epochs):
        ep_loss = 0.0
        ep_steps = 0
        # Simple shuffle-and-step
        import numpy as np
        idx = np.random.permutation(len(toks))
        for s in range(0, len(toks), args.batch_size):
            batch_idx = idx[s : s + args.batch_size]
            batch = []
            for bi in batch_idx:
                t = toks[bi]
                if t.shape[-1] < 32:
                    continue
                # Crop to args.max_tokens random window
                T = t.shape[-1]
                if T > args.max_tokens:
                    st = int(np.random.randint(0, T - args.max_tokens))
                    t = t[..., st : st + args.max_tokens]
                batch.append(t)
            if not batch:
                continue
            # Pad to same length
            maxT = max(b.shape[-1] for b in batch)
            stacked = torch.stack(
                [F.pad(b, (0, maxT - b.shape[-1]), value=0) for b in batch],
                dim=0,
            ).to(iface.device)

            # Build random mask
            target = stacked.clone()
            mask = (torch.rand_like(stacked.float()) < args.mask_frac)
            inp = stacked.masked_fill(mask, mask_token)

            optim.zero_grad()
            # VampNet.forward() expects LATENTS (float), not token ids.
            # Token-id-to-latent conversion: coarse.embedding.from_codes(z, codec).
            # This is the key fix — passing long tokens directly hits
            # CodebookEmbedding.out_proj (Conv1d) with wrong dtype.
            inp = inp.long()
            with torch.no_grad():
                # Truncate inp to coarse's n_codebooks (4 in vampnet).
                n_cb = coarse.n_codebooks
                inp_cb = inp[:, :n_cb, :]
                latents = coarse.embedding.from_codes(inp_cb, iface.codec)
            logits = coarse(latents)
            if logits.dim() == 3:
                logits = logits.unsqueeze(1)
            # Match logits target shape to coarse's n_predict_codebooks
            target_cb = target[:, :coarse.n_predict_codebooks if hasattr(
                coarse, "n_predict_codebooks") else n_cb, :].long()
            loss = F.cross_entropy(
                logits.permute(0, 3, 1, 2),
                target_cb,
                ignore_index=-100,
                reduction="mean",
            )
            loss.backward()
            optim.step()
            ep_loss += float(loss.item())
            ep_steps += 1
        history.append(ep_loss / max(1, ep_steps))
        print(f"[ft] epoch {ep+1}/{args.epochs} loss={history[-1]:.4f}")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": coarse.state_dict(),
                 "history": history,
                 "epochs": args.epochs,
                 "n_train": len(toks)},
                str(out_p))
    print(f"[ft] saved coarse ckpt -> {out_p}")


if __name__ == "__main__":
    main()
