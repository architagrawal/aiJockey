"""Audiobox-reward DPO over VampNet bridges.

Phase 1: generate N bridge candidates per (clip_a, clip_b) pair with
temperature sampling. Phase 2: score each with Audiobox. Phase 3: form
(winner, loser) bridge pairs and run a DPO-style preference update on
VampNet's coarse model.

Loss (DPO-on-token-sequences):
    L = -log σ(β · (log π(z_win|ctx) - log π(z_lose|ctx)
                     - log π_ref(z_win|ctx) + log π_ref(z_lose|ctx)))

References: DPO (Rafailov et al., 2023) adapted to discrete-token
generative models (cf. MR-FlowDPO for music, arXiv 2512.10264).

Run on droplet (after vampnet_finetune.py + Audiobox available):
    export AIJOCKEY_VAMPNET_ENABLE=1 AIJOCKEY_AUDIOBOX_AESTHETICS=1
    /opt/venv/bin/python scripts/vampnet_dpo.py \\
        --clips /workspace/user_set \\
        --pairs-per-clip 2 --candidates 4 \\
        --out /scratch/vampnet_dpo_coarse.pth \\
        --epochs 3
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _list_audio(d: str) -> list[Path]:
    p = Path(d)
    out: list[Path] = []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.m4a"):
        out.extend(p.rglob(ext))
    return sorted(set(out))


def _generate_candidate(iface, a_path: Path, b_path: Path, gap_s: float,
                          context_s: float, sr: int, temperature: float,
                          top_p: float):
    """Returns (tokens_z, decoded_signal, mask) for one bridge sample."""
    import librosa
    import numpy as np
    import torch
    from audiotools import AudioSignal  # type: ignore
    a_full, _ = librosa.load(str(a_path), sr=sr, mono=True)
    b_full, _ = librosa.load(str(b_path), sr=sr, mono=True)
    n_ctx = int(sr * context_s)
    a_tail = a_full[-n_ctx:] if len(a_full) >= n_ctx else a_full
    b_head = b_full[:n_ctx] if len(b_full) >= n_ctx else b_full
    n_gap = int(sr * gap_s)
    full_in = np.concatenate([a_tail, np.zeros(n_gap, dtype=np.float32), b_head])
    sig = AudioSignal(full_in[None, :], sample_rate=sr).to(iface.device)
    z = iface.encode(sig)
    n_tokens = z.shape[-1]
    a_tok = int(round(len(a_tail) / len(full_in) * n_tokens))
    g_tok = int(round(n_gap / len(full_in) * n_tokens))
    mask = torch.zeros_like(z, dtype=torch.long)
    mask[..., a_tok : a_tok + g_tok] = 1
    z_out = iface.coarse_to_fine(
        iface.coarse_vamp(z, mask=mask, temperature=temperature, top_p=top_p),
        temperature=temperature,
    )
    sig_out = iface.decode(z_out)
    return z_out, sig_out, mask, (a_tok, g_tok)


def _save_wav(sig, tmp_dir: Path, name: str) -> Path:
    import torch
    import torchaudio
    samples = sig.samples.detach().cpu()
    if samples.dim() == 3:
        samples = samples.squeeze(0)
    wav = samples.mean(0, keepdim=False) if samples.shape[0] == 2 else samples[0]
    out = tmp_dir / f"{name}.wav"
    torchaudio.save(str(out), wav.unsqueeze(0), int(sig.sample_rate))
    return out


def _audiobox_pq(p: Path) -> float | None:
    try:
        from audiobox_critic import enabled, score
        if not enabled():
            return None
        s = score(str(p))
        if s:
            return float(s.get("PQ", 0.0)) + float(s.get("CE", 0.0))
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True,
                    help="dir with user audio clips")
    ap.add_argument("--pairs-per-clip", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--context", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--min-delta", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vampnet_wrapper import enabled, _load
    if not enabled():
        sys.exit("AIJOCKEY_VAMPNET_ENABLE != 1; aborting")
    pipe = _load()
    if pipe is None:
        sys.exit("VampNet load failed")
    iface = pipe["interface"]
    sr = int(pipe["sr"])

    import torch
    import torch.nn.functional as F

    # Reference (frozen) coarse for DPO. deepcopy fails on weight_norm
    # parametrized modules; instead snapshot state_dict and create a
    # fresh-architecture clone via VampNet's interface, then load.
    coarse = iface.coarse
    ref_state = {k: v.detach().clone() for k, v in coarse.state_dict().items()}
    try:
        # Reconstruct ref via the same VampNet class. _load_model exists
        # in vampnet/interface.py; fall back to a copied module via
        # parametrized-safe deepcopy fallback.
        from vampnet.interface import _load_model  # type: ignore
        ref = _load_model(
            ckpt=str(iface.coarse_path),
            lora_ckpt=None,
            device=iface.device,
            chunk_size_s=getattr(coarse, "chunk_size_s", 10),
        )
        ref.load_state_dict(ref_state)
    except Exception:
        # Last-ditch: torch.save+load to a temp file to clone
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
            torch.save(ref_state, tf.name)
            ref_path = tf.name
        ref = type(coarse).__new__(type(coarse))
        ref.__dict__ = {k: v for k, v in coarse.__dict__.items()}
        ref.load_state_dict(torch.load(ref_path, weights_only=False))
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(coarse.parameters(), lr=args.lr,
                                weight_decay=0.0)

    files = _list_audio(args.clips)
    if len(files) < 2:
        sys.exit(f"need >= 2 audio files in {args.clips}")
    print(f"[dpo] {len(files)} clips")

    # Build (a, b) pairs
    import random
    pairs = []
    for a in files:
        others = [x for x in files if x != a]
        random.shuffle(others)
        for b in others[: args.pairs_per_clip]:
            pairs.append((a, b))

    pair_records = []
    tmp = Path(tempfile.mkdtemp(prefix="vampnet_dpo_"))
    print(f"[dpo] generating candidates -> {tmp}")
    for n, (a, b) in enumerate(pairs):
        cands = []
        for k in range(args.candidates):
            try:
                z, sig, mask, (a_tok, g_tok) = _generate_candidate(
                    iface, a, b, args.gap, args.context, sr,
                    args.temperature, args.top_p,
                )
            except Exception as e:
                print(f"[dpo] gen fail {a.stem}->{b.stem} #{k}: {e}")
                continue
            wav_p = _save_wav(sig, tmp, f"p{n:03d}_k{k}")
            pq = _audiobox_pq(wav_p)
            if pq is None:
                pq = 0.0
            cands.append({"z": z.detach().cpu(),
                           "mask": mask.detach().cpu(),
                           "a_tok": a_tok, "g_tok": g_tok,
                           "wav": wav_p, "pq": pq})
        if len(cands) < 2:
            continue
        cands.sort(key=lambda c: -c["pq"])
        delta = cands[0]["pq"] - cands[-1]["pq"]
        if delta < args.min_delta:
            continue
        pair_records.append({"a": str(a), "b": str(b),
                              "win": cands[0], "lose": cands[-1],
                              "delta": delta})
        print(f"[dpo] pair {n+1}/{len(pairs)} delta={delta:.2f}")
    print(f"[dpo] {len(pair_records)} preference pairs")

    if not pair_records:
        sys.exit("no preference pairs above min-delta")

    def _logp(model, z, mask):
        """Sum of log-probs of the masked positions under model.

        VampNet.forward expects latents not tokens (per finetune fix):
            latents = embedding.from_codes(z, codec)
        Output shape: (B, vocab, T*n_pred) interleaved time-fastest.
        """
        z = z.to(iface.device)
        mask = mask.to(iface.device)
        n_cb = model.n_codebooks
        n_pred = model.n_predict_codebooks
        z_cb = z[:, :n_cb, :]
        mask_cb = mask[:, :n_cb, :]
        # Mask token ids before latent conversion
        mask_token = int(getattr(model, "mask_token", 1024))
        z_masked = z_cb.long().clone()
        z_masked[mask_cb.bool()] = mask_token
        with torch.no_grad():
            latents = model.embedding.from_codes(z_masked, iface.codec)
        logits = model(latents)   # (B, V, T*n_pred)
        B, V, TC = logits.shape
        T_actual = TC // n_pred
        # Target codebooks
        tgt = z[:, :n_pred, :T_actual].long()
        mask_pred = mask[:, :n_pred, :T_actual].float()
        # logits permute → (B, T*n_pred, V); reshape to (B, T, n_pred, V)
        logits = logits.permute(0, 2, 1).reshape(B, T_actual, n_pred, V)
        logp = F.log_softmax(logits, dim=-1)
        # target: (B, n_pred, T) → (B, T, n_pred)
        tgt_t = tgt.permute(0, 2, 1)
        gathered = torch.gather(logp, dim=-1,
                                 index=tgt_t.unsqueeze(-1)).squeeze(-1)
        # mask_pred: (B, n_pred, T) → (B, T, n_pred)
        mask_t = mask_pred.permute(0, 2, 1)
        return (gathered * mask_t).sum()

    for ep in range(args.epochs):
        ep_loss = 0.0
        for rec in pair_records:
            optim.zero_grad()
            zw, mw = rec["win"]["z"], rec["win"]["mask"]
            zl, ml = rec["lose"]["z"], rec["lose"]["mask"]
            lp_win = _logp(coarse, zw, mw)
            lp_lose = _logp(coarse, zl, ml)
            with torch.no_grad():
                lp_win_ref = _logp(ref, zw, mw)
                lp_lose_ref = _logp(ref, zl, ml)
            logits = args.beta * ((lp_win - lp_win_ref) - (lp_lose - lp_lose_ref))
            loss = -F.logsigmoid(logits)
            loss.backward()
            optim.step()
            ep_loss += float(loss.item())
        print(f"[dpo] epoch {ep+1}/{args.epochs} loss={ep_loss/len(pair_records):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": coarse.state_dict(),
                 "n_pairs": len(pair_records),
                 "epochs": args.epochs,
                 "beta": args.beta},
                args.out)
    print(f"[dpo] saved DPO-tuned coarse -> {args.out}")


if __name__ == "__main__":
    main()
