"""Train MERT-95M → 4-axis Audiobox-PQ regressor head from render logs.

Run on droplet:
    export AIJOCKEY_MERT_REWARD_ENABLE=1
    /opt/venv/bin/python scripts/train_mert_reward.py \\
        --jsonl /scratch/probes/log.jsonl \\
        --audio-root /workspace \\
        --out /scratch/mert_reward.pt \\
        --epochs 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/scratch/probes/log.jsonl")
    ap.add_argument("--audio-root", default="/workspace")
    ap.add_argument("--out", default="/scratch/mert_reward.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from mert_reward import train_from_jsonl
    res = train_from_jsonl(args.jsonl, args.audio_root, args.out,
                             epochs=args.epochs, lr=args.lr,
                             batch_size=args.batch_size,
                             device=args.device)
    print(res)


if __name__ == "__main__":
    main()
