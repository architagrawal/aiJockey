"""S8 — generative bridge fine-tune.

Fine-tune MusicGen-Small on (pre, transition, post) triplets from
/scratch/transitions/. Bridge model generates 4-bar transition audio
between incompat-key/BPM clips at inference (S9 final-render).

Phase A polish §14.2 P2 (hierarchical generation), §16.2 D-G (efficiency).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pipeline.common import scratch_dir


MIN_TRIPLETS = 200
RETRAIN_INTERVAL_SEC = 3600


def _list_triplets() -> list[Path]:
    return list(scratch_dir('transitions').rglob('t*.json'))


def fine_tune(epoch: int) -> bool:
    """Single fine-tune pass on MusicGen-Small. Reconstruction loss on
    the transition window conditioned on (pre + post) audio context.
    """
    triplets = _list_triplets()
    if len(triplets) < MIN_TRIPLETS:
        print(f"S8 only {len(triplets)} triplets (< {MIN_TRIPLETS}); waiting")
        return False
    try:
        import torch
        from transformers import MusicgenForConditionalGeneration, AutoProcessor
        from training.efficiency import (autocast_ctx, get_dtype, maybe_compile,
                                          make_optimizer, hf_attn_implementation)
        from training.augment import AugChain
    except ImportError as e:
        print(f"S8 deps missing ({e}); skip")
        return False

    base = os.getenv('AIJOCKEY_BRIDGE_BASE', 'facebook/musicgen-small')
    print(f"S8 fine-tune {base} on {len(triplets)} triplets (epoch {epoch})")

    try:
        proc = AutoProcessor.from_pretrained(base)
        mdl = MusicgenForConditionalGeneration.from_pretrained(
            base, attn_implementation=hf_attn_implementation(),
            torch_dtype=get_dtype(),
        )
    except Exception as e:
        print(f"S8 model load failed ({e})")
        return False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mdl = mdl.to(device)
    mdl = maybe_compile(mdl)
    mdl.train()

    opt = make_optimizer(mdl.parameters(), lr=1e-5)
    aug = AugChain(p_pitch=0.3, p_stretch=0.3, p_gain=0.5, p_codec=0.2)

    # Real implementation: load each triplet's audio, slice (pre|trans|post),
    # tokenize via EnCodec, run reconstruction loss on trans tokens
    # conditioned on pre-context. Stub returns True and saves an empty
    # adapter directory until full audio loader is wired.
    out_dir = scratch_dir('models') / f'bridge_musicgen_e{epoch:03d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        mdl.save_pretrained(str(out_dir))
        print(f"S8 saved {out_dir} (stub training pass — wire full audio loader for production)")
        return True
    except Exception as e:
        print(f"S8 save failed ({e})")
        return False


def watch_loop(interval_sec: float) -> None:
    epoch = 0
    while True:
        if fine_tune(epoch + 1):
            epoch += 1
        time.sleep(interval_sec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', type=float, default=RETRAIN_INTERVAL_SEC)
    args = ap.parse_args()
    watch_loop(args.interval)


if __name__ == '__main__':
    main()
