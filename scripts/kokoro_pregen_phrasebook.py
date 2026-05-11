"""Pre-render Kokoro DJ-tag phrasebook to a cache dir.

Run once per deploy. Output WAVs become zero-latency tag overlays at
mix-render time.

Usage (on MI300X):
    pip install kokoro soundfile librosa
    apt-get install -y espeak-ng
    export AIJOCKEY_KOKORO_ENABLE=1
    /opt/venv/bin/python scripts/kokoro_pregen_phrasebook.py \\
        --out /cache/kokoro_phrasebook
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-sr", type=int, default=44100)
    args = ap.parse_args()

    from kokoro_tags import enabled, render_phrasebook
    if not enabled():
        sys.exit("AIJOCKEY_KOKORO_ENABLE != 1; aborting")
    manifest = render_phrasebook(args.out, target_sr=args.target_sr)
    print(f"[kokoro_pregen] {len(manifest)} entries written to {args.out}")


if __name__ == "__main__":
    main()
