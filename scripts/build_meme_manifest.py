"""Scan samples/memes/ and generate manifest.json so SampleBank loads them.

Each meme is registered with fx_type='meme', BPM-agnostic, length-agnostic.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEMES_DIR = ROOT / "samples" / "memes"
MANIFEST = ROOT / "samples" / "manifest.json"


def main() -> None:
    if not MEMES_DIR.exists():
        print(f"no memes dir at {MEMES_DIR}")
        return
    existing = []
    if MANIFEST.exists():
        try:
            with open(MANIFEST) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    # de-dup against existing entries
    have = {e.get("file") for e in existing}
    added = 0
    for wav in sorted(MEMES_DIR.glob("*.wav")):
        rel = f"memes/{wav.name}"
        if rel in have:
            continue
        # tag = part before "__"
        tag = wav.stem.split("__")[0] if "__" in wav.stem else wav.stem
        existing.append({
            "file": rel,
            "type": "meme",
            "tag": tag,
            "bpm": "agnostic",
            "length_beats": None,
            "key": None,
        })
        added += 1
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"wrote {MANIFEST}: total {len(existing)} entries (+{added} new memes)")


if __name__ == "__main__":
    main()
