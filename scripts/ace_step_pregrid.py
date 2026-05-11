"""Pre-generate ACE-Step bridge clips across a BPM x key x mood grid.

Output: WAV files into target dir, sidecar JSON with metadata.
After generation, runs Audiobox Aesthetics on each → discards PQ < 6.5
so picker only sees high-quality bridges.

Default grid:
    BPM:   120 128 140
    Keys:  12 Camelot keys (A and B sides)
    Moods: ['atmospheric pad uplifting trance',
             'dark techno warehouse drone',
             'bright melodic future bass shimmer',
             'deep house warm chord progression']
    Duration: 24s (long enough for ~8 bars at 128 BPM, short enough to mix)

Run on MI300X:
    export AIJOCKEY_ACE_STEP_ENABLE=1 AIJOCKEY_AUDIOBOX_AESTHETICS=1
    /opt/venv/bin/python scripts/ace_step_pregrid.py \\
        --out /cache/ace_step_bridges \\
        --bpms 120,128,140 --duration 24
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


DEFAULT_MOODS = {
    "atmos_trance": "atmospheric pad uplifting trance, lush synth strings, no drums, no vocals",
    "dark_techno": "dark techno warehouse drone, industrial atmosphere, no drums, no vocals",
    "bright_future": "bright melodic future bass shimmer, sparkly arpeggio, no drums, no vocals",
    "deep_house": "deep house warm chord progression, soft pads, no drums, no vocals",
}
DEFAULT_KEYS = [f"{n}{ab}" for n in range(1, 13) for ab in ("A", "B")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bpms", default="120,128,140",
                    help="comma-separated target BPMs")
    ap.add_argument("--keys", default=",".join(DEFAULT_KEYS),
                    help="comma-separated Camelot keys")
    ap.add_argument("--moods", default=",".join(DEFAULT_MOODS.keys()),
                    help="comma-separated mood slugs (see DEFAULT_MOODS)")
    ap.add_argument("--duration", type=float, default=24.0)
    ap.add_argument("--steps", type=int, default=27)
    ap.add_argument("--min-pq", type=float, default=6.5,
                    help="discard bridges with Audiobox PQ below this")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap total generations (0 = full grid)")
    ap.add_argument("--seed-base", type=int, default=42)
    args = ap.parse_args()

    from ace_step_wrapper import enabled as ace_enabled, generate_bridge, camelot_to_letter
    if not ace_enabled():
        sys.exit("AIJOCKEY_ACE_STEP_ENABLE != 1; aborting")

    try:
        from audiobox_critic import enabled as aae_enabled, score as aae_score
    except Exception:
        aae_enabled = lambda: False
        aae_score = lambda _p: None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rejected = out / "_rejected"
    rejected.mkdir(parents=True, exist_ok=True)

    bpms = [int(b.strip()) for b in args.bpms.split(",") if b.strip()]
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    moods = [m.strip() for m in args.moods.split(",") if m.strip() in DEFAULT_MOODS]
    if not moods:
        moods = list(DEFAULT_MOODS.keys())

    grid = [(bpm, key, mood) for bpm in bpms for key in keys for mood in moods]
    if args.limit:
        grid = grid[: args.limit]
    print(f"[ace_pregrid] {len(grid)} bridges to generate "
          f"({len(bpms)} bpm x {len(keys)} keys x {len(moods)} moods)")

    manifest = []
    t_all = time.perf_counter()
    for i, (bpm, key, mood) in enumerate(grid):
        slug = f"bridge_{bpm}bpm_{key}_{mood}"
        target = out / f"{slug}.wav"
        if target.exists():
            print(f"[{i+1}/{len(grid)}] skip {slug} (exists)")
            continue
        caption = DEFAULT_MOODS[mood]
        key_letter = camelot_to_letter(key)
        t0 = time.perf_counter()
        res = generate_bridge(
            bpm=bpm, key=key_letter,
            duration_seconds=args.duration,
            caption=caption,
            negative_prompt="drums, kick, snare, hi-hat, vocals, singing",
            steps=args.steps,
            out_path=target,
            seed=args.seed_base + i,
        )
        dt = time.perf_counter() - t0
        if not res:
            print(f"[{i+1}/{len(grid)}] FAILED {slug}")
            continue

        # Audiobox score + auto-curate
        scores = aae_score(target) if aae_enabled() else None
        pq = float((scores or {}).get("PQ", 0.0))
        entry = {
            "bpm": bpm, "key_camelot": key, "key_letter": key_letter,
            "mood": mood, "caption": caption,
            "duration_s": args.duration,
            "audiobox": scores,
            "gen_seconds": round(dt, 2),
        }
        if scores and pq < args.min_pq:
            # Move to rejected so picker never sees it.
            new = rejected / target.name
            target.rename(new)
            entry["path"] = str(new)
            entry["rejected"] = True
            print(f"[{i+1}/{len(grid)}] reject {slug} PQ={pq:.2f} ({dt:.1f}s)")
        else:
            entry["path"] = str(target)
            entry["rejected"] = False
            print(f"[{i+1}/{len(grid)}] keep   {slug} "
                  f"PQ={pq:.2f} ({dt:.1f}s)")
        manifest.append(entry)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    kept = sum(1 for m in manifest if not m["rejected"])
    print(f"[ace_pregrid] done in {(time.perf_counter()-t_all)/60:.1f} min: "
          f"{kept} kept / {len(manifest)} generated")


if __name__ == "__main__":
    main()
