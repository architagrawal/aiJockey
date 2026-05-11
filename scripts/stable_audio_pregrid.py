"""Pre-generate Stable Audio Open bridges across BPM × key × mood.

Mirrors scripts/ace_step_pregrid.py but uses Stable Audio Open. After
generation, scores each bridge with Audiobox and discards PQ < min-pq.

Run on MI300X:
    export AIJOCKEY_STABLE_AUDIO_ENABLE=1 AIJOCKEY_AUDIOBOX_AESTHETICS=1
    /opt/venv/bin/python scripts/stable_audio_pregrid.py \\
        --out /cache/stable_audio_bridges
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
    "atmos_trance": "atmospheric pad uplifting trance, lush synths, instrumental, no vocals",
    "dark_techno": "dark techno warehouse drone, industrial atmosphere, instrumental",
    "bright_future": "bright melodic future bass shimmer, sparkly arpeggio, instrumental",
    "deep_house": "deep house warm chord progression, soft pads, instrumental",
}
DEFAULT_KEYS = ["C minor", "F minor", "A minor", "D minor",
                 "G minor", "B minor", "E minor", "F# major",
                 "C major", "G major", "D major", "A major"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bpms", default="120,128,140")
    ap.add_argument("--keys", default=",".join(DEFAULT_KEYS))
    ap.add_argument("--moods", default=",".join(DEFAULT_MOODS.keys()))
    ap.add_argument("--duration", type=float, default=24.0)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--min-pq", type=float, default=6.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed-base", type=int, default=42)
    args = ap.parse_args()

    from stable_audio_wrapper import enabled, generate_bridge
    if not enabled():
        sys.exit("AIJOCKEY_STABLE_AUDIO_ENABLE != 1; aborting")
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
    grid = [(b, k, m) for b in bpms for k in keys for m in moods]
    if args.limit:
        grid = grid[: args.limit]
    print(f"[stable_audio_pregrid] {len(grid)} bridges")

    manifest = []
    t_all = time.perf_counter()
    for i, (bpm, key, mood) in enumerate(grid):
        slug = f"sa_bridge_{bpm}bpm_{key.replace(' ', '_')}_{mood}"
        target = out / f"{slug}.wav"
        if target.exists():
            print(f"[{i+1}/{len(grid)}] skip {slug}")
            continue
        caption = DEFAULT_MOODS[mood]
        t0 = time.perf_counter()
        res = generate_bridge(
            bpm=bpm, key=key, duration_seconds=args.duration,
            caption=caption,
            negative_prompt="vocals, singing, voice, drums, kick",
            steps=args.steps, guidance_scale=args.cfg,
            out_path=target, seed=args.seed_base + i,
        )
        dt = time.perf_counter() - t0
        if not res:
            print(f"[{i+1}/{len(grid)}] FAILED {slug}")
            continue
        scores = aae_score(target) if aae_enabled() else None
        pq = float((scores or {}).get("PQ", 0.0))
        entry = {"bpm": bpm, "key": key, "mood": mood, "caption": caption,
                  "duration_s": args.duration, "audiobox": scores,
                  "gen_seconds": round(dt, 2)}
        if scores and pq < args.min_pq:
            new = rejected / target.name
            target.rename(new)
            entry["path"] = str(new); entry["rejected"] = True
            print(f"[{i+1}/{len(grid)}] reject {slug} PQ={pq:.2f}")
        else:
            entry["path"] = str(target); entry["rejected"] = False
            print(f"[{i+1}/{len(grid)}] keep   {slug} PQ={pq:.2f} ({dt:.1f}s)")
        manifest.append(entry)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    kept = sum(1 for m in manifest if not m["rejected"])
    print(f"[stable_audio_pregrid] {kept}/{len(manifest)} kept "
          f"in {(time.perf_counter()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
