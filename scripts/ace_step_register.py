"""Register ACE-Step-generated bridges into the library cache.

Reads the manifest emitted by scripts/ace_step_pregrid.py, copies each
kept (non-rejected) WAV into the library cache dir with a `bridge__`
filename prefix, then runs the standard pool analyzer over the new
files so they become first-class clips the planner can pick.

Picker will see bridges as normal clips with `source='library'` and
genre prefix `bridge`. Use Director prompt routing to bias toward
bridges only at junctions where library coverage is sparse.

Usage:
    /opt/venv/bin/python scripts/ace_step_register.py \\
        --pregrid /cache/ace_step_bridges \\
        --cache /cache --device cuda
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pregrid", required=True,
                    help="output dir from ace_step_pregrid.py")
    ap.add_argument("--cache", default="/cache")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    pregrid = Path(args.pregrid)
    manifest_path = pregrid / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"manifest.json missing in {pregrid}")
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    staging = pregrid / "_staged_for_cache"
    staging.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    kept = [m for m in manifest if not m.get("rejected")]
    print(f"[register] {len(kept)} kept bridges from {pregrid}")

    copied = []
    for m in kept:
        src = Path(m["path"])
        if not src.exists():
            print(f"[skip-missing] {src}")
            continue
        # Filename: bridge__<bpm>_<key>_<mood>.wav so planner reads
        # genre-prefix 'bridge' for any provenance-aware routing.
        slug = f"bridge__{m['bpm']}bpm_{m['key_camelot']}_{m['mood']}"
        dst = staging / f"{slug}.wav"
        if not dst.exists():
            shutil.copy2(src, dst)
        copied.append({"src": str(src), "staged": str(dst), "slug": slug,
                        "meta": m})
    print(f"[register] staged {len(copied)} files at {staging}")

    if not copied:
        print("[register] nothing to analyze")
        return

    # Run pool analyzer over the staging dir; outputs land in cache/.
    from analyze import analyze_pool
    print(f"[register] analyzing {len(copied)} bridges → {cache}")
    analyze_pool(str(staging), str(cache), device=args.device,
                  force=False, workers=args.workers)

    # Persist registration log.
    (pregrid / "registration.json").write_text(
        json.dumps({"cache": str(cache), "copied": copied}, indent=2))
    print(f"[register] done. Bridges now visible via load_clips({cache})")


if __name__ == "__main__":
    main()
