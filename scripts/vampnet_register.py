"""Register VampNet-generated bridges into /cache as first-class clips.

Reads /cache/vampnet_bridges/manifest.json (output of vampnet_pregrid.py),
copies each kept bridge to a staging dir with `bridge__` prefix, then
runs analyze_pool over the staging dir so each bridge gets full
json + npz metadata in /cache.

Run:
    /opt/venv/bin/python scripts/vampnet_register.py \\
        --pregrid /cache/vampnet_bridges \\
        --cache /cache --device cuda --workers 2
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
    ap.add_argument("--pregrid", required=True)
    ap.add_argument("--cache", default="/cache")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    pre = Path(args.pregrid)
    manifest_p = pre / "manifest.json"
    if not manifest_p.exists():
        sys.exit(f"manifest.json missing in {pre}")
    manifest = json.loads(manifest_p.read_text())
    kept = [m for m in manifest if not m.get("rejected")]
    print(f"[vampnet_register] {len(kept)} kept bridges")

    staging = pre / "_staged_for_cache"
    staging.mkdir(parents=True, exist_ok=True)

    copied = []
    for m in kept:
        src = Path(m.get("path", ""))
        if not src.exists():
            continue
        a = m.get("a_id", "a")[:24]
        b = m.get("b_id", "b")[:24]
        # Prefix with bridge__ so planner / Director can route on genre
        # prefix (string contains 'bridge'). Strip slug noise.
        slug = m.get("slug") or f"bridge_{a}_to_{b}"
        if not slug.startswith("bridge"):
            slug = "bridge__" + slug
        else:
            slug = slug.replace("vamp_bridge_", "bridge__", 1)
        dst = staging / f"{slug}.wav"
        if not dst.exists():
            shutil.copy2(src, dst)
        copied.append({"src": str(src), "staged": str(dst), "slug": slug,
                        "meta": m})
    print(f"[vampnet_register] staged {len(copied)} files at {staging}")

    if not copied:
        return

    from analyze import analyze_pool
    print(f"[vampnet_register] analyzing {len(copied)} bridges -> {args.cache}")
    analyze_pool(str(staging), str(args.cache), device=args.device,
                  force=False, workers=args.workers)
    (pre / "registration.json").write_text(
        json.dumps({"cache": args.cache, "copied": copied}, indent=2))
    print(f"[vampnet_register] done")


if __name__ == "__main__":
    main()
