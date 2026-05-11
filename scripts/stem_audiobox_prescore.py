"""Per-stem Audiobox prescore — separate PQ/CE for drums/bass/other.

Runs Audiobox Aesthetics on each cached clip's individual stem
(loaded from cache/stems/<clip_id>/<stem>.wav). Writes sidecar
<clip_id>.stem_audiobox.json with per-stem 4-axis scores. Picker can
weight per-stem quality at plan time.

Run:
    export AIJOCKEY_AUDIOBOX_AESTHETICS=1
    /opt/venv/bin/python scripts/stem_audiobox_prescore.py --cache /cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/cache")
    ap.add_argument("--stems", nargs="+",
                    default=["drums", "bass", "other", "vocals"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from audiobox_critic import enabled, score, score_batch
    if not enabled():
        sys.exit("AIJOCKEY_AUDIOBOX_AESTHETICS != 1; aborting")

    cache = Path(args.cache)
    stems_root = cache / "stems"
    if not stems_root.exists():
        sys.exit(f"no stems dir at {stems_root}")
    jsons = sorted(p for p in cache.glob("*.json")
                    if not p.name.endswith(".audiobox_slices.json")
                    and not p.name.endswith(".mert_pred.json")
                    and not p.name.endswith(".stem_audiobox.json"))
    print(f"[stem_audiobox] {len(jsons)} clips")
    t_all = time.perf_counter()
    summary = {"ok": 0, "skip": 0, "fail": 0}
    for i, jp in enumerate(jsons):
        cid = jp.stem
        out_p = cache / f"{cid}.stem_audiobox.json"
        if out_p.exists() and not args.force:
            summary["skip"] += 1; continue
        stem_dir = stems_root / cid
        paths = []
        names = []
        for s in args.stems:
            p = stem_dir / f"{s}.wav"
            if p.exists():
                paths.append(str(p))
                names.append(s)
        if not paths:
            summary["fail"] += 1; continue
        try:
            scores = score_batch(paths)
        except Exception as e:
            print(f"[{i+1}/{len(jsons)}] FAIL {cid}: {e}")
            summary["fail"] += 1; continue
        if not scores or all(s is None for s in scores):
            summary["fail"] += 1; continue
        blob = {n: s for n, s in zip(names, scores) if s}
        out_p.write_text(json.dumps(blob, indent=2))
        summary["ok"] += 1
        if i % 10 == 0:
            print(f"[{i+1}/{len(jsons)}] {cid} stems={list(blob.keys())}")
    print(f"[stem_audiobox] done in {(time.perf_counter()-t_all)/60:.1f} min: "
          f"{summary}")


if __name__ == "__main__":
    main()
