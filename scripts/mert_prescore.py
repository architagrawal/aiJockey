"""Pre-score every cached clip with the trained MERT reward head.

Writes sidecar `<clip_id>.mert_pred.json` next to existing metadata at
`/cache`. Format:

    {"PQ": 7.21, "PC": 5.40, "CE": 6.95, "CU": 7.10}

Picker reads this via library_picker_score.mert_lift_term.

Run on droplet:
    export AIJOCKEY_MERT_REWARD_ENABLE=1
    export AIJOCKEY_MERT_REWARD_CKPT=/scratch/mert_reward.pt
    /opt/venv/bin/python scripts/mert_prescore.py --cache /cache
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
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from mert_reward import enabled, predict
    if not enabled():
        sys.exit("AIJOCKEY_MERT_REWARD_ENABLE != 1; aborting")
    cache = Path(args.cache)
    jsons = sorted(p for p in cache.glob("*.json")
                    if not p.name.endswith(".audiobox_slices.json")
                    and not p.name.endswith(".mert_pred.json"))
    if args.limit:
        jsons = jsons[: args.limit]
    print(f"[mert_prescore] {len(jsons)} clips, cache={cache}")
    summary = {"ok": 0, "skip_exists": 0, "skip_no_audio": 0, "fail": 0}
    t_all = time.perf_counter()
    for i, jp in enumerate(jsons):
        cid = jp.stem
        side = cache / f"{cid}.mert_pred.json"
        if side.exists() and not args.force:
            summary["skip_exists"] += 1
            continue
        try:
            meta = json.loads(jp.read_text())
        except Exception:
            summary["fail"] += 1
            continue
        ap_p = meta.get("path")
        if not ap_p or not Path(ap_p).exists():
            summary["skip_no_audio"] += 1
            continue
        t0 = time.perf_counter()
        pred = predict(ap_p)
        dt = time.perf_counter() - t0
        if not pred:
            summary["fail"] += 1
            print(f"[{i+1}/{len(jsons)}] FAIL {cid}")
            continue
        side.write_text(json.dumps(pred, indent=2))
        summary["ok"] += 1
        print(f"[{i+1}/{len(jsons)}] ok {cid} ({dt:.1f}s) "
              f"PQ={pred['PQ']:.2f} CE={pred['CE']:.2f}")
    print(f"[mert_prescore] done in {(time.perf_counter()-t_all)/60:.1f} min: {summary}")


if __name__ == "__main__":
    main()
