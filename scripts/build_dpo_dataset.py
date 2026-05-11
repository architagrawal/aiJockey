"""Merge iter_refine + grid_sweep + probe_log into unified DPO dataset.

Output: JSONL with rows
    {
      "prompt": str,
      "plan": <director.json content>,
      "audiobox": {"PQ":..., "PC":..., "CE":..., "CU":...},
      "path": str,
      "composite": (PQ+CE)/2,
      "pool_fingerprint": str
    }

Sources (any combination):
    1. /workspace/output/iter_refined*/_runs/run_NN/director.json
       + sibling raw_mix.wav/mp3 → score live via audiobox_critic.
    2. /workspace/output/grid_sweep/results.jsonl (has audiobox + path,
       missing plan → skipped unless paired with a sibling director.json).

Run:
    export AIJOCKEY_AUDIOBOX_AESTHETICS=1
    /opt/venv/bin/python scripts/build_dpo_dataset.py \\
        --out /scratch/dpo_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _audiobox(path: Path):
    try:
        from audiobox_critic import enabled, score
        if not enabled():
            return None
        return score(str(path))
    except Exception as e:
        print(f"[dpo_merge] audiobox fail {path}: {e}")
        return None


def _pool_fp(clips_dir: str | None) -> str:
    if not clips_dir:
        return ""
    p = Path(clips_dir)
    return p.name


def _find_audio(run_dir: Path) -> Path | None:
    for ext in ("*.mp3", "*.wav"):
        cands = sorted(run_dir.glob(ext), key=lambda x: x.stat().st_mtime)
        if cands:
            return cands[-1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/dpo_dataset.jsonl")
    ap.add_argument("--root", default="/workspace/output")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    dirs = sorted(root.glob("iter_refined*"))
    print(f"[dpo_merge] {len(dirs)} iter_refined dirs")
    for d in dirs:
        runs_root = d / "_runs"
        if not runs_root.exists():
            continue
        for run_dir in sorted(runs_root.glob("run_*")):
            dj = run_dir / "director.json"
            tl = run_dir / "timeline.json"
            plan: dict | None = None
            if dj.exists():
                try:
                    plan = json.loads(dj.read_text())
                except Exception:
                    plan = None
            if plan is None and tl.exists():
                # Fallback: reconstruct minimal plan from executed timeline.
                try:
                    blob = json.loads(tl.read_text())
                    tl_entries = blob.get("timeline") or []
                    plan = {
                        "text_prompt": (blob.get("meta") or {}).get("prompt"),
                        "arc": (blob.get("meta") or {}).get("arc"),
                        "transition_tiers": [
                            (e.get("transition_in") or {}).get("tier", "minor")
                            for e in tl_entries[1:]
                        ],
                        "clip_sequence": [e.get("clip_id") for e in tl_entries],
                        "_from_timeline_fallback": True,
                    }
                except Exception:
                    plan = None
            if plan is None:
                continue
            audio = _find_audio(run_dir)
            if not audio:
                continue
            aae = _audiobox(audio)
            if not aae:
                continue
            pq = float(aae.get("PQ", 0.0))
            ce = float(aae.get("CE", 0.0))
            rows.append({
                "prompt": (plan.get("text_prompt") or ""),
                "plan": plan,
                "audiobox": aae,
                "path": str(audio),
                "composite": (pq + ce) / 2.0,
                "pool_fingerprint": d.name,
            })
            print(f"[dpo_merge] +{d.name}/{run_dir.name} PQ={pq:.2f}")
            if args.limit and len(rows) >= args.limit:
                break

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[dpo_merge] wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
