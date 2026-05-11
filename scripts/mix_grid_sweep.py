"""Sweep mix-render combinatorics: (clip pool) x (mode) x (style) x (arc).

For each cell, runs `python -m src.main all` once with the matching env
+ CLI flags. Audiobox-scores the output. Writes a results JSONL.

Use to curate a new featured gallery without manual cherry-picking.

Usage (MI300X):
    export AIJOCKEY_USE_DIRECTOR_LLM=1 AIJOCKEY_AUDIOBOX_AESTHETICS=1
    export AIJOCKEY_DIRECTOR_N_SAMPLES=3
    /opt/venv/bin/python scripts/mix_grid_sweep.py \\
        --pools /workspace/user_set:userset,/workspace/user_genres/chillstep:chill \\
        --modes dj_set,mashup \\
        --styles festival_inferno,midnight_noir,east_meets_bass \\
        --arcs tomorrowland,build \\
        --duration 180 --cache /cache --out /workspace/output/grid_sweep
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _audiobox_score(path: Path) -> dict | None:
    try:
        from audiobox_critic import enabled, score
        if not enabled():
            return None
        return score(str(path))
    except Exception:
        return None


def _run_one(pool: str, mode: str, style: str, arc: str,
              args, out_dir: Path) -> dict:
    cmd = [
        sys.executable, "-m", "main", "all",
        "--clips", pool,
        "--cache", args.cache,
        "--out_dir", str(out_dir),
        "--duration", str(args.duration),
        "--arc", arc,
        "--lufs", str(args.lufs),
        "--use_director", "--apply_llm_tiers",
        "--workers", str(args.workers),
    ]
    env = dict(os.environ)
    env["AIJOCKEY_MODE"] = mode
    env["AIJOCKEY_STYLE"] = style
    env["PYTHONPATH"] = str(ROOT / "src")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=env, cwd=str(ROOT),
                               capture_output=True, text=True,
                               timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed_s": args.timeout}
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        (out_dir / "stderr.log").write_text(
            (proc.stdout or "") + "\n---STDERR---\n" + (proc.stderr or ""))
        return {"status": "fail", "rc": proc.returncode, "elapsed_s": dt}
    mp3 = sorted(out_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
    if not mp3:
        mp3 = sorted(out_dir.glob("*.wav"))
    if not mp3:
        return {"status": "no_output", "elapsed_s": dt}
    target = mp3[-1]
    scores = _audiobox_score(target)
    return {"status": "ok", "path": str(target), "audiobox": scores,
            "elapsed_s": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True,
                    help="comma-separated 'path:name' pairs")
    ap.add_argument("--modes", default="dj_set",
                    help="comma-separated: dj_set,mashup")
    ap.add_argument("--styles", default="festival_inferno")
    ap.add_argument("--arcs", default="tomorrowland,build")
    ap.add_argument("--duration", type=int, default=180)
    ap.add_argument("--cache", default="/cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lufs", type=float, default=-9.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pools = []
    for tok in args.pools.split(","):
        path, _, name = tok.partition(":")
        name = name or Path(path).name
        pools.append((path.strip(), name.strip()))
    modes = [s.strip() for s in args.modes.split(",") if s.strip()]
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    arcs = [s.strip() for s in args.arcs.split(",") if s.strip()]

    combos = list(itertools.product(pools, modes, styles, arcs))
    print(f"[grid_sweep] {len(combos)} renders")

    results_path = out / "results.jsonl"
    log = []
    for i, ((pool_path, pool_name), mode, style, arc) in enumerate(combos):
        slug = f"{pool_name}_{mode}_{style}_{arc}"
        cell_dir = out / slug
        cell_dir.mkdir(parents=True, exist_ok=True)
        if list(cell_dir.glob("*.mp3")):
            print(f"[{i+1}/{len(combos)}] skip {slug} (cached)")
            continue
        print(f"[{i+1}/{len(combos)}] run {slug}")
        r = _run_one(pool_path, mode, style, arc, args, cell_dir)
        r.update({"slug": slug, "pool": pool_name, "mode": mode,
                  "style": style, "arc": arc})
        log.append(r)
        with open(results_path, "a") as f:
            f.write(json.dumps(r) + "\n")
        if r.get("audiobox"):
            print(f"           PQ={r['audiobox']['PQ']:.2f} "
                  f"CE={r['audiobox']['CE']:.2f}")

    # Auto-curate top-10 by (PQ+CE)/2
    ranked = []
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "ok" or not r.get("audiobox"):
                continue
            aae = r["audiobox"]
            comp = (float(aae.get("PQ", 0)) + float(aae.get("CE", 0))) / 2
            ranked.append((comp, r))
    ranked.sort(key=lambda x: -x[0])
    top = out / "_top"
    top.mkdir(parents=True, exist_ok=True)
    for rank, (comp, r) in enumerate(ranked[:10], start=1):
        src = Path(r["path"])
        if not src.exists():
            continue
        dst = top / f"{rank:02d}_PQ{r['audiobox']['PQ']:.2f}_{r['slug']}.mp3"
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass
    print(f"\n[grid_sweep] done. top-10 in {top}")


if __name__ == "__main__":
    main()
