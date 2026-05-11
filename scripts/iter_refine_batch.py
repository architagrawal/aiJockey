"""Iterative-refinement batch: render same spec N times, keep best by Audiobox PQ+CE.

Variance comes from `AIJOCKEY_DIRECTOR_N_SAMPLES > 1` + `AIJOCKEY_DIRECTOR_TEMPERATURE`,
so each subprocess draws a fresh Director plan.

Usage (on MI300X, inside rocm container):
    export AIJOCKEY_AUDIOBOX_AESTHETICS=1 AIJOCKEY_USE_DIRECTOR_LLM=1
    export AIJOCKEY_DIRECTOR_N_SAMPLES=3 AIJOCKEY_DIRECTOR_TEMPERATURE=0.7
    /opt/venv/bin/python scripts/iter_refine_batch.py \\
        --clips /workspace/user_set --cache /cache \\
        --duration 180 --arc tomorrowland \\
        --prompt "Tomorrowland mainstage" \\
        --n 5 --out /workspace/output/iter_refined

Picks output with highest (PQ + CE)/2.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _audiobox_score(mp3_path: Path) -> dict | None:
    """Score one finished mix. Returns dict with PQ/PC/CE/CU or None."""
    try:
        from audiobox_critic import enabled, score
    except Exception as e:
        print(f"[iter_refine] audiobox import failed: {e}")
        return None
    if not enabled():
        print("[iter_refine] AIJOCKEY_AUDIOBOX_AESTHETICS not set, no scoring")
        return None
    return score(str(mp3_path))


def _run_one(args, run_idx: int, out_dir: Path) -> dict | None:
    """Invoke main.py all → return {idx, path, scores} or None on failure."""
    run_dir = out_dir / f"run_{run_idx:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "main", "all",
        "--clips", args.clips,
        "--cache", args.cache,
        "--out_dir", str(run_dir),
        "--duration", str(args.duration),
        "--arc", args.arc,
        "--lufs", str(args.lufs),
        "--workers", str(args.workers),
        "--use_director",
        "--apply_llm_tiers",
    ]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    if args.n_best > 1:
        cmd += ["--n_best", str(args.n_best)]
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT / "src"))
    print(f"\n[iter_refine] run {run_idx}/{args.n} cmd={' '.join(cmd[:7])} ...")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=env, cwd=str(ROOT),
                               capture_output=True, text=True,
                               timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f"[iter_refine] run {run_idx} TIMEOUT after {args.timeout}s")
        return None
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        log = run_dir / "stderr.log"
        log.write_text((proc.stdout or "") + "\n---STDERR---\n" + (proc.stderr or ""))
        print(f"[iter_refine] run {run_idx} FAILED rc={proc.returncode} log={log}")
        return None
    mp3 = sorted(run_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
    if not mp3:
        wavs = sorted(run_dir.glob("*.wav"))
        if not wavs:
            print(f"[iter_refine] run {run_idx} produced no audio")
            return None
        out_path = wavs[-1]
    else:
        out_path = mp3[-1]
    scores = _audiobox_score(out_path)
    if not scores:
        return {"idx": run_idx, "path": str(out_path), "scores": None,
                "elapsed_s": dt}
    return {"idx": run_idx, "path": str(out_path), "scores": scores,
            "elapsed_s": dt}


def _composite(scores: dict | None) -> float:
    if not scores:
        return float("-inf")
    return (float(scores.get("PQ", 0)) + float(scores.get("CE", 0))) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--cache", default="/cache")
    ap.add_argument("--out", required=True, help="final-best output dir")
    ap.add_argument("--n", type=int, default=5, help="renders per spec")
    ap.add_argument("--duration", type=int, default=180)
    ap.add_argument("--arc", default="tomorrowland")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--lufs", type=float, default=-9.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n_best", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-render timeout seconds")
    ap.add_argument("--keep-all", action="store_true",
                    help="keep all run_NN dirs (default: keep only best)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for i in range(1, args.n + 1):
        r = _run_one(args, i, runs_dir)
        if r:
            results.append(r)
            print(f"[iter_refine] run {i}: scores={r['scores']} "
                  f"composite={_composite(r['scores']):.3f}")

    if not results:
        sys.exit("[iter_refine] no successful runs")

    results.sort(key=lambda r: _composite(r["scores"]), reverse=True)
    best = results[0]
    final_name = (Path(best["path"]).stem
                  + f"_PQ{(best['scores'] or {}).get('PQ', 0):.2f}"
                  + Path(best["path"]).suffix)
    final_path = out / final_name
    shutil.copy2(best["path"], final_path)
    summary = {
        "spec": vars(args),
        "results": [
            {"idx": r["idx"], "scores": r["scores"], "elapsed_s": r["elapsed_s"],
             "path": r["path"]}
            for r in results
        ],
        "best": {"idx": best["idx"], "scores": best["scores"],
                  "final_path": str(final_path),
                  "composite": _composite(best["scores"])},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    if not args.keep_all:
        for r in results:
            if r["idx"] == best["idx"]:
                continue
            run_dir = Path(r["path"]).parent
            try:
                shutil.rmtree(run_dir)
            except Exception:
                pass

    print(f"\n[iter_refine] best run_{best['idx']:02d} "
          f"composite={_composite(best['scores']):.3f} → {final_path}")


if __name__ == "__main__":
    main()
