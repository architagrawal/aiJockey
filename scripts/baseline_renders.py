"""Baseline distribution runner.

Renders N prompts × M seeds against a fixed cache, logs each via the
probe pipeline. Output: a baseline severity distribution we can compare
improver runs against.

Usage:
    python scripts/baseline_renders.py \
        --cache /workspace/test_user_cache_v6 \
        --out_dir /workspace/output/baseline \
        --duration 180 \
        --prompts_file scripts/prompts/baseline.json \
        --seeds 5

Each render appends one row to $AIJOCKEY_PROBE_LOG.
After completion, prints summarize() over only the rows produced this run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PROMPTS = [
    "warm-up to peak hour",
    "after-hours smoky lo-fi",
    "festival peak euphoric drops",
    "long groove, hypnotic minimal",
    "wild journey, peaks and valleys",
    "deep house cooldown",
    "rolling tech-house build",
    "melodic techno set",
    "drum and bass energy ramp",
    "ambient warmup",
]


def _load_prompts(p: str | None) -> list[str]:
    if not p:
        return DEFAULT_PROMPTS
    try:
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict) and 'prompts' in data:
            return [str(x) for x in data['prompts']]
    except Exception as e:
        print(f"warn: prompts file unreadable ({e}); using defaults")
    return DEFAULT_PROMPTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out_dir', default='output/baseline')
    ap.add_argument('--duration', type=float, default=180.0)
    ap.add_argument('--arc', default='build')
    ap.add_argument('--prompts_file', default=None)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--max_clips', type=int, default=8)
    ap.add_argument('--min_unique_clips', type=int, default=2)
    ap.add_argument('--use_director', action='store_true')
    ap.add_argument('--apply_llm_tiers', action='store_true')
    ap.add_argument('--src_dir', default='src',
                    help='where main.py lives')
    ap.add_argument('--improve_max_passes', type=int, default=0,
                    help='Forward to main.py execute (0=baseline, 1+=improver)')
    ap.add_argument('--improve_threshold', type=float, default=0.5)
    args = ap.parse_args()

    prompts = _load_prompts(args.prompts_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(args.src_dir)

    started = time.time()
    n_renders = len(prompts) * args.seeds
    print(f"[baseline] {len(prompts)} prompts × {args.seeds} seeds = "
          f"{n_renders} renders, duration={args.duration}s")

    n_done = 0
    n_failed = 0
    for pi, prompt in enumerate(prompts):
        for seed in range(args.seeds):
            tag = f"p{pi}_s{seed}"
            tl = out_dir / f'tl_{tag}.json'
            wav = out_dir / f'mix_{tag}.wav'
            print(f"\n[baseline] {n_done + n_failed + 1}/{n_renders}  "
                  f"prompt='{prompt}' seed={seed}")
            # Plan
            plan_cmd = [
                sys.executable, str(src_dir / 'main.py'), 'plan',
                '--cache', args.cache,
                '--out', str(tl),
                '--duration', str(args.duration),
                '--arc', args.arc,
                '--prompt', prompt,
                '--min_unique_clips', str(args.min_unique_clips),
                '--max_clips', str(args.max_clips),
                # No --seed in planner; vary surprises (1..N) per seed to
                # produce different N-best ranker picks. Crude but enough
                # variation for baseline distribution.
                '--surprises', str(1 + seed * 2),
            ]
            if args.use_director:
                plan_cmd.append('--use_director')
            if args.apply_llm_tiers:
                plan_cmd.append('--apply_llm_tiers')
            r = subprocess.run(plan_cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  PLAN FAILED: {r.stderr[-300:]}")
                n_failed += 1
                continue
            # Execute (logs probe automatically via cmd_execute)
            exec_cmd = [
                sys.executable, str(src_dir / 'main.py'), 'execute',
                '--timeline', str(tl),
                '--cache', args.cache,
                '--out', str(wav),
                '--improve_max_passes', str(args.improve_max_passes),
                '--improve_threshold', str(args.improve_threshold),
            ]
            r = subprocess.run(exec_cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  EXEC FAILED: {r.stderr[-300:]}")
                n_failed += 1
                continue
            # Look for probe line in stdout
            for line in r.stdout.splitlines()[-20:]:
                if line.startswith('[probe]'):
                    print(f"  {line}")
                    break
            n_done += 1

    elapsed = time.time() - started
    print(f"\n[baseline] done: {n_done} ok, {n_failed} failed, "
          f"{elapsed:.0f}s elapsed ({elapsed / max(1, n_done):.1f}s/render)")

    # Summarize all runs
    sys.path.insert(0, str(src_dir))
    try:
        from probe_log import read_log, summarize
        rows = read_log()
        # Filter to only the rows from THIS baseline run (job_id starts with cli_<ts>
        # where ts >= started). Lenient: include all cli_ rows from now on.
        recent = [r for r in rows if r.get('job_id', '').startswith('cli_')
                  and r.get('ts', '') >= time.strftime(
                      '%Y-%m-%dT%H:%M:%S', time.gmtime(started - 60))]
        print(f"\n[baseline] summary over {len(recent)} recent renders:")
        print(json.dumps(summarize(recent), indent=2))
    except Exception as e:
        print(f"warn: summary skipped ({e})")


if __name__ == '__main__':
    main()
