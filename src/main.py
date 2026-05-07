"""
AiJockey CLI orchestrator.

Subcommands:
    analyze   — stems + beats + key + structure + hooks + CLAP per clip
    plan      — beam-search timeline from analyzed clip pool
    execute   — render timeline to raw_mix.wav
    master    — apply mastering chain
    eval      — quantitative metrics on mix
    all       — analyze + plan + execute + master in one go

Usage:
    python src/main.py all --clips clips/ --duration 600 --out_dir output/
    python src/main.py analyze --clips clips/
    python src/main.py plan --duration 900 --surprises 2
    python src/main.py execute
    python src/main.py master --lufs -9
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse


def cmd_analyze(args: argparse.Namespace) -> None:
    from analyze import analyze_pool
    analyze_pool(args.clips, args.cache, args.device, args.force)


def cmd_plan(args: argparse.Namespace) -> None:
    from planner import load_clips, plan, save_timeline, PlannerConfig
    clips = load_clips(args.cache)
    if not clips:
        print(f"no analyzed clips in {args.cache}. Run 'analyze' first.")
        sys.exit(1)
    cfg = PlannerConfig(
        target_duration=args.duration,
        surprise_budget=args.surprises,
        callback_budget=args.callbacks,
        max_clips=args.max_clips,
        style_rag_dir=args.style_rag,
    )
    tl = plan(clips, cfg)
    save_timeline(tl, args.out)
    print(f"wrote {args.out} ({len(tl)} entries)")


def cmd_execute(args: argparse.Namespace) -> None:
    from execute import execute
    execute(args.timeline, args.cache, args.out, args.samples)


def cmd_master(args: argparse.Namespace) -> None:
    from master import master
    master(args.in_path, args.out, args.lufs)


def cmd_eval(args: argparse.Namespace) -> None:
    from eval import evaluate
    import json as _json
    print(_json.dumps(
        evaluate(args.mix, args.timeline,
                 reference_cache_dir=args.reference_cache),
        indent=2))


def cmd_all(args: argparse.Namespace) -> None:
    from analyze import analyze_pool
    from planner import load_clips, plan, save_timeline, PlannerConfig
    from execute import execute
    from master import master
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("[1/4] analyze")
    analyze_pool(args.clips, args.cache, args.device, args.force)
    print("[2/4] plan")
    clips = load_clips(args.cache)
    if not clips:
        print("no analyzed clips after analyze. Aborting.")
        sys.exit(1)
    cfg = PlannerConfig(
        target_duration=args.duration,
        surprise_budget=args.surprises,
        callback_budget=args.callbacks,
        max_clips=args.max_clips,
        style_rag_dir=args.style_rag,
    )
    tl = plan(clips, cfg)
    timeline_path = str(out_dir / 'timeline.json')
    save_timeline(tl, timeline_path)
    print(f"  -> {timeline_path}")
    print("[3/4] execute")
    raw_path = str(out_dir / 'raw_mix.wav')
    execute(timeline_path, args.cache, raw_path, args.samples)
    print("[4/4] master")
    final_path = str(out_dir / 'final_mix.wav')
    master(raw_path, final_path, args.lufs)
    print(f"\nDONE: {final_path}")


def main() -> None:
    ap = argparse.ArgumentParser(prog='aijockey', description='AI DJ set generator')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('analyze')
    p.add_argument('--clips', required=True)
    p.add_argument('--cache', default='cache')
    p.add_argument('--device', default='cuda')
    p.add_argument('--force', action='store_true')
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser('plan')
    p.add_argument('--cache', default='cache')
    p.add_argument('--out', default='output/timeline.json')
    p.add_argument('--duration', type=float, default=1800.0)
    p.add_argument('--surprises', type=int, default=10)
    p.add_argument('--callbacks', type=int, default=1)
    p.add_argument('--max_clips', type=int, default=200)
    p.add_argument('--style_rag', default=None,
                   help='reference dir for Style-RAG bias (optional)')
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser('execute')
    p.add_argument('--timeline', default='output/timeline.json')
    p.add_argument('--cache', default='cache')
    p.add_argument('--out', default='output/raw_mix.wav')
    p.add_argument('--samples', default='samples')
    p.set_defaults(func=cmd_execute)

    p = sub.add_parser('master')
    p.add_argument('--in_path', default='output/raw_mix.wav')
    p.add_argument('--out', default='output/final_mix.wav')
    p.add_argument('--lufs', type=float, default=-9.0)
    p.set_defaults(func=cmd_master)

    p = sub.add_parser('eval')
    p.add_argument('--mix', required=True)
    p.add_argument('--timeline', required=True)
    p.add_argument('--reference_cache', default=None,
                   help='cache dir of reference clips (for FAD)')
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser('all')
    p.add_argument('--clips', required=True)
    p.add_argument('--cache', default='cache')
    p.add_argument('--out_dir', default='output')
    p.add_argument('--samples', default='samples')
    p.add_argument('--device', default='cuda')
    p.add_argument('--force', action='store_true')
    p.add_argument('--duration', type=float, default=1800.0)
    p.add_argument('--surprises', type=int, default=10)
    p.add_argument('--callbacks', type=int, default=1)
    p.add_argument('--max_clips', type=int, default=200)
    p.add_argument('--lufs', type=float, default=-9.0)
    p.add_argument('--style_rag', default=None,
                   help='reference dir for Style-RAG bias (optional)')
    p.set_defaults(func=cmd_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
