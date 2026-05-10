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
    analyze_pool(args.clips, args.cache, args.device, args.force,
                 workers=getattr(args, 'workers', 1))


def cmd_plan(args: argparse.Namespace) -> None:
    from planner import (
        load_clips, plan, plan_n_best, save_timeline, PlannerConfig,
        compute_pool_coherence, apply_llm_transition_tiers_to_timeline,
        attach_accent_hints,
    )
    clips = load_clips(args.cache)
    if not clips:
        print(f"no analyzed clips in {args.cache}. Run 'analyze' first.")
        sys.exit(1)

    director_out: dict | None = None
    if getattr(args, 'use_director', False):
        from director import run_director
        # Collect audio paths for multimodal Director (used when
        # HF_DIRECTOR_MODEL is an audio-LLM, e.g. Qwen2-Audio).
        audio_paths: list[str] = []
        clips_dir = getattr(args, 'clips', None)
        if clips_dir:
            from pathlib import Path as _P
            cd = _P(clips_dir)
            if cd.exists():
                for ext in ('*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg'):
                    for p in sorted(cd.glob(ext)):
                        audio_paths.append(str(p))
        director_out = run_director(
            user_prompt=getattr(args, 'prompt', None) or '',
            arc_preset=getattr(args, 'arc', 'build'),
            clip_count_estimate=len(clips),
            approx_duration_seconds=float(args.duration),
            audio_clip_paths=audio_paths or None,
            clips_meta=clips,
        )
        print(f"[director] arc={director_out.get('arc')} "
              f"prompt='{(director_out.get('text_prompt') or '')[:60]}' "
              f"tiers={len(director_out.get('transition_tiers') or [])}")
        # Persist Director JSON + pool diagnostic + narrative card next
        # to output mix.
        try:
            out_path = getattr(args, 'out', None) or getattr(args, 'output', None)
            if out_path:
                from pathlib import Path as _P
                import json as _json
                out_parent = _P(out_path).parent
                out_parent.mkdir(parents=True, exist_ok=True)
                with open(out_parent / 'director.json', 'w') as _f:
                    _json.dump(director_out, _f, indent=2)
                try:
                    from pool_intelligence import diagnose
                    diag = diagnose(clips)
                    card = {
                        'set_narrative':   director_out.get('set_narrative', ''),
                        'narrative_notes': director_out.get('narrative_notes', ''),
                        'arc':             director_out.get('arc'),
                        'pool_diagnostic': diag,
                        'transition_plan': list(zip(
                            director_out.get('transition_tiers', []),
                            director_out.get('transition_intents', []),
                        )),
                    }
                    with open(out_parent / 'card.json', 'w') as _f:
                        _json.dump(card, _f, indent=2)
                    print(f"[director] narrative: {card['set_narrative']}")
                    if card['narrative_notes']:
                        print(f"[director] notes: {card['narrative_notes']}")
                    print(f"[diagnostic] verdict={diag['verdict']} "
                          f"coherence={diag['coherence']} "
                          f"genres={diag['n_genres']} "
                          f"bpm_spread={diag['bpm_spread_pct']}%")
                except Exception:
                    pass
        except Exception as _e:
            print(f"warn: could not save director.json/card ({_e})")

    arc_final = (director_out or {}).get('arc') or getattr(args, 'arc', 'build')
    prompt_final = (director_out or {}).get('text_prompt') or getattr(args, 'prompt', None)
    surprise_final = int((director_out or {}).get('surprise_budget', args.surprises))
    callback_final = int((director_out or {}).get('callback_budget', args.callbacks))
    same_genre = bool((director_out or {}).get('same_genre_tight_mix'))

    cfg = PlannerConfig(
        target_duration=args.duration,
        surprise_budget=surprise_final,
        callback_budget=callback_final,
        max_clips=args.max_clips,
        min_unique_clips=getattr(args, 'min_unique_clips', 5),
        arc_shape=arc_final,
        style_rag_dir=args.style_rag,
        classifier_ckpt=args.classifier,
        compat_head_ckpt=args.compat_head,
        text_prompt=prompt_final,
        pool_coherence=compute_pool_coherence(clips),
        same_genre_tight_mix=same_genre,
    )
    n_best = getattr(args, 'n_best', 1)
    if n_best > 1:
        tl, meta = plan_n_best(clips, cfg, cache_dir=args.cache, n_candidates=n_best)
        print(f"N-best rerank picked: score={meta.get('best_score', 0):.3f} "
              f"breakdown={meta.get('best_breakdown', {})}")
    else:
        tl = plan(clips, cfg)
    if director_out is not None and getattr(args, 'apply_llm_tiers', False):
        tiers = director_out.get('transition_tiers') or []
        accents = director_out.get('accent_hints') or []
        if tiers:
            apply_llm_transition_tiers_to_timeline(tl, tiers, director_out.get("transition_intents"))
            print(f"[director] applied {len(tiers)} LLM tier transitions")
        if accents:
            attach_accent_hints(tl, accents)
            print(f"[director] attached {len(accents)} accent hints")
    save_timeline(tl, args.out)
    print(f"wrote {args.out} ({len(tl)} entries)")


def cmd_execute(args: argparse.Namespace) -> None:
    from execute import execute
    execute(args.timeline, args.cache, args.out, args.samples)
    # Optional probe + improve pass: render → probe → diagnose → mutate
    # → re-render touched segments. Cap retries at args.improve_max_passes
    # (default 1) to avoid infinite loops.
    max_passes = int(getattr(args, 'improve_max_passes', 0))
    if max_passes <= 0:
        return
    threshold = float(getattr(args, 'improve_threshold', 0.5))
    import json as _json
    from pathlib import Path as _P
    try:
        from audio_probes import probe_mix
        from improver import improve_timeline, apply_edits
    except ImportError as e:
        print(f"[improver] modules unavailable ({e})")
        return
    for pass_i in range(max_passes):
        probe = probe_mix(args.out, args.timeline)
        print(f"[improver] pass {pass_i + 1}: severity={probe['overall_severity']:.2f} "
              f"verdict={probe['verdict']}")
        if probe['overall_severity'] < threshold:
            print(f"[improver] severity {probe['overall_severity']:.2f} < threshold "
                  f"{threshold:.2f}; done")
            break
        with open(args.timeline) as f:
            blob = _json.load(f)
        tl = blob['timeline'] if isinstance(blob, dict) else blob
        report = improve_timeline(tl, probe)
        print(f"[improver] {len(report.edits)} edit(s), "
              f"{report.issues_skipped} skipped")
        for ed in report.edits:
            print(f"  j{ed.junction_index} {ed.action}: {ed.rationale}")
        if not report.edits:
            print("[improver] no actionable edits; stopping")
            break
        touched = apply_edits(tl, report.edits)
        edited_path = args.timeline.replace('.json', f'.improved{pass_i + 1}.json')
        if isinstance(blob, dict):
            blob['timeline'] = tl
            blob.setdefault('meta', {})['improver_touched'] = touched
            blob['meta'][f'improver_pass_{pass_i + 1}'] = report.to_dict()
            _json.dump(blob, open(edited_path, 'w'), indent=2)
        else:
            _json.dump(tl, open(edited_path, 'w'), indent=2)
        # Cascade re-render: touched segments + 1 either side. For now just
        # re-execute the whole edited timeline (simpler; planner-stage
        # surgical re-render is P2).
        out_improved = args.out.replace('.wav', f'.improved{pass_i + 1}.wav')
        execute(edited_path, args.cache, out_improved, args.samples)
        args.timeline = edited_path
        args.out = out_improved
    print(f"[improver] final mix: {args.out}")


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
    import os as _os
    from analyze import analyze_pool
    from planner import (
        load_clips, plan, plan_n_best, save_timeline, PlannerConfig,
        compute_pool_coherence, apply_llm_transition_tiers_to_timeline,
        attach_accent_hints,
    )
    from execute import execute
    from master import master
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("[1/4] analyze")
    analyze_pool(args.clips, args.cache, args.device, args.force,
                 workers=getattr(args, 'workers', 1))
    print("[2/4] plan")
    clips = load_clips(args.cache)
    if not clips:
        print("no analyzed clips after analyze. Aborting.")
        sys.exit(1)

    # Director LLM (optional) — produces sanitized plan dict overriding prompt/arc/budgets
    director_out: dict | None = None
    if getattr(args, 'use_director', False):
        from director import run_director
        # Collect audio paths for multimodal Director (used when
        # HF_DIRECTOR_MODEL is an audio-LLM, e.g. Qwen2-Audio).
        audio_paths: list[str] = []
        clips_dir = getattr(args, 'clips', None)
        if clips_dir:
            from pathlib import Path as _P
            cd = _P(clips_dir)
            if cd.exists():
                for ext in ('*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg'):
                    for p in sorted(cd.glob(ext)):
                        audio_paths.append(str(p))
        director_out = run_director(
            user_prompt=getattr(args, 'prompt', None) or '',
            arc_preset=getattr(args, 'arc', 'build'),
            clip_count_estimate=len(clips),
            approx_duration_seconds=float(args.duration),
            audio_clip_paths=audio_paths or None,
            clips_meta=clips,
        )
        print(f"[director] arc={director_out.get('arc')} "
              f"prompt='{(director_out.get('text_prompt') or '')[:60]}' "
              f"tiers={len(director_out.get('transition_tiers') or [])}")
        # Persist Director JSON + pool diagnostic + narrative card next
        # to output mix.
        try:
            out_path = getattr(args, 'out', None) or getattr(args, 'output', None)
            if out_path:
                from pathlib import Path as _P
                import json as _json
                out_parent = _P(out_path).parent
                out_parent.mkdir(parents=True, exist_ok=True)
                with open(out_parent / 'director.json', 'w') as _f:
                    _json.dump(director_out, _f, indent=2)
                try:
                    from pool_intelligence import diagnose
                    diag = diagnose(clips)
                    card = {
                        'set_narrative':   director_out.get('set_narrative', ''),
                        'narrative_notes': director_out.get('narrative_notes', ''),
                        'arc':             director_out.get('arc'),
                        'pool_diagnostic': diag,
                        'transition_plan': list(zip(
                            director_out.get('transition_tiers', []),
                            director_out.get('transition_intents', []),
                        )),
                    }
                    with open(out_parent / 'card.json', 'w') as _f:
                        _json.dump(card, _f, indent=2)
                    print(f"[director] narrative: {card['set_narrative']}")
                    if card['narrative_notes']:
                        print(f"[director] notes: {card['narrative_notes']}")
                    print(f"[diagnostic] verdict={diag['verdict']} "
                          f"coherence={diag['coherence']} "
                          f"genres={diag['n_genres']} "
                          f"bpm_spread={diag['bpm_spread_pct']}%")
                except Exception:
                    pass
        except Exception as _e:
            print(f"warn: could not save director.json/card ({_e})")

    arc_final = (director_out or {}).get('arc') or getattr(args, 'arc', 'build')
    prompt_final = (director_out or {}).get('text_prompt') or getattr(args, 'prompt', None)
    surprise_final = int((director_out or {}).get('surprise_budget', args.surprises))
    callback_final = int((director_out or {}).get('callback_budget', args.callbacks))
    same_genre = bool((director_out or {}).get('same_genre_tight_mix'))

    cfg = PlannerConfig(
        target_duration=args.duration,
        surprise_budget=surprise_final,
        callback_budget=callback_final,
        max_clips=args.max_clips,
        min_unique_clips=getattr(args, 'min_unique_clips', 5),
        arc_shape=arc_final,
        style_rag_dir=args.style_rag,
        classifier_ckpt=args.classifier,
        compat_head_ckpt=args.compat_head,
        text_prompt=prompt_final,
        pool_coherence=compute_pool_coherence(clips),
        same_genre_tight_mix=same_genre,
    )
    n_best = getattr(args, 'n_best', 1)
    if n_best > 1:
        tl, meta = plan_n_best(clips, cfg, cache_dir=args.cache, n_candidates=n_best)
        print(f"N-best rerank picked: score={meta.get('best_score', 0):.3f} "
              f"breakdown={meta.get('best_breakdown', {})}")
    else:
        tl = plan(clips, cfg)

    # Apply Director-suggested transition tiers + accent hints if requested
    if director_out is not None and getattr(args, 'apply_llm_tiers', False):
        tiers = director_out.get('transition_tiers') or []
        accents = director_out.get('accent_hints') or []
        if tiers:
            apply_llm_transition_tiers_to_timeline(tl, tiers, director_out.get("transition_intents"))
            print(f"[director] applied {len(tiers)} LLM tier transitions")
        if accents:
            attach_accent_hints(tl, accents)
            print(f"[director] attached {len(accents)} accent hints")
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
    p.add_argument('--workers', type=int, default=1,
                   help='parallel workers; >=2 spawns process pool. Each loads its own model. Tune for VRAM.')
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser('plan')
    p.add_argument('--cache', default='cache')
    p.add_argument('--out', default='output/timeline.json')
    p.add_argument('--duration', type=float, default=1800.0)
    p.add_argument('--surprises', type=int, default=10)
    p.add_argument('--callbacks', type=int, default=1)
    p.add_argument('--max_clips', type=int, default=200)
    p.add_argument('--min_unique_clips', type=int, default=5,
                   help='min distinct clips that must appear in mix')
    p.add_argument('--arc', default='build',
                   choices=['build', 'peak', 'rollercoaster',
                            'descend', 'flat_high', 'flat_low', 'custom'],
                   help='energy arc shape (planner intent)')
    p.add_argument('--style_rag', default=None,
                   help='reference dir for Style-RAG bias (optional)')
    p.add_argument('--classifier', default=None,
                   help='path to trained technique classifier .pt (optional)')
    p.add_argument('--prompt', default=None,
                   help='natural-language mix description (e.g. "uplifting trance set")')
    p.add_argument('--compat_head', default=None,
                   help='path to CLAP compat head .pt (Tier 1.5, optional)')
    p.add_argument('--n_best', type=int, default=1,
                   help='generate N candidate timelines, rerank by CLAP coherence + vocal-collision penalty, pick best')
    p.add_argument('--use_director', action='store_true',
                   help='use HF Director LLM to refine prompt/arc/budgets/tiers')
    p.add_argument('--apply_llm_tiers', action='store_true',
                   help='replace planner-chosen transitions with LLM tier-mapped tasteful techniques')
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser('execute')
    p.add_argument('--timeline', default='output/timeline.json')
    p.add_argument('--cache', default='cache')
    p.add_argument('--out', default='output/raw_mix.wav')
    p.add_argument('--samples', default='samples')
    p.add_argument('--improve_max_passes', type=int, default=0,
                   help='Probe→improver passes after initial render (0=off, 1+ enable)')
    p.add_argument('--improve_threshold', type=float, default=0.5,
                   help='Stop when overall_severity below this')
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
    p.add_argument('--workers', type=int, default=1,
                   help='parallel analyze workers')
    p.add_argument('--duration', type=float, default=1800.0)
    p.add_argument('--surprises', type=int, default=10)
    p.add_argument('--callbacks', type=int, default=1)
    p.add_argument('--max_clips', type=int, default=200)
    p.add_argument('--min_unique_clips', type=int, default=5,
                   help='min distinct clips that must appear in mix')
    p.add_argument('--arc', default='build',
                   choices=['build', 'peak', 'rollercoaster',
                            'descend', 'flat_high', 'flat_low', 'custom'],
                   help='energy arc shape (planner intent)')
    p.add_argument('--lufs', type=float, default=-9.0)
    p.add_argument('--style_rag', default=None,
                   help='reference dir for Style-RAG bias (optional)')
    p.add_argument('--classifier', default=None,
                   help='path to trained technique classifier .pt (optional)')
    p.add_argument('--prompt', default=None,
                   help='natural-language mix description (e.g. "uplifting trance set")')
    p.add_argument('--compat_head', default=None,
                   help='path to CLAP compat head .pt (Tier 1.5, optional)')
    p.add_argument('--n_best', type=int, default=1,
                   help='N-best candidate generation + heuristic rerank')
    p.add_argument('--use_director', action='store_true',
                   help='use HF Director LLM to refine prompt/arc/budgets/tiers (env: AIJOCKEY_USE_DIRECTOR_LLM=0 forces deterministic fallback)')
    p.add_argument('--apply_llm_tiers', action='store_true',
                   help='replace planner-chosen transitions with LLM tier-mapped tasteful techniques (requires --use_director)')
    p.set_defaults(func=cmd_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
