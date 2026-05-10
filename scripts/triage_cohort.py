"""Read a single cohort probe-log JSONL, print verdict + top failure modes.

Aligned to teammate's `probe_log.py` schema (commit 0b2c742):
  - All cohort rows in one file (e.g. `/scratch/probes/log_cohorts.jsonl`)
  - Each row has `improver_state = {energy, overlap, swap, cohort}`
  - Cohort labels: typically 'A', 'B', 'C', 'D' (set by run_baseline.sh
    via AIJOCKEY_COHORT env at render time). When unlabeled, derive from
    {energy, overlap, swap} flags:
        000 → A   100 → B   110 → C   111 → D

For each cohort:
  - n rows, mean/p50/p90 severity, % below 0.5, % below 0.3
  - top issue types (energy/phase/xcorr threshold breaches)
A vs D delta → verdict bucket + recommendation.

Verdict buckets:
  Δ ≥ 0.3      → BIG_WIN — ship improvers ON, move to listen test
  0.1 ≤ Δ < 0.3 → MODEST — lock + generalize to /cache pool
  -0.05 ≤ Δ < 0.1 → NO_SIGNAL — escalate to LLM critic fallback
  Δ < -0.05    → REGRESSION — revert + debug per-improver

Exit codes: 0 ship-ready, 1 ineffective, 2 regression, 3 no data.

Usage:
  python scripts/triage_cohort.py                                # uses default log
  python scripts/triage_cohort.py --log /scratch/probes/log_cohorts.jsonl
  python scripts/triage_cohort.py --log /scratch/probes/log.jsonl --since 2026-05-10T05
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

COHORT_LABEL = {
    'A': 'baseline (all OFF)',
    'B': 'ENERGY only',
    'C': 'ENERGY + OVERLAP',
    'D': 'ENERGY + OVERLAP + SWAP',
}


def _truthy(v) -> bool:
    s = str(v).strip().lower()
    return s in ('1', 'true', 'yes', 'on')


def _derive_cohort(state: dict) -> str:
    """Derive cohort letter from improver_state booleans."""
    if not isinstance(state, dict):
        return 'unlabeled'
    label = state.get('cohort')
    if label and label.strip():
        return label.strip()
    e = _truthy(state.get('energy'))
    o = _truthy(state.get('overlap'))
    s = _truthy(state.get('swap'))
    if not e and not o and not s:
        return 'A'
    if e and not o and not s:
        return 'B'
    if e and o and not s:
        return 'C'
    if e and o and s:
        return 'D'
    return f"E{int(e)}O{int(o)}S{int(s)}"


def _severity(row: dict) -> float | None:
    """Pull overall severity from probe schema."""
    p = row.get('probe') or row.get('probes') or {}
    for k in ('overall_severity', 'severity'):
        if isinstance(p, dict):
            v = p.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        v = row.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _junctions(row: dict) -> list[dict]:
    p = row.get('probe') or row.get('probes') or {}
    if isinstance(p, dict):
        v = p.get('junctions')
        if isinstance(v, list):
            return v
    for k in ('junctions', 'issues', 'findings'):
        v = row.get(k)
        if isinstance(v, list):
            return v
    return []


def _load_rows(path: Path, since: str | None = None) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if since and row.get('ts', '') < since:
            continue
        rows.append(row)
    return rows


def _summarize(rows: list[dict], label: str) -> dict:
    sevs = [s for s in (_severity(r) for r in rows) if s is not None]
    if not sevs:
        return {'label': label, 'n': 0}
    sevs_sorted = sorted(sevs)
    return {
        'label': label,
        'n': len(sevs),
        'mean': statistics.fmean(sevs),
        'median': statistics.median(sevs_sorted),
        'p10': sevs_sorted[max(0, int(len(sevs_sorted) * 0.10) - 1)],
        'p90': sevs_sorted[min(len(sevs_sorted) - 1, int(len(sevs_sorted) * 0.90))],
        'min': sevs_sorted[0],
        'max': sevs_sorted[-1],
        'pct_under_05': sum(1 for s in sevs if s < 0.5) / len(sevs) * 100,
        'pct_under_03': sum(1 for s in sevs if s < 0.3) / len(sevs) * 100,
    }


def _top_issue_types(rows: list[dict],
                      energy_db_threshold: float = 2.0,
                      phase_threshold: float = 0.20,
                      xcorr_threshold: float = 0.20,
                      n: int = 4) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for r in rows:
        for issue in _junctions(r):
            t = issue.get('issue_type') or issue.get('type')
            if t:
                counts[t] = counts.get(t, 0) + 1
                continue
            rms = issue.get('rms_db', issue.get('energy_db'))
            phase = issue.get('phase')
            xcorr = issue.get('xcorr')
            if isinstance(rms, (int, float)) and abs(rms) >= energy_db_threshold:
                counts['energy_mismatch'] = counts.get('energy_mismatch', 0) + 1
            if isinstance(phase, (int, float)) and phase >= phase_threshold:
                counts['phase_cancel'] = counts.get('phase_cancel', 0) + 1
            if isinstance(xcorr, (int, float)) and xcorr >= xcorr_threshold:
                counts['vocal_collision'] = counts.get('vocal_collision', 0) + 1
    return sorted(counts.items(), key=lambda x: -x[1])[:n]


def _hist_bar(value: float, max_value: float, width: int = 40) -> str:
    if max_value <= 0:
        return ''
    return '█' * max(0, int(value / max_value * width))


def _print_summary(s: dict) -> None:
    if s['n'] == 0:
        print(f"  {s['label']:32s} : no data")
        return
    print(f"  {s['label']:32s} n={s['n']:3d}  "
          f"mean={s['mean']:.3f}  p50={s['median']:.3f}  p90={s['p90']:.3f}  "
          f"<0.5={s['pct_under_05']:.0f}%  <0.3={s['pct_under_03']:.0f}%")


def _verdict(delta: float, A: dict, D: dict) -> tuple[int, str, str]:
    if A.get('n', 0) == 0 or D.get('n', 0) == 0:
        return (3, 'NO_DATA', 'cohort A or D has no rows — re-run baseline')
    if delta >= 0.3:
        return (0, 'BIG_WIN', 'ship improvers ON by default → move to listen test')
    if delta >= 0.1:
        return (0, 'MODEST', "lock current improver set → generalize to /cache pool")
    if delta >= -0.05:
        return (1, 'NO_SIGNAL', "probes flag issues but improvers can't fix → "
                                 "add LLM critic fallback OR new improver types")
    return (2, 'REGRESSION', 'improvers actively hurt → revert all three knobs, '
                              'debug per-improver in isolation')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=os.environ.get(
        'AIJOCKEY_PROBE_LOG', '/scratch/probes/log_cohorts.jsonl'),
        help='path to probe JSONL (default: $AIJOCKEY_PROBE_LOG or /scratch/probes/log_cohorts.jsonl)')
    ap.add_argument('--since', default=None,
                     help='only include rows with ts >= this ISO prefix '
                          '(e.g. 2026-05-10T05)')
    args = ap.parse_args()

    log_path = Path(args.log)
    rows = _load_rows(log_path, since=args.since)
    if not rows:
        print(f"no rows in {log_path}", file=sys.stderr)
        return 3

    # Bucket by cohort
    by_cohort: dict[str, list[dict]] = {}
    for r in rows:
        c = _derive_cohort(r.get('improver_state') or {})
        by_cohort.setdefault(c, []).append(r)

    print('=' * 80)
    print(f"COHORT BASELINE TRIAGE — {log_path} ({len(rows)} rows)")
    print('=' * 80)
    print()

    # Sort cohorts: A B C D first, then alphabetical extras
    canonical = ['A', 'B', 'C', 'D']
    extras = sorted(c for c in by_cohort if c not in canonical)
    cohort_order = [c for c in canonical if c in by_cohort] + extras

    print('Severity distributions:')
    summaries: dict[str, dict] = {}
    for c in cohort_order:
        label = COHORT_LABEL.get(c, c)
        s = _summarize(by_cohort[c], f"{c}: {label}")
        summaries[c] = s
        _print_summary(s)
    print()

    # Histogram bars on mean
    means = [s.get('mean', 0) for s in summaries.values() if s.get('n', 0)]
    if means:
        max_mean = max(means)
        print('Mean severity (lower = better):')
        for c in cohort_order:
            s = summaries[c]
            if s.get('n', 0) == 0:
                continue
            print(f"  {c}: {s['mean']:.3f}  {_hist_bar(s['mean'], max_mean)}")
        print()

    # Top issue types per cohort
    print('Top issue types per cohort:')
    for c in cohort_order:
        rs = by_cohort.get(c, [])
        if not rs:
            continue
        top = _top_issue_types(rs)
        if not top:
            print(f"  {c}: (no junction-level issues parsed)")
            continue
        items = '  '.join(f"{t}={n}" for t, n in top)
        print(f"  {c}: {items}")
    print()

    # Verdict
    A = summaries.get('A', {})
    D = summaries.get('D', {})
    if A.get('n') and D.get('n'):
        delta = A['mean'] - D['mean']
        exit_code, bucket, rec = _verdict(delta, A, D)
        print('=' * 80)
        print(f"VERDICT: cohort A→D mean delta = {delta:+.3f}  ({bucket})")
        print(f"  recommendation: {rec}")
        print('=' * 80)
        return exit_code

    print('VERDICT: insufficient data (need both A and D rows)', file=sys.stderr)
    return 3


if __name__ == '__main__':
    sys.exit(main())
