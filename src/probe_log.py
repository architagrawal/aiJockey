"""Per-render probe logging for baseline distribution + DPO data accumulation.

Atomic JSONL append. One row per render. Becomes:
  - DPO signal (high-severity render → 'rejected' half of pair)
  - Director self-grading (Director plan + outcome severity → adapter)
  - Quality dashboard (severity over time, by mix_mode/arc/tier)
  - Free, no human labels needed.

Schema (one JSON object per line):
  {
    "ts": "2026-05-09T...",
    "job_id": "abc123",
    "prompt": "festival peak",
    "arc": "build",
    "mix_mode": "balanced",
    "duration_target_s": 360,
    "duration_actual_s": 176.1,
    "n_segments": 8,
    "n_user_clips": 2,
    "n_library_clips": 6,
    "director_used": true,
    "director_fallback": false,
    "set_narrative": "...",
    "transition_tiers": ["minor","major","drop",...],
    "transition_intents": ["breath","build_tension",...],
    "beat_source": "madmom" | "beat_this" | "librosa",
    "render_time_s": 5.2,
    "probe": {
      "verdict": "audible_artifacts",
      "overall_severity": 0.94,
      "n_junctions": 7,
      "junctions": [{"idx": 0, "t": 18.8, "rms_db": +1.05, "xcorr": 0.05, "phase": 0.09, "sev": 0.31}, ...]
    },
    "improver": {
      "passes": 0,
      "edits_applied": [],
      "severity_after": null
    }
  }

Path: $AIJOCKEY_PROBE_LOG (default /scratch/probes/log.jsonl, falls back to ./probes_log.jsonl)
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _log_path() -> Path:
    p = os.environ.get('AIJOCKEY_PROBE_LOG')
    if p:
        return Path(p)
    # Prefer /scratch on droplet, fall back to repo-local on laptop.
    for cand in (Path('/scratch/probes/log.jsonl'),
                 Path('/workspace/scratch/probes/log.jsonl'),
                 Path('./probes_log.jsonl')):
        try:
            cand.parent.mkdir(parents=True, exist_ok=True)
            return cand
        except Exception:
            continue
    return Path('./probes_log.jsonl')


def _atomic_append(path: Path, line: str) -> None:
    """Append a single JSONL line atomically.

    Uses tempfile + rename within the same dir to ensure POSIX atomicity.
    For high-frequency append (>1/sec) consider switching to flock+seek;
    /generate frequency is seconds-to-minutes, this is fine.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Read current contents (cheap for JSONL of expected size), append, rename.
    # On first write the file may not exist.
    cur = b''
    if path.exists():
        try:
            cur = path.read_bytes()
        except Exception:
            cur = b''
    payload = cur + line.encode('utf-8') + b'\n'
    fd, tmp = tempfile.mkstemp(prefix='.probelog_', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(payload)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _improver_env_state() -> dict:
    """Snapshot the improver gating env at log time. Required field for
    cohort bucketing in summarize(). Without this, post-hoc analysis can't
    tell which improver knobs were on for any given row.
    """
    return {
        'energy': os.environ.get('AIJOCKEY_IMPROVER_ENERGY', '1'),
        'overlap': os.environ.get('AIJOCKEY_IMPROVER_OVERLAP', '1'),
        'swap': os.environ.get('AIJOCKEY_IMPROVER_SWAP', '1'),
        'cohort': os.environ.get('AIJOCKEY_COHORT', ''),  # human label
    }


def log_render(*,
               job_id: str,
               prompt: str | None = None,
               arc: str | None = None,
               mix_mode: str | None = None,
               duration_target_s: float | None = None,
               duration_actual_s: float | None = None,
               n_user_clips: int | None = None,
               n_library_clips: int | None = None,
               director_used: bool | None = None,
               director_fallback: bool | None = None,
               set_narrative: str | None = None,
               transition_tiers: list[str] | None = None,
               transition_intents: list[str] | None = None,
               beat_source: str | None = None,
               render_time_s: float | None = None,
               probe: dict | None = None,
               improver: dict | None = None,
               extra: dict | None = None,
               ) -> Path:
    """Append one row to the probe log. Returns the log path written."""
    junctions_compact: list[dict] = []
    if probe and probe.get('junctions'):
        for r in probe['junctions']:
            rms = r.get('rms') or {}
            bleed = r.get('vocal_bleed') or {}
            phase = r.get('phasing') or {}
            junctions_compact.append({
                'idx': r.get('junction_index'),
                't': r.get('time_sec'),
                'rms_db': round(float(rms.get('db_diff', 0.0)), 2),
                'xcorr': round(float(bleed.get('xcorr_max', 0.0)), 3),
                'phase': round(float(phase.get('phase_delta', 0.0)), 3),
                'sev': round(float(r.get('overall_severity', 0.0)), 2),
            })
    row: dict[str, Any] = {
        'ts': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'job_id': job_id,
        'prompt': prompt,
        'arc': arc,
        'mix_mode': mix_mode,
        'duration_target_s': duration_target_s,
        'duration_actual_s': duration_actual_s,
        'n_segments': len(junctions_compact) + 1 if junctions_compact else None,
        'n_user_clips': n_user_clips,
        'n_library_clips': n_library_clips,
        'director_used': director_used,
        'director_fallback': director_fallback,
        'set_narrative': set_narrative,
        'transition_tiers': transition_tiers,
        'transition_intents': transition_intents,
        'beat_source': beat_source,
        'render_time_s': render_time_s,
        'probe': {
            'verdict': probe.get('verdict') if probe else None,
            'overall_severity': probe.get('overall_severity') if probe else None,
            'n_junctions': probe.get('n_junctions') if probe else None,
            'junctions': junctions_compact,
        } if probe else None,
        'improver': improver,
    }
    row['improver_state'] = _improver_env_state()
    if extra:
        row['extra'] = extra
    p = _log_path()
    _atomic_append(p, json.dumps(row, default=str, separators=(',', ':')))
    return p


def read_log(path: str | None = None) -> list[dict]:
    """Parse the JSONL log into a list of dicts. Skips malformed rows."""
    p = Path(path) if path else _log_path()
    if not p.exists():
        return []
    out: list[dict] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def summarize(rows: list[dict] | None = None) -> dict:
    """Quick distribution summary — feed to dashboard / quality reports.

    Returns: {n, mean_severity, p50, p90, by_mode, by_arc, fallback_rate}.
    """
    if rows is None:
        rows = read_log()
    if not rows:
        return {'n': 0}
    sevs = [r.get('probe', {}).get('overall_severity')
            for r in rows
            if r.get('probe') and r['probe'].get('overall_severity') is not None]
    sevs = [s for s in sevs if s is not None]
    if not sevs:
        return {'n': len(rows), 'mean_severity': None}
    sevs_sorted = sorted(sevs)
    n = len(sevs_sorted)
    p50 = sevs_sorted[n // 2]
    p90 = sevs_sorted[min(n - 1, int(n * 0.9))]
    by_mode: dict[str, list[float]] = {}
    by_arc: dict[str, list[float]] = {}
    for r in rows:
        s = (r.get('probe') or {}).get('overall_severity')
        if s is None:
            continue
        m = r.get('mix_mode') or 'unknown'
        a = r.get('arc') or 'unknown'
        by_mode.setdefault(m, []).append(float(s))
        by_arc.setdefault(a, []).append(float(s))
    fallback = sum(1 for r in rows if r.get('director_fallback'))
    by_cohort: dict[str, list[float]] = {}
    by_state: dict[str, list[float]] = {}
    for r in rows:
        s = (r.get('probe') or {}).get('overall_severity')
        if s is None:
            continue
        st = r.get('improver_state') or {}
        cohort = st.get('cohort') or 'unlabeled'
        by_cohort.setdefault(cohort, []).append(float(s))
        # State key = "E1O0S0" etc — collapses energy/overlap/swap into one tag
        skey = (f"E{st.get('energy', '?')}"
                f"O{st.get('overlap', '?')}"
                f"S{st.get('swap', '?')}")
        by_state.setdefault(skey, []).append(float(s))
    return {
        'n': len(rows),
        'mean_severity': round(sum(sevs) / len(sevs), 3),
        'p50_severity': round(p50, 3),
        'p90_severity': round(p90, 3),
        'by_mode': {k: {'n': len(v), 'mean': round(sum(v) / len(v), 3)}
                    for k, v in by_mode.items()},
        'by_arc': {k: {'n': len(v), 'mean': round(sum(v) / len(v), 3)}
                   for k, v in by_arc.items()},
        'by_cohort': {k: {'n': len(v), 'mean': round(sum(v) / len(v), 3),
                          'p50': round(sorted(v)[len(v)//2], 3)}
                      for k, v in by_cohort.items()},
        'by_improver_state': {k: {'n': len(v), 'mean': round(sum(v) / len(v), 3)}
                               for k, v in by_state.items()},
        'fallback_rate': round(fallback / len(rows), 3),
    }


def main() -> None:
    """CLI: probe_log summary [--path X]"""
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('summary')
    s.add_argument('--path', default=None)
    s.add_argument('--by_cohort', action='store_true',
                   help='emit per-cohort table only')
    args = ap.parse_args()
    if args.cmd == 'summary':
        rows = read_log(args.path)
        out = summarize(rows)
        if args.by_cohort:
            cohorts = out.get('by_cohort') or {}
            print(f"{'cohort':<14} {'n':>4} {'mean':>7} {'p50':>7}")
            print('-' * 40)
            for c in sorted(cohorts):
                d = cohorts[c]
                print(f"{c:<14} {d['n']:>4} {d['mean']:>7.3f} {d['p50']:>7.3f}")
            return
        print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
