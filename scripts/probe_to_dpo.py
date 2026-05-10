"""Convert probe_log JSONL into DPO preference pairs for S7.

Strategy: pair-wise ranking by probe severity within bucket. For each
(prompt, mix_mode) bucket, take pairs (low_severity, high_severity) and
emit a DPO row where the low-severity Director plan is "chosen" and the
high-severity is "rejected".

This uses the probe severity as automated reward signal — no human labels
required. Naturally biased toward the metric we already track (probe is
imperfect proxy; DPO will inherit that bias). Iterate the metric, the
preferences automatically improve.

Output schema (per ORPO/DPO TRL trainer):
  {
    "prompt": "<system + user message that produced the plan>",
    "chosen": "<low-severity Director JSON, formatted as assistant turn>",
    "rejected": "<high-severity Director JSON, formatted as assistant turn>",
    "_chosen_severity": 0.42,
    "_rejected_severity": 0.91,
    "_chosen_job_id": "...",
    "_rejected_job_id": "...",
    "_bucket": "<prompt>|<mix_mode>"
  }

Usage:
    python scripts/probe_to_dpo.py \
        --log /scratch/probes/log_cohorts.jsonl \
        --out /scratch/preferences/dpo_pairs.jsonl \
        --min_delta 0.20

Emits ~N(N-1)/2 pairs per bucket capped by --max_pairs_per_bucket.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SYSTEM_PROMPT_PLACEHOLDER = (
    "You are a club DJ planning a tightly-mixed offline set. "
    "Output ONLY valid JSON. (Full system prompt: see src/director.py)"
)


def _row_to_director_plan(row: dict) -> dict:
    """Reconstruct the Director's emitted JSON plan from a logged row.
    The probe_log captures key fields; we serialize back to JSON the way
    Director would have emitted (chat-template assistant turn)."""
    return {
        "arc": row.get("arc"),
        "set_narrative": row.get("set_narrative"),
        "transition_tiers": row.get("transition_tiers") or [],
        "transition_intents": row.get("transition_intents") or [],
    }


def _user_prompt_text(row: dict) -> str:
    """Reconstruct the user-message text from logged row."""
    parts = []
    if row.get("prompt"):
        parts.append(f"User DJ request: {row['prompt']}")
    if row.get("arc"):
        parts.append(f"Arc preset: {row['arc']}")
    if row.get("mix_mode"):
        parts.append(f"Mix mode: {row['mix_mode']}")
    if row.get("n_user_clips") is not None:
        parts.append(f"User clips: {row['n_user_clips']}")
    if row.get("n_library_clips") is not None:
        parts.append(f"Library clips: {row['n_library_clips']}")
    return "\n".join(parts) if parts else "<no prompt>"


def export_pairs(log_path: Path, out_path: Path,
                  min_delta: float = 0.20,
                  max_pairs_per_bucket: int = 20,
                  drop_fallback: bool = True) -> dict:
    """Read probe log JSONL, emit DPO pairs to out_path. Returns stats."""
    rows: list[dict] = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    # Filter: must have probe.overall_severity + Director plan + non-fallback.
    eligible = []
    for r in rows:
        sev = (r.get("probe") or {}).get("overall_severity")
        if sev is None:
            continue
        if drop_fallback and r.get("director_fallback"):
            continue
        if not (r.get("transition_tiers") or r.get("set_narrative")):
            continue
        eligible.append(r)

    # Bucket by (prompt, mix_mode) so we compare like-with-like.
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in eligible:
        key = f"{r.get('prompt') or '?'}|{r.get('mix_mode') or '?'}"
        buckets[key].append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    n_skipped_delta = 0
    with open(out_path, "w") as out_f:
        for bkey, brows in buckets.items():
            if len(brows) < 2:
                continue
            # Sort ascending by severity. low = chosen, high = rejected.
            brows = sorted(brows,
                           key=lambda r: (r.get("probe") or {}).get(
                               "overall_severity", 1.0))
            pairs_this_bucket = 0
            for i in range(len(brows)):
                for j in range(i + 1, len(brows)):
                    if pairs_this_bucket >= max_pairs_per_bucket:
                        break
                    chosen = brows[i]
                    rejected = brows[j]
                    delta = ((rejected.get("probe") or {}).get(
                              "overall_severity", 1.0)
                             - (chosen.get("probe") or {}).get(
                                 "overall_severity", 0.0))
                    if delta < min_delta:
                        n_skipped_delta += 1
                        continue
                    pair = {
                        "prompt": _user_prompt_text(chosen),
                        "system": SYSTEM_PROMPT_PLACEHOLDER,
                        "chosen": json.dumps(_row_to_director_plan(chosen),
                                              separators=(",", ":")),
                        "rejected": json.dumps(_row_to_director_plan(rejected),
                                                separators=(",", ":")),
                        "_chosen_severity": (chosen.get("probe") or {}).get(
                            "overall_severity"),
                        "_rejected_severity": (rejected.get("probe") or {}).get(
                            "overall_severity"),
                        "_chosen_job_id": chosen.get("job_id"),
                        "_rejected_job_id": rejected.get("job_id"),
                        "_bucket": bkey,
                        "_delta": round(delta, 3),
                    }
                    out_f.write(json.dumps(pair) + "\n")
                    n_pairs += 1
                    pairs_this_bucket += 1
                if pairs_this_bucket >= max_pairs_per_bucket:
                    break

    return {
        "n_rows_total": len(rows),
        "n_eligible": len(eligible),
        "n_buckets": len(buckets),
        "n_pairs_written": n_pairs,
        "n_skipped_low_delta": n_skipped_delta,
        "out_path": str(out_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="/scratch/preferences/dpo_pairs.jsonl")
    ap.add_argument("--min_delta", type=float, default=0.20,
                    help="minimum severity gap to count as a pair")
    ap.add_argument("--max_pairs_per_bucket", type=int, default=20)
    ap.add_argument("--include_fallback", action="store_true")
    args = ap.parse_args()
    stats = export_pairs(
        Path(args.log), Path(args.out),
        min_delta=args.min_delta,
        max_pairs_per_bucket=args.max_pairs_per_bucket,
        drop_fallback=not args.include_fallback,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
