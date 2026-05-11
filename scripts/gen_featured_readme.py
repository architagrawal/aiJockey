"""Auto-generate featured/README.md from gallery WAV/MP3 files.

For each track in /workspace/output/featured (recursive), extract:
    - PQ from filename (pattern: `NN_PQ7.87_slug.wav`) or live audiobox-score
    - duration via ffprobe
    - file size

Writes a sorted markdown table.

Run:
    /opt/venv/bin/python scripts/gen_featured_readme.py \\
        --root /workspace/output/featured \\
        --out  /workspace/output/featured/README.md
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PQ_RE = re.compile(r"PQ(\d+(?:\.\d+)?)", re.IGNORECASE)


def _ffprobe_duration(p: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
            timeout=10,
        )
        return float(out.decode().strip())
    except Exception:
        return 0.0


def _pq_from_name(name: str) -> float | None:
    m = PQ_RE.search(name)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _pq_live(p: Path) -> float | None:
    try:
        from audiobox_critic import enabled, score
        if not enabled():
            return None
        s = score(str(p))
        if s:
            return float(s.get("PQ", 0.0))
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"missing: {root}")

    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".mp3", ".wav", ".flac"):
            continue
        pq = _pq_from_name(p.name) or _pq_live(p) or 0.0
        dur = _ffprobe_duration(p)
        size_mb = p.stat().st_size / 1e6
        rows.append({"path": p, "pq": pq, "duration": dur, "size_mb": size_mb})

    rows.sort(key=lambda r: -r["pq"])

    lines = [
        "# Featured Gallery",
        "",
        "Auto-generated. PQ = Audiobox Production Quality (higher = better).",
        "",
        "| # | File | PQ | Duration | Size |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, start=1):
        rel = r["path"].relative_to(root)
        mins = int(r["duration"] // 60)
        secs = int(r["duration"] % 60)
        lines.append(f"| {i} | `{rel}` | {r['pq']:.2f} | "
                      f"{mins}:{secs:02d} | {r['size_mb']:.1f} MB |")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} entries -> {args.out}")


if __name__ == "__main__":
    main()
