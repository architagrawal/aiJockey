"""One-shot: walk a cache dir, apply tempo_octave.normalize_tempo to each
clip JSON's `tempo` field. Fixes trap half-time / dnb double-time
interpretations downstream without re-running Beat-This!.

Reads `<cache_dir>/<clip_id>.json`, applies normalize_tempo(bpm, genre),
writes back atomically. Genre extracted from clip_id prefix (e.g.
'trap__Future_-_Mask_Off' → 'trap'). Falls back to genre-less
normalization if no prefix.

Idempotent: re-running on already-normalized cache is a no-op (the
normalizer is a fixed point for tempos already in canonical band).

Adds field `tempo_original` so the pre-normalization value is preserved
for audit / revert.

Usage:
    python scripts/normalize_cache_tempos.py /cache
    python scripts/normalize_cache_tempos.py /scratch/cache --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tempo_octave import normalize_tempo, CANONICAL_LO, CANONICAL_HI


# Map common clip-id prefixes (genre tag) to genre keys understood by
# tempo_octave. Anything else passes None (canonical band only).
_PREFIX_GENRE = {
    "dnb": "drum_and_bass",
    "drum_and_bass": "drum_and_bass",
    "drumnbass": "drum_and_bass",
    "jungle": "jungle",
    "footwork": "footwork",
    "hardcore": "hardcore",
    "ambient": "ambient",
    "lofi": "lofi",
    "lofi_hip_hop": "lofi_hip_hop",
    "downtempo": "downtempo",
    "classical": "classical",
    "chill": "chillout",
    "ghazal": "ghazal",
}


def _genre_from_clip_id(cid_or_filename: str) -> str | None:
    base = cid_or_filename.lower()
    # strip leading directory components
    base = base.split("/")[-1].split("\\")[-1]
    # strip .json
    if base.endswith(".json"):
        base = base[:-5]
    # split on '__' (common pattern: 'genre__title__id') or '_'
    for sep in ("__", "_"):
        head = base.split(sep, 1)[0]
        if head in _PREFIX_GENRE:
            return _PREFIX_GENRE[head]
    return None


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", help="path to cache dir (e.g. /cache or /scratch/cache)")
    ap.add_argument("--dry-run", action="store_true",
                     help="report changes without writing")
    ap.add_argument("--force", action="store_true",
                     help="re-normalize even when tempo_original already present")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    if not cache.is_dir():
        sys.exit(f"not a directory: {cache}")

    json_files = sorted(cache.glob("*.json"))
    json_files = [p for p in json_files if p.name != "source_map.json"]
    if not json_files:
        sys.exit(f"no clip jsons in {cache}")

    print(f"scanning {len(json_files)} clip jsons under {cache}")
    print(f"canonical band: [{CANONICAL_LO}, {CANONICAL_HI}] BPM")
    print()

    changed = 0
    skipped_already = 0
    skipped_no_change = 0
    failed = 0
    histogram = {"halftime_doubled": 0, "doubletime_halved": 0,
                  "in_band": 0, "missing": 0}

    for fp in json_files:
        try:
            meta = json.loads(fp.read_text())
        except Exception as e:
            failed += 1
            print(f"  ! {fp.name}: parse failed ({e})")
            continue

        if not args.force and "tempo_original" in meta:
            skipped_already += 1
            continue

        bpm = meta.get("tempo")
        if not isinstance(bpm, (int, float)) or bpm <= 0:
            histogram["missing"] += 1
            continue
        bpm = float(bpm)

        genre = _genre_from_clip_id(fp.stem)
        new_bpm = normalize_tempo(bpm, genre=genre)

        if abs(new_bpm - bpm) < 0.01:
            histogram["in_band"] += 1
            skipped_no_change += 1
            continue

        # Tag direction for histogram
        if new_bpm > bpm * 1.5:
            histogram["halftime_doubled"] += 1
        elif new_bpm < bpm * 0.75:
            histogram["doubletime_halved"] += 1

        if args.dry_run:
            print(f"  {fp.name}: {bpm:.1f} → {new_bpm:.1f} (genre={genre or '-'})")
            changed += 1
            continue

        meta["tempo_original"] = bpm
        meta["tempo"] = new_bpm
        meta["tempo_normalized_at"] = "tempo_octave"
        try:
            _atomic_write_json(fp, meta)
            changed += 1
            print(f"  {fp.name}: {bpm:.1f} → {new_bpm:.1f} (genre={genre or '-'})")
        except Exception as e:
            failed += 1
            print(f"  ! {fp.name}: write failed ({e})")

    print()
    print("=" * 60)
    print(f"changed:          {changed}")
    print(f"skipped (already): {skipped_already}")
    print(f"skipped (no change): {skipped_no_change}")
    print(f"failed:           {failed}")
    print()
    print("change directions:")
    for k, v in histogram.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\n(dry run — no files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
