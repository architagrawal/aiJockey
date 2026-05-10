"""Diagnose why expanded transition vocabulary isn't firing.

Run on droplet (or anywhere with the repo loaded):
    python scripts/diagnose_transitions.py [--timeline /scratch/output/last_timeline.json]

Checks 4 things in order:
  1. Catalog reachable + populated
  2. tier_to_technique catalog path firing for each tier (vs falling
     through to legacy hardcoded pool)
  3. Last rendered timeline.json — distribution of transition NAMES that
     actually fired (if all are crossfade/eq_swap, vocal_guard probably
     downgrading everything)
  4. Vocal-activity distribution across the user pool — if >50% sections
     have va > 0.30, vocal_guard fires every junction

Output: per-step verdict + recommended action.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _heading(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Step 1 — catalog reachable + populated
# ---------------------------------------------------------------------------

def check_catalog() -> bool:
    _heading("STEP 1 — Transition catalog reachable")
    try:
        from transition_catalog import all_techniques, list_techniques
    except ImportError as e:
        print(f"  ✗ FAIL: cannot import transition_catalog: {e}")
        print(f"      → catalog wasn't pulled. git fetch + checkout tier1-upgrades.")
        return False

    techs = all_techniques()
    if not techs:
        print(f"  ✗ FAIL: catalog loaded but empty.")
        print(f"      → datasets/transitions/catalog.json missing or malformed.")
        print(f"      → look for: {ROOT / 'datasets' / 'transitions' / 'catalog.json'}")
        return False

    impl = list_techniques(status="implemented")
    print(f"  ✓ catalog has {len(techs)} techniques, {len(impl)} implemented")
    by_tier: dict[str, int] = {}
    for t in impl:
        by_tier.setdefault(t.get("tier", "?"), 0)
        by_tier[t.get("tier", "?")] += 1
    for tier in ("minor", "major", "drop", "cut", "loop"):
        n = by_tier.get(tier, 0)
        marker = "✓" if n >= 3 else "!"
        print(f"    {marker} tier={tier}: {n} implemented techniques")
    return True


# ---------------------------------------------------------------------------
# Step 2 — tier_to_technique catalog path firing
# ---------------------------------------------------------------------------

def check_tier_mapping() -> bool:
    _heading("STEP 2 — tier_to_technique catalog path")
    try:
        from transition_mapping import tier_to_technique
    except ImportError as e:
        print(f"  ✗ FAIL: cannot import transition_mapping: {e}")
        return False

    print(f"  Cycling junctions 0..7 per tier (vocal_active=False, no section_label):")
    print()
    fail = False
    for tier in ("minor", "major", "drop", "loop", "cut"):
        names = []
        for j in range(8):
            tech = tier_to_technique(tier, j, vocal_active=False)
            names.append(tech.get("name", "?"))
            catalog_picked = tech.get("catalog_picked", False)
        unique = set(names)
        marker = "✓" if len(unique) >= 3 else "!"
        path = "catalog" if catalog_picked else "legacy"
        print(f"    {marker} tier={tier:6s}  unique={len(unique):2d}  "
              f"path={path}  names={names}")
        if not catalog_picked:
            fail = True

    if fail:
        print()
        print(f"  ! WARNING: at least one tier fell through to LEGACY hardcoded pool")
        print(f"    → catalog import works but technique_for_context returned empty")
        print(f"    → check that catalog entries have implementation_status='implemented'")

    print()
    print(f"  With vocal_active=True (simulating vocal pool):")
    for tier in ("minor", "major", "drop", "loop"):
        names = []
        for j in range(6):
            tech = tier_to_technique(tier, j, vocal_active=True)
            names.append(tech.get("name", "?"))
        unique = set(names)
        marker = "✓" if len(unique) >= 2 else "!"
        print(f"    {marker} tier={tier:6s}  unique={len(unique)}  names={names}")
    return not fail


# ---------------------------------------------------------------------------
# Step 3 — actual transition distribution from last render
# ---------------------------------------------------------------------------

def check_render_distribution(timeline_path: Path | None) -> None:
    _heading("STEP 3 — Last render's transition distribution")
    if timeline_path is None:
        # Search common locations
        candidates = [
            ROOT / "output" / "timeline.json",
            ROOT / "output" / "live",
            Path("/scratch/output"),
            Path("/scratch/renders"),
        ]
        timeline_path = None
        for p in candidates:
            if p.exists():
                if p.is_file():
                    timeline_path = p
                    break
                # find newest *.json
                jsons = sorted(p.rglob("timeline*.json"), key=lambda x: -x.stat().st_mtime)
                if jsons:
                    timeline_path = jsons[0]
                    break

    if timeline_path is None or not timeline_path.exists():
        print(f"  ! no timeline.json found")
        print(f"    → render at least one mix, then re-run with --timeline path/to/timeline.json")
        return

    print(f"  Reading: {timeline_path}")
    try:
        data = json.loads(timeline_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ✗ failed to parse: {e}")
        return

    tl = data.get("timeline") or data
    if not isinstance(tl, list) or not tl:
        print(f"  ! timeline empty")
        return

    techs: list[dict] = []
    tiers: list[str] = []
    downgraded: list[dict] = []
    for entry in tl:
        ti = entry.get("transition_in") or {}
        if not ti:
            continue
        techs.append(ti)
        tiers.append(ti.get("tier", "?"))
        if "_vocal_guard_downgraded_from" in ti:
            downgraded.append(ti)

    name_dist = Counter(t.get("name", "?") for t in techs)
    tier_dist = Counter(tiers)

    print(f"  Total junctions: {len(techs)}")
    print(f"  Tier distribution: {dict(tier_dist)}")
    print(f"  Technique distribution:")
    for n, c in name_dist.most_common():
        print(f"    {n:25s}: {c}")

    if len(name_dist) <= 3 and len(techs) >= 5:
        print()
        print(f"  ! ALERT: only {len(name_dist)} unique transition names across {len(techs)} junctions")
        print(f"    → expanded vocab is NOT firing")

    if downgraded:
        print()
        print(f"  ! VOCAL_GUARD fired {len(downgraded)} times — downgraded:")
        for d in downgraded[:5]:
            print(f"      {d.get('_vocal_guard_downgraded_from')} → {d.get('name')}")
        print(f"    → user pool is vocal-heavy; aggressive techs auto-rejected")
        print(f"    → fix: balance pool VA via library_picker_score.vocal_diversity_bonus")
        print(f"           OR set AIJOCKEY_VOCAL_GUARD=0 (loses vocal protection)")


# ---------------------------------------------------------------------------
# Step 4 — pool vocal_activity distribution
# ---------------------------------------------------------------------------

def check_pool_vocals(cache_dir: Path | None = None) -> None:
    _heading("STEP 4 — User pool vocal_activity distribution")
    if cache_dir is None:
        for p in (Path("/scratch/cache"),
                   Path("/cache"),
                   ROOT / "cache",
                   ROOT / "test_user_cache_v6"):
            if p.exists():
                cache_dir = p
                break
    if cache_dir is None or not cache_dir.exists():
        print(f"  ! no cache dir found — pass --cache /path/to/cache to inspect")
        return

    jsons = list(cache_dir.glob("*.json"))
    jsons = [j for j in jsons if j.name != "source_map.json"]
    if not jsons:
        print(f"  ! no clip JSONs in {cache_dir}")
        return

    print(f"  Reading {len(jsons)} clip JSONs from {cache_dir}")
    high_va_clips = 0
    high_va_sections = 0
    total_sections = 0
    pool_means: list[float] = []

    for jp in jsons:
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        sections = d.get("sections") or []
        clip_vas: list[float] = []
        for s in sections:
            va = s.get("vocal_activity")
            if isinstance(va, (int, float)):
                total_sections += 1
                clip_vas.append(float(va))
                if va > 0.30:
                    high_va_sections += 1
        if clip_vas:
            mean = sum(clip_vas) / len(clip_vas)
            pool_means.append(mean)
            if mean > 0.50:
                high_va_clips += 1

    if not pool_means:
        print(f"  ! no clip has vocal_activity tagged on its sections")
        print(f"    → analyzer didn't backfill VA. run scripts/backfill_vocal_activity.py")
        return

    pool_va = sum(pool_means) / len(pool_means)
    print(f"  Pool mean vocal_activity:        {pool_va:.3f}")
    print(f"  Clips with mean VA > 0.50:       {high_va_clips}/{len(pool_means)}")
    print(f"  Sections with VA > 0.30:         {high_va_sections}/{total_sections}")
    print()
    if pool_va > 0.50:
        print(f"  ! POOL IS VOCAL-HEAVY (mean {pool_va:.2f})")
        print(f"    → vocal_guard fires every junction → expanded vocab dies")
        print(f"    → wire library_picker_score.vocal_diversity_bonus to balance")
        print(f"    → OR ship `mix_mode=tight` for vocal-only commercial-pop pools")
    elif pool_va < 0.20:
        print(f"  ! POOL IS INSTRUMENTAL-HEAVY (mean {pool_va:.2f})")
        print(f"    → all aggressive techs CAN fire — if they're not, root cause is elsewhere")
    else:
        print(f"  ✓ pool VA balanced — vocal_guard fires selectively")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default=None,
                     help="path to a rendered timeline.json")
    ap.add_argument("--cache", default=None,
                     help="path to clip cache dir (analyzed JSONs)")
    args = ap.parse_args()

    catalog_ok = check_catalog()
    if catalog_ok:
        check_tier_mapping()

    timeline_path = Path(args.timeline) if args.timeline else None
    check_render_distribution(timeline_path)

    cache_dir = Path(args.cache) if args.cache else None
    check_pool_vocals(cache_dir)

    print()
    print("=" * 72)
    print("DIAGNOSIS COMPLETE")
    print("=" * 72)
    print("Most likely root causes (in order):")
    print("  1. Render branch lacks tier1-upgrades (catalog absent)")
    print("  2. vocal_guard downgrading every junction (vocal-heavy pool)")
    print("  3. tier_to_technique catalog path silently falling to legacy")
    print("  4. Director output shape doesn't include `name` override")
    return 0


if __name__ == "__main__":
    sys.exit(main())
