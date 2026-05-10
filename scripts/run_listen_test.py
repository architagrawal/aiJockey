"""Render a fixed prompt set on N branch states, save to blind-shuffled folders.

Workflow:
  1. Load test prompt list from --prompts (default: scripts/prompts/listen_test.json)
  2. For each branch ref in --refs (default current HEAD only), checkout, render
     each prompt against the same user-clip pool, save to a labeled subfolder.
  3. Build a manifest CSV mapping shuffled file names to (ref, prompt_id), so
     a human can listen blind and rank.
  4. Run CriticV2 score (when checkpoint exists) for each render — gives a
     numeric reference alongside human ranking.

This is the measurement infra for "did evening session improve quality vs
morning session". Run once per significant code change. Output goes under
output/listen_test/<run_id>/.

Usage:
    python scripts/run_listen_test.py \\
        --refs HEAD,5b4f9cd \\
        --prompts scripts/prompts/listen_test.json \\
        --user-pool clips_test \\
        --duration 120

  Then play files in output/listen_test/<run_id>/blind/ in random order, write
  rank into the rankings CSV. Reveal mapping after.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                        capture_output=True, text=True, timeout=1800)
    return p.returncode, (p.stdout or "") + "\n" + (p.stderr or "")


def _git_current_ref() -> str:
    rc, out = _run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
    return out.strip().splitlines()[0] if rc == 0 else "unknown"


def _git_is_clean() -> bool:
    rc, out = _run(["git", "status", "--porcelain"], cwd=ROOT)
    return rc == 0 and not out.strip()


def _checkout(ref: str) -> None:
    rc, out = _run(["git", "checkout", ref], cwd=ROOT)
    if rc != 0:
        raise RuntimeError(f"git checkout {ref} failed: {out[:300]}")


def _render_one(prompt: str, arc: str, user_pool: Path, out_path: Path,
                duration: int, mix_mode: str) -> tuple[bool, str]:
    """Render via src/main.py CLI — keeps decoupled from FastAPI."""
    cmd = [
        sys.executable, "-m", "src.main",
        "all",
        "--clips", str(user_pool),
        "--cache", "cache",
        "--out", str(out_path),
        "--prompt", prompt,
        "--arc", arc,
        "--duration", str(duration),
        "--use_director", "1",
    ]
    if mix_mode:
        cmd += ["--mix_mode", mix_mode]
    rc, out = _run(cmd, cwd=ROOT)
    return rc == 0, out


def _try_critic_score(audio_path: Path) -> float | None:
    """Score rendered audio with CriticV2 if checkpoint available."""
    ck = ROOT / "checkpoints" / "mix_critic.pt"
    ck_v2 = ROOT / "checkpoints" / "mix_critic_v2.pt"
    use = ck_v2 if ck_v2.exists() else (ck if ck.exists() else None)
    if use is None:
        return None
    try:
        from training.mix_critic import score_audio   # type: ignore
        return float(score_audio(str(audio_path), checkpoint=str(use)))
    except Exception as e:
        print(f"[critic] score failed ({e}); skipping")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="HEAD",
                    help="Comma-separated git refs (e.g. 'HEAD,abc123,main')")
    ap.add_argument("--prompts", default="scripts/prompts/listen_test.json",
                    help="JSON list of {id, prompt, arc} dicts")
    ap.add_argument("--user-pool", default="clips_test",
                    help="Directory of user clips to render against")
    ap.add_argument("--duration", type=int, default=120)
    ap.add_argument("--mix-mode", default="balanced",
                    choices=["tight", "balanced", "exploratory"])
    ap.add_argument("--out-root", default="output/listen_test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not _git_is_clean():
        print("WARN: working tree dirty. Stash or commit before running across refs.",
              file=sys.stderr)
        if "," in args.refs:
            sys.exit(1)

    prompts_path = ROOT / args.prompts
    if not prompts_path.exists():
        # Default minimal listen-test set if none authored yet.
        prompts_path.parent.mkdir(parents=True, exist_ok=True)
        prompts_path.write_text(json.dumps([
            {"id": "lt01", "prompt": "warmup deep house slow burn", "arc": "build"},
            {"id": "lt02", "prompt": "festival peak euphoric drops", "arc": "peak"},
            {"id": "lt03", "prompt": "after-hours techno hypnotic", "arc": "flat_low"},
            {"id": "lt04", "prompt": "melodic house emotional sunset", "arc": "build"},
            {"id": "lt05", "prompt": "trance peak time euphoria", "arc": "peak"},
        ], indent=2))
        print(f"created default {prompts_path}")
    prompts = json.loads(prompts_path.read_text())

    user_pool = (ROOT / args.user_pool).resolve()
    if not user_pool.exists():
        sys.exit(f"user-pool {user_pool} does not exist")

    refs = [r.strip() for r in args.refs.split(",") if r.strip()]
    starting_ref = _git_current_ref()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / args.out_root / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    rendered: list[Path] = []

    print(f"=== listen-test run {run_id} | refs={refs} | prompts={len(prompts)} ===")

    try:
        for ref in refs:
            if ref != "HEAD":
                _checkout(ref)
            rev = _git_current_ref()
            ref_dir = out_root / "labeled" / rev
            ref_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n--- ref={ref} (resolved {rev}) ---")

            for p in prompts:
                pid = p["id"]
                out_wav = ref_dir / f"{pid}.wav"
                t0 = time.time()
                ok, log = _render_one(
                    p["prompt"], p.get("arc", "build"),
                    user_pool, out_wav,
                    args.duration, args.mix_mode,
                )
                dur = time.time() - t0
                if not ok or not out_wav.exists():
                    print(f"  ! {pid} FAILED ({dur:.0f}s)")
                    (ref_dir / f"{pid}.log").write_text(log[-4000:])
                    continue
                score = _try_critic_score(out_wav)
                manifest_rows.append({
                    "ref": rev,
                    "prompt_id": pid,
                    "prompt": p["prompt"],
                    "arc": p.get("arc", "build"),
                    "out_path": str(out_wav.relative_to(out_root)),
                    "wall_s": round(dur, 1),
                    "critic_score": score if score is not None else "",
                })
                rendered.append(out_wav)
                print(f"  ✓ {pid} {dur:.0f}s critic={score}")
    finally:
        # Always restore the original ref
        if starting_ref != "unknown":
            _checkout(starting_ref)

    # Build blind-shuffled copies
    rng = random.Random(args.seed)
    blind_dir = out_root / "blind"
    blind_dir.mkdir(parents=True, exist_ok=True)
    blind_map: list[dict] = []
    shuffled = list(rendered)
    rng.shuffle(shuffled)
    for i, src in enumerate(shuffled):
        # Filename hides ref + prompt
        h = hashlib.sha1(str(src).encode()).hexdigest()[:8]
        dst = blind_dir / f"blind_{i:03d}_{h}.wav"
        shutil.copy2(src, dst)
        # Find this src in manifest_rows
        ref_dir_name = src.parent.name
        prompt_id = src.stem
        blind_map.append({
            "blind_filename": dst.name,
            "ref": ref_dir_name,
            "prompt_id": prompt_id,
        })

    # Write manifest + blind map
    manifest_csv = out_root / "manifest.csv"
    with open(manifest_csv, "w", newline="") as f:
        if manifest_rows:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)

    blind_csv = out_root / "blind_map.csv"
    with open(blind_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blind_filename", "ref", "prompt_id"])
        w.writeheader()
        w.writerows(blind_map)

    # Empty rankings template the listener fills
    ranking_csv = out_root / "rankings.csv"
    with open(ranking_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blind_filename", "rank_1_to_5",
                     "transitions_smooth_y_n", "vocal_collisions_y_n",
                     "energy_arc_matched_y_n", "free_text_notes"])
        for row in blind_map:
            w.writerow([row["blind_filename"], "", "", "", "", ""])

    print(f"\n=== done ===")
    print(f"manifest:   {manifest_csv}")
    print(f"blind dir:  {blind_dir}")
    print(f"blind map:  {blind_csv}  (KEEP HIDDEN until listener finishes)")
    print(f"rankings:   {ranking_csv}  (fill this in while listening)")
    print()
    print(f"Listener flow:")
    print(f"  1. Play files in {blind_dir.name}/ in any order")
    print(f"  2. Fill {ranking_csv.name}")
    print(f"  3. After done: join rankings + blind_map on blind_filename")


if __name__ == "__main__":
    main()
