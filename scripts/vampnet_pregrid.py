"""Pre-generate VampNet bridges for adjacent library clip pairs.

For each clip A in /cache (with audio present), pick K nearest clips
by CLAP cosine and synthesize a `gap_seconds` bridge from A → B. The
resulting bridges become cached library clips with `bridge__` prefix.

Run on droplet:
    export AIJOCKEY_VAMPNET_ENABLE=1 AIJOCKEY_AUDIOBOX_AESTHETICS=1
    /opt/venv/bin/python scripts/vampnet_pregrid.py \\
        --out /cache/vampnet_bridges --k 3 --gap 8.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_clips_with_audio(cache: Path) -> list[dict]:
    rows = []
    for jp in sorted(cache.glob("*.json")):
        if jp.name.endswith(".audiobox_slices.json") or \
           jp.name.endswith(".mert_pred.json"):
            continue
        try:
            meta = json.loads(jp.read_text())
        except Exception:
            continue
        ap = meta.get("path")
        if not ap or not Path(ap).exists():
            continue
        # Load CLAP from npz sidecar
        import numpy as np
        npz_p = cache / f"{jp.stem}.npz"
        if not npz_p.exists():
            continue
        try:
            npz = np.load(npz_p)
            clap = np.asarray(npz.get("clap"), dtype=np.float32)
        except Exception:
            continue
        if clap is None or clap.size == 0:
            continue
        rows.append({"id": jp.stem, "path": ap, "clap": clap,
                      "tempo": meta.get("tempo", 0.0),
                      "key": meta.get("key", "?")})
    return rows


def _topk_neighbors(rows: list[dict], k: int) -> dict[str, list[int]]:
    """Return {idx: [idx_b, idx_c, ...]} of top-k CLAP-neighbors."""
    import numpy as np
    claps = np.stack([r["clap"] for r in rows], axis=0).astype(np.float32)
    norms = np.linalg.norm(claps, axis=1, keepdims=True) + 1e-9
    unit = claps / norms
    sim = unit @ unit.T  # (n, n)
    np.fill_diagonal(sim, -1.0)
    out = {}
    for i, r in enumerate(rows):
        top_idx = np.argsort(-sim[i])[:k].tolist()
        out[i] = top_idx
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--context", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--min-pq", type=float, default=6.0)
    ap.add_argument("--limit-clips", type=int, default=0)
    args = ap.parse_args()

    from vampnet_wrapper import enabled, generate_bridge
    if not enabled():
        sys.exit("AIJOCKEY_VAMPNET_ENABLE != 1; aborting")
    try:
        from audiobox_critic import enabled as aae_en, score as aae_score
    except Exception:
        aae_en = lambda: False
        aae_score = lambda _p: None

    cache = Path(args.cache)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rejected = out / "_rejected"; rejected.mkdir(parents=True, exist_ok=True)

    print("[vampnet_pregrid] loading clips with audio...")
    rows = _load_clips_with_audio(cache)
    if args.limit_clips:
        rows = rows[: args.limit_clips]
    print(f"[vampnet_pregrid] {len(rows)} clips")
    neighbors = _topk_neighbors(rows, args.k)
    pairs = [(i, j) for i, js in neighbors.items() for j in js]
    print(f"[vampnet_pregrid] {len(pairs)} A→B pairs")

    manifest = []
    t_all = time.perf_counter()
    for n, (i, j) in enumerate(pairs):
        a, b = rows[i], rows[j]
        slug = f"vamp_bridge_{a['id'][:24]}__to__{b['id'][:24]}"
        target = out / f"{slug}.wav"
        if target.exists():
            print(f"[{n+1}/{len(pairs)}] skip {slug}")
            continue
        t0 = time.perf_counter()
        res = generate_bridge(
            a["path"], b["path"], args.gap, target,
            context_seconds=args.context,
            temperature=args.temperature, top_p=args.top_p,
        )
        dt = time.perf_counter() - t0
        if not res:
            print(f"[{n+1}/{len(pairs)}] FAILED {slug}")
            continue
        scores = aae_score(target) if aae_en() else None
        pq = float((scores or {}).get("PQ", 0.0))
        entry = {"a_id": a["id"], "b_id": b["id"], "slug": slug,
                  "gap_s": args.gap, "audiobox": scores,
                  "gen_seconds": round(dt, 2)}
        if scores and pq < args.min_pq:
            new = rejected / target.name
            target.rename(new)
            entry["path"] = str(new); entry["rejected"] = True
            print(f"[{n+1}/{len(pairs)}] reject {slug} PQ={pq:.2f}")
        else:
            entry["path"] = str(target); entry["rejected"] = False
            print(f"[{n+1}/{len(pairs)}] keep {slug} PQ={pq:.2f} ({dt:.1f}s)")
        manifest.append(entry)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    kept = sum(1 for m in manifest if not m["rejected"])
    print(f"[vampnet_pregrid] {kept}/{len(manifest)} kept in "
          f"{(time.perf_counter()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
