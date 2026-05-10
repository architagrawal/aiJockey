"""S9 — final batch render.

For each curated prompt in /scratch/prompts/final.json:
  - Load latest critic v2, latest director_dpo, latest bridge model
  - Render K=64 candidates (parallel where VRAM permits)
  - Critic-rank all candidates
  - Save top-3 to /scratch/output/{prompt_id}/{rank}.mp3 + metadata card

Phase A polish §13.8 + §14.8 deliverable spec.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pipeline.common import scratch_dir, atomic_write


PROMPTS_DEFAULT = scratch_dir('prompts') / 'final.json'
OUTPUT_DIR = lambda: scratch_dir('output')


def _latest_director() -> Path | None:
    cps = sorted(scratch_dir('models').glob('director_dpo_e*'))
    return cps[-1] if cps else None


def _latest_critic() -> Path | None:
    cps = sorted(scratch_dir('models').glob('critic_v2_e*.pt'))
    return cps[-1] if cps else None


def _latest_bridge() -> Path | None:
    cps = sorted(scratch_dir('models').glob('bridge_musicgen_e*'))
    return cps[-1] if cps else None


def render_one(prompt_id: str, payload: dict, k: int) -> dict:
    out_dir = OUTPUT_DIR() / prompt_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / 'card.json').exists():
        print(f"S9 {prompt_id} already done")
        return json.loads((out_dir / 'card.json').read_text())

    # Reuse S5 self-play machinery for the K-rendering loop.
    from stage5_selfplay import render_candidate, _critic_score
    scores: list[tuple[int, float, Path]] = []
    print(f"S9 final {prompt_id}: K={k}")
    for ki in range(k):
        seed = abs(hash((prompt_id, 'final', ki))) % (2 ** 31)
        path = render_candidate(prompt_id, seed, ki, payload)
        if path.exists():
            score = _critic_score(path)
            scores.append((ki, score, path))
    scores.sort(key=lambda t: t[1], reverse=True)
    top3 = scores[:3]

    import shutil
    metadata = {
        'prompt_id': prompt_id,
        'prompt': payload.get('prompt', ''),
        'arc': payload.get('arc', 'build'),
        'k': len(scores),
        'top3': [],
        'director_checkpoint': str(_latest_director() or ''),
        'critic_checkpoint': str(_latest_critic() or ''),
        'bridge_checkpoint': str(_latest_bridge() or ''),
    }
    for rank, (ki, score, path) in enumerate(top3, 1):
        dst = out_dir / f'rank{rank}.wav'
        try:
            shutil.copy2(path, dst)
        except Exception as e:
            print(f"warn: copy failed ({e})")
            continue
        metadata['top3'].append({
            'rank': rank, 'k_index': ki, 'critic_score': score,
            'path': str(dst),
        })
    with atomic_write(out_dir / 'card.json') as f:
        json.dump(metadata, f, indent=2)
    print(f"S9 wrote {out_dir} top3")
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', default=str(PROMPTS_DEFAULT))
    ap.add_argument('--k', type=int, default=int(os.getenv('AIJOCKEY_K', '64')))
    args = ap.parse_args()
    if not Path(args.prompts).exists():
        # Auto-bootstrap from the seed prompts shipped in repo. Lets users
        # run the script immediately without curating prompts manually.
        repo_seed = Path(__file__).resolve().parents[1] / 'scripts' / 'prompts' / 'final.json'
        if repo_seed.exists():
            print(f"S9 prompts file missing at {args.prompts}; "
                  f"copying repo seed {repo_seed}")
            Path(args.prompts).parent.mkdir(parents=True, exist_ok=True)
            Path(args.prompts).write_bytes(repo_seed.read_bytes())
        else:
            print(f"S9 no prompts at {args.prompts}; nothing to do")
            return
    prompts = json.loads(Path(args.prompts).read_text())
    for entry in prompts:
        pid = entry.get('id') or entry.get('prompt_id')
        if not pid:
            continue
        render_one(pid, entry, args.k)


if __name__ == '__main__':
    main()
