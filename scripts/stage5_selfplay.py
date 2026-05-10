"""S5 — self-play render farm.

For each prompt in /scratch/prompts/list.json, render K candidates with
varying humanization seeds + Director temperatures. Score each with
critic v2. Top-1 + bottom-1 per prompt = a preference pair feeding S6.

K=8 default, scalable to K=32+ if VRAM permits.

Phase A polish §14.2 P1 (self-play data gen).
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pipeline.common import scratch_dir, atomic_write


PROMPTS_DEFAULT = scratch_dir('prompts') / 'list.json'
RENDERS_ROOT = lambda: scratch_dir('renders')


_CRITIC_CACHE = {'ckpt': None, 'model': None}
_CLAP_CACHE = {'model': None}


def _load_critic():
    """Lazy-load + cache CriticV2 for repeated scoring calls."""
    cps = sorted(scratch_dir('models').glob('critic_v2_e*.pt'))
    if not cps:
        return None
    latest = cps[-1]
    if _CRITIC_CACHE['ckpt'] == latest and _CRITIC_CACHE['model'] is not None:
        return _CRITIC_CACHE['model']
    try:
        import torch
        from stage4_critic import CriticV2
        state = torch.load(latest, map_location='cpu')
        m = CriticV2()
        m.load_state_dict(state['model'])
        m.eval()
        _CRITIC_CACHE['ckpt'] = latest
        _CRITIC_CACHE['model'] = m
        return m
    except Exception as e:
        print(f"warn: critic load failed ({e})")
        return None


def _load_clap():
    if _CLAP_CACHE['model'] is not None:
        return _CLAP_CACHE['model']
    try:
        from clap_wrapper import CLAP_Module
        m = CLAP_Module(enable_fusion=False)
        m.load_ckpt()
        _CLAP_CACHE['model'] = m
        return m
    except Exception as e:
        print(f"warn: CLAP load failed ({e})")
        return None


def _audio_to_clap_windows(audio_path: Path, n_windows: int = 5,
                            window_sec: float = 8.0) -> list:
    """Sample n equally-spaced windows of `audio_path`, embed each via CLAP.

    Returns list of 512-vec numpy arrays.
    """
    try:
        import numpy as np
        import torchaudio
        wav, sr = torchaudio.load(str(audio_path))
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        total = wav.size(1) / sr
        if total < window_sec:
            return []
        starts = [(total - window_sec) * i / max(1, n_windows - 1)
                  for i in range(n_windows)]
        clap = _load_clap()
        if clap is None:
            return []
        out = []
        for st in starts:
            s_idx = int(st * sr)
            e_idx = s_idx + int(window_sec * sr)
            seg = wav[:, s_idx:e_idx].numpy().astype(np.float32)
            try:
                emb = clap.get_audio_embedding_from_data(x=seg, use_tensor=False)
                out.append(np.asarray(emb).reshape(-1)[:512])
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"warn: clap windows failed for {audio_path} ({e})")
        return []


def _critic_score(audio_path: Path) -> float:
    """Score render via critic v2.

    Steps:
      1. Sample 5 CLAP windows from audio_path.
      2. Build pre/trans/post triplet by triple-stacking each window's CLAP.
      3. Run critic forward, sigmoid of `real` head.
      4. Average across windows.

    Returns 0.5 (uniform) when critic or CLAP missing — degrades gracefully
    so self-play loop still produces preference pairs without strong critic.
    """
    if not audio_path.exists():
        return 0.0
    try:
        import numpy as np
        import torch
    except ImportError:
        return random.random()
    critic = _load_critic()
    if critic is None:
        return 0.5
    embs = _audio_to_clap_windows(audio_path)
    if not embs:
        return 0.5
    feats = np.stack([np.concatenate([e, e, e]).astype(np.float32) for e in embs])
    with torch.no_grad():
        out = critic(torch.from_numpy(feats))
        probs = torch.sigmoid(out['real']).cpu().numpy()
    return float(np.mean(probs))


def render_candidate(prompt_id: str, seed: int, k_idx: int,
                     prompt_payload: dict) -> Path:
    """Run main.py end-to-end on prompt with given seed. Returns mix path."""
    out_dir = RENDERS_ROOT() / prompt_id / f'k{k_idx:03d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'mix.wav'
    if out_path.exists():
        return out_path

    env = os.environ.copy()
    env['AIJOCKEY_RENDER_SEED'] = str(seed)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')

    args = [
        sys.executable, '-m', 'main', 'all',
        '--clips', prompt_payload['clips'],
        '--cache', prompt_payload.get('cache', 'cache'),
        '--out', str(out_path),
        '--prompt', prompt_payload.get('prompt', ''),
        '--arc', prompt_payload.get('arc', 'build'),
        '--seed', str(seed),
    ]
    import subprocess
    try:
        subprocess.run(args, check=True, env=env, timeout=900)
    except Exception as e:
        print(f"warn: render failed for {prompt_id} k{k_idx}: {e}")
        return out_path
    return out_path


def process_prompt(prompt_id: str, payload: dict, k: int) -> dict:
    print(f"S5 prompt {prompt_id}: K={k}")
    scores: list[tuple[int, float, Path]] = []
    for ki in range(k):
        seed = abs(hash((prompt_id, ki))) % (2 ** 31)
        path = render_candidate(prompt_id, seed, ki, payload)
        if path.exists():
            score = _critic_score(path)
            scores.append((ki, score, path))
    scores.sort(key=lambda t: t[1])
    if not scores:
        return {'prompt_id': prompt_id, 'pairs': []}
    worst = scores[0]
    best = scores[-1]
    pair = {
        'prompt_id': prompt_id,
        'prompt': payload.get('prompt', ''),
        'chosen_path': str(best[2]),
        'rejected_path': str(worst[2]),
        'chosen_score': best[1],
        'rejected_score': worst[1],
        'k': len(scores),
    }
    out = RENDERS_ROOT() / prompt_id / 'pair.json'
    with atomic_write(out) as f:
        json.dump(pair, f, indent=2)
    return pair


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', default=str(PROMPTS_DEFAULT))
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--interval', type=float, default=300.0)
    args = ap.parse_args()
    while True:
        if Path(args.prompts).exists():
            try:
                prompts = json.loads(Path(args.prompts).read_text())
            except Exception as e:
                print(f"warn: failed to read prompts: {e}")
                prompts = []
            for entry in prompts:
                pid = entry.get('id') or entry.get('prompt_id')
                if not pid:
                    continue
                done_marker = RENDERS_ROOT() / pid / 'pair.json'
                if done_marker.exists():
                    continue
                process_prompt(pid, entry, args.k)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
