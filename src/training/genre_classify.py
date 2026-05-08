"""
Zero-shot genre classification via CLAP text-audio similarity.

No training needed — uses CLAP's pretrained audio-text alignment to score
audio against text prompts like "this is house music".

Output: tags each clip in cache/<id>.json with detected genre + confidence.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np


GENRE_PROMPTS = {
    'edm':            'big room electronic dance music with heavy drops',
    'house':          'four-on-the-floor house music with deep bass',
    'tech_house':     'tech house with rolling basslines and percussion',
    'deep_house':     'deep house with smooth chords and warm pads',
    'techno':         'driving techno music with hypnotic loops',
    'melodic_techno': 'melodic techno with emotional synths',
    'trance':         'uplifting trance music with euphoric leads',
    'progressive':    'progressive house with extended buildups',
    'dnb':            'drum and bass at 174 bpm with breakbeats',
    'dubstep':        'dubstep music with wobble bass and heavy drops',
    'trap':           'trap music with 808 bass and hi-hat rolls',
    'future_bass':    'future bass with chopped vocals and sidechain',
    'lofi':           'lofi hip hop chill beats',
    'disco':          'disco music with funky basslines and strings',
    'ambient':        'ambient electronic atmospheric music',
}


def classify_clip(clap_emb: np.ndarray, text_embs: dict[str, np.ndarray]
                  ) -> tuple[str, float, dict[str, float]]:
    """
    Returns (best_genre, confidence, all_scores).
    clap_emb: (512,) audio embedding from CLAP
    text_embs: {genre: (512,) text embedding from CLAP}
    """
    if clap_emb.ndim != 1 or clap_emb.size == 0:
        return ('unknown', 0.0, {})
    a = clap_emb / (np.linalg.norm(clap_emb) + 1e-8)
    scores = {}
    for genre, t_emb in text_embs.items():
        b = t_emb / (np.linalg.norm(t_emb) + 1e-8)
        scores[genre] = float(np.dot(a, b))
    best = max(scores, key=scores.get)
    return (best, scores[best], scores)


def get_text_embeddings(prompts: dict[str, str] = GENRE_PROMPTS,
                        device: str = 'auto') -> dict[str, np.ndarray]:
    """Compute CLAP text embeddings for genre prompts (one-time)."""
    from clap_wrapper import load_clap, _BACKEND, _MODEL, _PROCESSOR
    load_clap()
    import torch
    text_embs: dict[str, np.ndarray] = {}

    # transformers backend
    from clap_wrapper import _MODEL as M, _PROCESSOR as P, _BACKEND as B
    if B == 'transformers':
        for genre, text in prompts.items():
            inputs = P(text=text, return_tensors='pt')
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                out = M.get_text_features(**inputs)
            from clap_wrapper import _extract_embedding
            emb = _extract_embedding(out)
            text_embs[genre] = emb.cpu().numpy()[0].astype(np.float32)
    else:
        # laion-clap backend
        texts = list(prompts.values())
        embs = M.get_text_embedding(texts, use_tensor=False)
        for i, genre in enumerate(prompts.keys()):
            text_embs[genre] = embs[i].astype(np.float32)
    return text_embs


def classify_pool(cache_dir: str, write_back: bool = True) -> dict[str, dict]:
    """Classify all cached clips. Returns {clip_id: {genre, conf, scores}}."""
    cache = Path(cache_dir)
    text_embs = get_text_embeddings()
    print(f"computed {len(text_embs)} genre prompt embeddings")
    results: dict[str, dict] = {}
    for jp in sorted(cache.glob('*.json')):
        npz_path = cache / f"{jp.stem}.npz"
        if not npz_path.exists():
            continue
        clap_emb = np.load(str(npz_path))['clap'].astype(np.float32)
        genre, conf, scores = classify_clip(clap_emb, text_embs)
        results[jp.stem] = {'genre': genre, 'confidence': conf, 'scores': scores}
        print(f"  {jp.stem[:50]:50s} -> {genre:18s} (conf={conf:.3f})")
        if write_back:
            with open(jp) as f:
                meta = json.load(f)
            meta['genre'] = genre
            meta['genre_confidence'] = conf
            with open(jp, 'w') as f:
                json.dump(meta, f, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='cache')
    ap.add_argument('--no_write', action='store_true',
                    help='do not write genre back into cached JSONs')
    args = ap.parse_args()
    classify_pool(args.cache, write_back=not args.no_write)


if __name__ == '__main__':
    main()
