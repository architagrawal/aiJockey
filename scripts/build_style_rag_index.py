"""Build Style-RAG embed index from any cache directory.

Lightweight alternative to full pipeline `scripts/stage3_embed.py` — just
reads existing `<clip>.json` + `<clip>.npz` (containing `clap` field) from
a cache dir, optionally captions via Qwen2-Audio if available, and writes
the three files Director's `style_rag.few_shot_block_for_director` reads:

  /scratch/embed/clap.npy           (N, 512) float32 stack
  /scratch/embed/clap_index.json    {clip_id: row_index}
  /scratch/embed/captions.json      {clip_id: caption_text}

Without captions Director still gets retrieval but the few-shot block
falls back to the clip_id as label. With captions Director sees real
descriptions ("uplifting techno, 128 BPM, female vocals from 0:30").

Usage:
    python scripts/build_style_rag_index.py --cache /cache --out /scratch/embed
    # add --captions to run Qwen2-Audio (slow, GPU)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def build_index(cache_dir: Path, out_dir: Path,
                 captions: bool = False,
                 caption_window_s: float = 30.0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_paths = sorted(cache_dir.glob('*.json'))
    embs: list[np.ndarray] = []
    ids: list[str] = []
    audio_paths: list[str] = []
    for jp in json_paths:
        cid = jp.stem
        npz_p = jp.with_suffix('.npz')
        if not npz_p.exists():
            continue
        try:
            with np.load(npz_p) as d:
                if 'clap' not in d:
                    continue
                v = np.asarray(d['clap'], dtype=np.float32).reshape(-1)
            if v.size == 0:
                continue
            embs.append(v)
            ids.append(cid)
            try:
                meta = json.loads(jp.read_text())
                audio_paths.append(meta.get('path') or meta.get('audio_path') or '')
            except Exception:
                audio_paths.append('')
        except Exception as e:
            print(f"warn: skip {cid} ({e})")

    if not embs:
        print(f"no clips with CLAP embeddings in {cache_dir}")
        return {}

    stack = np.stack(embs).astype(np.float32)
    np.save(str(out_dir / 'clap.npy'), stack)
    idx = {cid: i for i, cid in enumerate(ids)}
    (out_dir / 'clap_index.json').write_text(json.dumps(idx, indent=2))
    print(f"wrote {len(ids)} embeddings → {out_dir}/clap.npy "
          f"({stack.nbytes // 1024} KB)")

    cap_map: dict[str, str] = {}
    cap_path = out_dir / 'captions.json'
    if cap_path.exists():
        try:
            cap_map = json.loads(cap_path.read_text())
        except Exception:
            cap_map = {}

    if captions:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
            import torch
            from transformers import (Qwen2AudioForConditionalGeneration,
                                       AutoProcessor)
            import librosa
            print("loading Qwen2-Audio for caption generation...")
            model_id = os.environ.get('AIJOCKEY_CAPTION_MODEL',
                                       'Qwen/Qwen2-Audio-7B-Instruct')
            proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            dtype = (torch.bfloat16 if torch.cuda.is_available()
                     and torch.cuda.is_bf16_supported() else torch.float16)
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype,
                device_map='auto' if torch.cuda.is_available() else None,
                trust_remote_code=True)
            for cid, ap in zip(ids, audio_paths):
                if cid in cap_map and cap_map[cid]:
                    continue
                if not ap or not Path(ap).exists():
                    continue
                try:
                    y, _ = librosa.load(ap, sr=16000, mono=True,
                                         duration=caption_window_s)
                    conv = [
                        {'role': 'system', 'content':
                         'Caption this music clip in one short sentence: '
                         'genre, tempo feel, vocal/instrumental, mood.'},
                        {'role': 'user', 'content': [
                            {'type': 'audio', 'audio_url': cid},
                            {'type': 'text', 'text': 'Caption:'},
                        ]},
                    ]
                    text = proc.apply_chat_template(
                        conv, add_generation_prompt=True, tokenize=False)
                    inputs = proc(text=text, audios=[y],
                                   return_tensors='pt', padding=True,
                                   sampling_rate=16000)
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() if hasattr(v, 'cuda') else v
                                  for k, v in inputs.items()}
                    with torch.inference_mode():
                        gen = model.generate(
                            **inputs, max_new_tokens=80, do_sample=False,
                            pad_token_id=proc.tokenizer.pad_token_id
                                          or proc.tokenizer.eos_token_id)
                    new_tokens = gen[0, inputs['input_ids'].shape[1]:]
                    cap = proc.tokenizer.decode(
                        new_tokens, skip_special_tokens=True).strip()
                    cap_map[cid] = cap[:240]
                    print(f"  {cid[:40]}: {cap[:80]}")
                except Exception as e:
                    print(f"  caption fail {cid}: {e}")
        except Exception as e:
            print(f"caption pipeline unavailable ({e}); skipping captions")

    if cap_map:
        cap_path.write_text(json.dumps(cap_map, indent=2))
        print(f"wrote {len(cap_map)} captions → {cap_path}")
    elif not cap_path.exists():
        cap_path.write_text(json.dumps({}, indent=2))
        print(f"wrote empty captions stub → {cap_path}")

    return {'n_clips': len(ids),
            'embed_path': str(out_dir / 'clap.npy'),
            'idx_path': str(out_dir / 'clap_index.json'),
            'cap_path': str(cap_path),
            'n_captions': len(cap_map)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True,
                    help='clip cache dir with .json + .npz files')
    ap.add_argument('--out', default='/scratch/embed',
                    help='output dir for clap.npy / clap_index.json / captions.json')
    ap.add_argument('--captions', action='store_true',
                    help='generate captions via Qwen2-Audio (slow, GPU)')
    args = ap.parse_args()
    out = build_index(Path(args.cache), Path(args.out),
                       captions=args.captions)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
