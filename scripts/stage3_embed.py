"""S3 — embed + auto-caption.

Watch /scratch/cache/ for new analyzed clips. Build:
  /scratch/embed/clap.npy        N x 512 stacked CLAP vectors
  /scratch/embed/clap_index.json {clip_id: row_idx}
  /scratch/embed/captions.json   {clip_id: caption_text}

Captions via Qwen2-Audio (7B default; 72B if AIJOCKEY_CAPTION_72B=1 and
QLoRA-load OK). Index used by Style-RAG retrieval (Track 7) and FAISS
nearest-neighbor lookup at planner time.

Mutually exclusive with S5 self-play in 192GB-VRAM scheduling.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pipeline.common import scratch_dir, atomic_write


def _load_existing_index() -> tuple[np.ndarray | None, dict[str, int]]:
    embed_dir = scratch_dir('embed')
    idx_path = embed_dir / 'clap_index.json'
    vec_path = embed_dir / 'clap.npy'
    if not idx_path.exists() or not vec_path.exists():
        return None, {}
    try:
        idx = json.loads(idx_path.read_text())
        vecs = np.load(vec_path)
        if vecs.shape[0] != len(idx):
            return None, {}
        return vecs, {k: int(v) for k, v in idx.items()}
    except Exception:
        return None, {}


def _save_index(vecs: np.ndarray, idx: dict[str, int]) -> None:
    embed_dir = scratch_dir('embed')
    np.save(embed_dir / 'clap.npy', vecs)
    with atomic_write(embed_dir / 'clap_index.json') as f:
        json.dump(idx, f)


def _caption_clip(audio_path: str, model=None) -> str:
    """Auto-caption via Qwen2-Audio. Lazy-loaded; returns empty on failure."""
    if model is None:
        return ''
    try:
        return model.caption(audio_path)
    except Exception:
        return ''


def _maybe_load_caption_model():
    if os.getenv('AIJOCKEY_CAPTIONS', '1') == '0':
        return None
    try:
        from training.efficiency import qlora_quant_config, hf_attn_implementation
        from transformers import AutoProcessor, AutoModelForCausalLM
    except ImportError:
        return None
    name = os.getenv('AIJOCKEY_CAPTION_MODEL', 'Qwen/Qwen2-Audio-7B-Instruct')
    qcfg = qlora_quant_config() if os.getenv('AIJOCKEY_CAPTION_72B') == '1' else None
    try:
        proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=qcfg, device_map='auto',
            attn_implementation=hf_attn_implementation(),
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"warn: caption model load failed ({e})")
        return None

    class _Captioner:
        """Wraps Qwen2-Audio for clip-level captioning.

        Loads first 30s of audio, asks model: 'Describe this music clip
        for a DJ planning a mix. Mention genre, BPM band, mood, key
        instruments, and any vocals.'
        """

        _PROMPT = (
            "Describe this music clip for a DJ planning a mix. Mention "
            "genre, BPM band, mood, key instruments, and any vocals. "
            "Output 1 short sentence."
        )

        def caption(self, path: str) -> str:
            try:
                import torch
                import librosa
                wav, sr = librosa.load(path, sr=16000, mono=True, duration=30.0)
                conv = [
                    {"role": "user", "content": [
                        {"type": "audio", "audio_url": path},
                        {"type": "text",  "text": self._PROMPT},
                    ]},
                ]
                text = proc.apply_chat_template(conv, add_generation_prompt=True,
                                                  tokenize=False)
                inputs = proc(text=text, audios=[wav], return_tensors='pt',
                              padding=True, sampling_rate=sr)
                inputs = {k: v.to(mdl.device) if hasattr(v, 'to') else v
                          for k, v in inputs.items()}
                with torch.no_grad():
                    out_ids = mdl.generate(**inputs, max_new_tokens=80,
                                            do_sample=False)
                # Strip prompt prefix
                gen = out_ids[:, inputs['input_ids'].size(1):]
                resp = proc.batch_decode(gen, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
                return resp.strip()
            except Exception as e:
                print(f"warn: caption {path} failed ({e})")
                return ''
    return _Captioner()


def process_one(cache_path: Path, vecs: list[np.ndarray], idx: dict[str, int],
                captions: dict[str, str], cap_model) -> bool:
    meta = json.loads(cache_path.read_text())
    cid = meta.get('clip_id') or cache_path.stem
    if cid in idx:
        return False
    clap = meta.get('clap_embedding')
    if clap is None or len(clap) == 0:
        print(f"warn: no CLAP for {cid}; skipping embed")
        return False
    vec = np.asarray(clap, dtype=np.float32)
    if vec.ndim != 1:
        vec = vec.reshape(-1)
    idx[cid] = len(vecs)
    vecs.append(vec)
    if cap_model is not None and meta.get('audio_path'):
        captions[cid] = _caption_clip(meta['audio_path'], cap_model)
    return True


def watch_loop(cache_root: Path, interval: float = 60.0) -> None:
    print(f"S3 watching {cache_root} every {interval}s")
    existing_vec, idx = _load_existing_index()
    vecs: list[np.ndarray] = list(existing_vec) if existing_vec is not None else []
    cap_path = scratch_dir('embed') / 'captions.json'
    captions: dict[str, str] = {}
    if cap_path.exists():
        try:
            captions = json.loads(cap_path.read_text())
        except Exception:
            captions = {}
    cap_model = _maybe_load_caption_model()

    while True:
        wrote = False
        if cache_root.exists():
            for fp in sorted(cache_root.glob('*.json')):
                if process_one(fp, vecs, idx, captions, cap_model):
                    wrote = True
        if wrote and vecs:
            _save_index(np.stack(vecs).astype(np.float32), idx)
            with atomic_write(cap_path) as f:
                json.dump(captions, f, indent=2)
            print(f"S3 wrote {len(vecs)} embeddings, {len(captions)} captions")
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', default=str(scratch_dir('cache')))
    ap.add_argument('--interval', type=float, default=60.0)
    args = ap.parse_args()
    watch_loop(Path(args.watch), interval=args.interval)


if __name__ == '__main__':
    main()
