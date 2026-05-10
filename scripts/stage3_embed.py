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

        Supports both single-clip (`caption`) and batched
        (`caption_batch`) inference. Batching shares one model.generate
        call across multiple audios — gives near-linear speedup until
        per-batch VRAM saturates (Qwen2-Audio 7B bf16 ≈ 16-32 clips/batch
        on MI300X 192GB).
        """

        _PROMPT = (
            "Describe this music clip for a DJ planning a mix. Mention "
            "genre, BPM band, mood, key instruments, and any vocals. "
            "Output 1 short sentence."
        )

        def _build_inputs(self, audios: list, paths: list[str]):
            convs = []
            for p in paths:
                convs.append([
                    {"role": "user", "content": [
                        {"type": "audio", "audio_url": p},
                        {"type": "text",  "text": self._PROMPT},
                    ]},
                ])
            texts = [proc.apply_chat_template(c, add_generation_prompt=True,
                                               tokenize=False) for c in convs]
            return proc(text=texts, audios=audios, return_tensors='pt',
                        padding=True, sampling_rate=16000)

        def caption(self, path: str) -> str:
            out = self.caption_batch([path])
            return out[0] if out else ''

        def caption_batch(self, paths: list[str]) -> list[str]:
            if not paths:
                return []
            try:
                import torch
                import librosa
            except ImportError:
                return ['' for _ in paths]
            audios: list = []
            kept_paths: list[str] = []
            kept_idx: list[int] = []
            for i, p in enumerate(paths):
                try:
                    wav, _sr = librosa.load(p, sr=16000, mono=True, duration=30.0)
                    audios.append(wav)
                    kept_paths.append(p)
                    kept_idx.append(i)
                except Exception as e:
                    print(f"warn: caption load {p} failed ({e})")
            results = ['' for _ in paths]
            if not audios:
                return results
            try:
                inputs = self._build_inputs(audios, kept_paths)
                inputs = {k: v.to(mdl.device) if hasattr(v, 'to') else v
                          for k, v in inputs.items()}
                with torch.inference_mode():
                    out_ids = mdl.generate(**inputs, max_new_tokens=80,
                                            do_sample=False)
                # Per-row prompt-prefix length differs when padding=True;
                # use attention_mask sums to find each row's input length.
                input_ids = inputs['input_ids']
                in_lens = inputs.get('attention_mask',
                                      torch.ones_like(input_ids)).sum(dim=1)
                for row, (orig_i, in_len) in enumerate(zip(kept_idx, in_lens)):
                    gen = out_ids[row, int(in_len):]
                    resp = proc.tokenizer.decode(
                        gen, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False).strip()
                    results[orig_i] = resp
            except Exception as e:
                print(f"warn: caption_batch failed ({e}); per-clip fallback")
                # Best-effort per-clip retry so a single bad sample
                # doesn't poison the whole batch.
                for orig_i, p in zip(kept_idx, kept_paths):
                    try:
                        results[orig_i] = self._caption_one_safe(p)
                    except Exception:
                        results[orig_i] = ''
            return results

        def _caption_one_safe(self, path: str) -> str:
            import torch
            import librosa
            wav, _sr = librosa.load(path, sr=16000, mono=True, duration=30.0)
            inputs = self._build_inputs([wav], [path])
            inputs = {k: v.to(mdl.device) if hasattr(v, 'to') else v
                      for k, v in inputs.items()}
            with torch.inference_mode():
                out_ids = mdl.generate(**inputs, max_new_tokens=80,
                                        do_sample=False)
            gen = out_ids[:, inputs['input_ids'].size(1):]
            return proc.batch_decode(gen, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()
    return _Captioner()


def process_one(cache_path: Path, vecs: list[np.ndarray], idx: dict[str, int],
                ) -> tuple[str | None, str | None]:
    """Index one clip. Returns (clip_id, audio_path) for deferred batched
    captioning, or (None, None) if the clip was already indexed or had no
    CLAP vector to add.
    """
    meta = json.loads(cache_path.read_text())
    cid = meta.get('clip_id') or cache_path.stem
    if cid in idx:
        return None, None
    clap = meta.get('clap_embedding')
    if clap is None or len(clap) == 0:
        print(f"warn: no CLAP for {cid}; skipping embed")
        return None, None
    vec = np.asarray(clap, dtype=np.float32)
    if vec.ndim != 1:
        vec = vec.reshape(-1)
    idx[cid] = len(vecs)
    vecs.append(vec)
    return cid, meta.get('audio_path')


def watch_loop(cache_root: Path, interval: float = 60.0,
               caption_batch_size: int = 8) -> None:
    print(f"S3 watching {cache_root} every {interval}s "
          f"(caption_batch_size={caption_batch_size})")
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
        # Phase 1: index every new CLAP vector immediately (cheap).
        # Defer captioning into batches — captioning is the dominant cost.
        pending_caption: list[tuple[str, str]] = []
        if cache_root.exists():
            for fp in sorted(cache_root.glob('*.json')):
                cid, audio_path = process_one(fp, vecs, idx)
                if cid is None:
                    continue
                wrote = True
                if cap_model is not None and audio_path:
                    pending_caption.append((cid, audio_path))

        # Phase 2: batched captioning. Single GPU forward across N clips
        # is dramatically faster than N separate forwards on Qwen2-Audio.
        for batch_start in range(0, len(pending_caption), caption_batch_size):
            chunk = pending_caption[batch_start:batch_start + caption_batch_size]
            paths = [p for _cid, p in chunk]
            try:
                texts = cap_model.caption_batch(paths)
            except Exception as e:
                print(f"warn: caption_batch err ({e}); per-clip fallback")
                texts = [_caption_clip(p, cap_model) for p in paths]
            for (cid, _p), t in zip(chunk, texts):
                if t:
                    captions[cid] = t

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
    ap.add_argument('--caption-batch-size', type=int,
                    default=int(os.environ.get('AIJOCKEY_CAPTION_BATCH', '8')))
    args = ap.parse_args()
    watch_loop(Path(args.watch), interval=args.interval,
               caption_batch_size=args.caption_batch_size)


if __name__ == '__main__':
    main()
