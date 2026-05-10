"""Pre-fetch all HF + Demucs models needed by AiJockey.

Run after resuming a slim droplet snapshot (HF caches deleted to save snapshot size).
Takes ~3-8 min on a fast MI300X connection (htdemucs_ft is bag-of-models, ~640 MB).

Usage:
    python scripts/prefetch_models.py [--skip-qwen-audio] [--skip-qwen-text]
                                      [--skip-beat-this] [--demucs-model NAME]
"""
from __future__ import annotations
import argparse, os, sys, time


def fetch_demucs(name: str = "htdemucs_ft") -> None:
    print(f"[1/5] Demucs {name} ...", flush=True)
    t = time.time()
    from demucs.pretrained import get_model
    get_model(name)
    print(f"  ok in {time.time()-t:.1f}s")


def fetch_clap() -> None:
    print("[2/5] CLAP (laion/clap-htsat-unfused) ...", flush=True)
    t = time.time()
    from transformers import ClapModel, ClapProcessor
    ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    ClapModel.from_pretrained("laion/clap-htsat-unfused")
    print(f"  ok in {time.time()-t:.1f}s")


def fetch_qwen_text() -> None:
    print("[3/5] Qwen2.5-7B-Instruct (text Director fallback) ...", flush=True)
    t = time.time()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    print(f"  ok in {time.time()-t:.1f}s")


def fetch_qwen_audio() -> None:
    print("[4/4] Qwen2-Audio-7B-Instruct (multimodal Director) ...", flush=True)
    t = time.time()
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True)
    Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    print(f"  ok in {time.time()-t:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-qwen-audio", action="store_true",
                    help="skip Qwen2-Audio (16 GB)")
    ap.add_argument("--skip-qwen-text", action="store_true",
                    help="skip Qwen2.5-7B (14 GB)")
    args = ap.parse_args()

    t0 = time.time()
    fetch_demucs()
    fetch_clap()
    if not args.skip_qwen_text:
        fetch_qwen_text()
    if not args.skip_qwen_audio:
        fetch_qwen_audio()
    print(f"\nAll models fetched in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
