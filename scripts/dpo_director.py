"""DPO LoRA fine-tune of Director LLM (Qwen2.5-7B) on logged renders.

Pairs better/worse plans by Audiobox composite (PQ + CE)/2 delta over
shared prompts. Trains a LoRA adapter via TRL's DPOTrainer.

Strategy (MR-FlowDPO-inspired, arXiv 2512.10264):
    - Group render rows by (prompt, pool_fingerprint).
    - Within group, form (chosen, rejected) pairs with composite delta > THR.
    - Format chat: system=Director system_prompt, user=task,
      assistant=director_json. chosen vs rejected.

Run on MI300X (MI300X 192 GB easily handles 7B + reference):
    export AIJOCKEY_DIRECTOR_DPO_ENABLE=1
    /opt/venv/bin/python scripts/dpo_director.py \\
        --jsonl /scratch/probes/log.jsonl \\
        --base Qwen/Qwen2.5-7B-Instruct \\
        --out /scratch/director_dpo_lora \\
        --epochs 1 --batch-size 1 --grad-accum 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def _composite(aae: dict | None) -> float:
    if not aae:
        return 0.0
    return (float(aae.get("PQ", 0.0)) + float(aae.get("CE", 0.0))) / 2.0


def _build_pairs(jsonl_path: str, min_delta: float = 0.25) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            aae = r.get("audiobox") or {}
            plan = r.get("director_plan") or r.get("plan") or r.get("director")
            prompt = r.get("prompt") or r.get("user_prompt") or ""
            pool = r.get("pool_fingerprint") or r.get("pool")
            if not (aae and plan and prompt):
                continue
            key = f"{prompt}::{pool}"
            groups[key].append({
                "prompt": prompt,
                "plan": plan,
                "composite": _composite(aae),
            })
    pairs = []
    for key, items in groups.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: -x["composite"])
        # Form top-vs-bottom pairs (composite delta > min_delta)
        for i in range(len(items) - 1):
            for j in range(len(items) - 1, i, -1):
                d = items[i]["composite"] - items[j]["composite"]
                if d < min_delta:
                    continue
                pairs.append({
                    "prompt": items[i]["prompt"],
                    "chosen": items[i]["plan"],
                    "rejected": items[j]["plan"],
                    "delta": d,
                })
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="/scratch/director_dpo_lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--min-delta", type=float, default=0.25)
    ap.add_argument("--max-pairs", type=int, default=2000)
    args = ap.parse_args()

    if os.environ.get("AIJOCKEY_DIRECTOR_DPO_ENABLE", "0") != "1":
        sys.exit("AIJOCKEY_DIRECTOR_DPO_ENABLE != 1; aborting")

    pairs = _build_pairs(args.jsonl, min_delta=args.min_delta)
    if not pairs:
        sys.exit(f"[dpo_director] no preference pairs (min_delta={args.min_delta})")
    pairs = pairs[: args.max_pairs]
    print(f"[dpo_director] {len(pairs)} pairs from {args.jsonl}")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOTrainer, DPOConfig
    except Exception as e:
        sys.exit(f"[dpo_director] install trl + peft + datasets first: {e}")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    def _fmt(plan):
        return plan if isinstance(plan, str) else json.dumps(plan, indent=2)

    ds = Dataset.from_list([
        {"prompt": p["prompt"],
         "chosen": _fmt(p["chosen"]),
         "rejected": _fmt(p["rejected"])}
        for p in pairs
    ])

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    cfg = DPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        beta=args.beta,
        bf16=True,
        save_strategy="epoch",
        logging_steps=10,
        warmup_ratio=0.03,
    )
    trainer = DPOTrainer(
        model=model, args=cfg, peft_config=lora,
        train_dataset=ds, tokenizer=tok, max_length=2048,
        max_prompt_length=1536,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[dpo_director] LoRA saved at {args.out}")


if __name__ == "__main__":
    main()
