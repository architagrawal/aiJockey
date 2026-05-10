"""S7 — DPO/ORPO LoRA fine-tune of the Director.

Watch /scratch/preferences/ for new preference jsonl files. Once enough
pairs accumulate (>= MIN_PAIRS), run ORPO LoRA pass on Qwen2-Audio-7B.
Saves to /scratch/models/director_dpo_e{N}/ (PEFT adapter dir).

Phase A polish §14.2 P1 + §16.2 D-G (QLoRA + ORPO + Lion).
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
from pipeline.common import scratch_dir


MIN_PAIRS = 30
RETRAIN_INTERVAL_SEC = 3600


def _collect_pairs() -> list[dict]:
    pref_dir = scratch_dir('preferences')
    pairs: list[dict] = []
    for fp in sorted(pref_dir.glob('*.jsonl')):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    pairs.append(json.loads(line))
                except Exception:
                    continue
    return pairs


def _build_pairs_from_renders() -> int:
    """Convert /scratch/renders/{pid}/pair.json to /scratch/preferences/iter*.jsonl."""
    renders = scratch_dir('renders')
    pref_dir = scratch_dir('preferences')
    out = pref_dir / 'iter_auto.jsonl'
    new_pairs = []
    seen_ids: set[str] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                seen_ids.add(json.loads(line)['prompt_id'])
            except Exception:
                pass
    for pjson in renders.glob('*/pair.json'):
        try:
            p = json.loads(pjson.read_text())
        except Exception:
            continue
        if p.get('prompt_id') in seen_ids:
            continue
        new_pairs.append(p)
    if not new_pairs:
        return 0
    with open(out, 'a') as f:
        for p in new_pairs:
            f.write(json.dumps(p) + '\n')
    return len(new_pairs)


def run_dpo_train(epoch: int) -> bool:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from training.efficiency import (qlora_quant_config, lora_config,
                                         hf_attn_implementation, preference_trainer)
        from peft import get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        print(f"S7 deps missing ({e}); skip")
        return False

    pairs = _collect_pairs()
    if len(pairs) < MIN_PAIRS:
        print(f"S7 only {len(pairs)} pairs (< {MIN_PAIRS}); waiting")
        return False

    base = os.getenv('AIJOCKEY_DIRECTOR_BASE',
                     'Qwen/Qwen2-Audio-7B-Instruct')
    print(f"S7 ORPO pass on {base} with {len(pairs)} pairs (epoch {epoch})")

    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    qcfg = qlora_quant_config()
    mdl = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=qcfg, device_map='auto',
        attn_implementation=hf_attn_implementation(), trust_remote_code=True,
    )
    if qcfg is not None:
        mdl = prepare_model_for_kbit_training(mdl)
    mdl = get_peft_model(mdl, lora_config(r=16, alpha=32))

    method = os.getenv('AIJOCKEY_PREF_METHOD', 'orpo')
    Trainer = preference_trainer(method)

    # Build dataset
    try:
        from datasets import Dataset
    except ImportError:
        print("S7 datasets lib missing; skip")
        return False
    ds_rows = [{'prompt': p.get('prompt', ''),
                'chosen': p.get('chosen_text', p.get('chosen_path', '')),
                'rejected': p.get('rejected_text', p.get('rejected_path', ''))}
               for p in pairs]
    ds = Dataset.from_list(ds_rows)

    out_dir = scratch_dir('models') / f'director_dpo_e{epoch:03d}'
    try:
        from trl import ORPOConfig
        cfg = ORPOConfig(output_dir=str(out_dir),
                         per_device_train_batch_size=1,
                         gradient_accumulation_steps=8,
                         learning_rate=5e-5,
                         num_train_epochs=1,
                         logging_steps=10,
                         save_strategy='no',
                         bf16=True)
        trainer = Trainer(model=mdl, args=cfg, train_dataset=ds,
                          tokenizer=tok)
    except Exception as e:
        print(f"S7 trainer init failed ({e})")
        return False
    trainer.train()
    mdl.save_pretrained(str(out_dir))
    print(f"S7 saved {out_dir}")
    return True


def watch_loop(interval_sec: float) -> None:
    epoch = 0
    while True:
        added = _build_pairs_from_renders()
        if added:
            print(f"S7 ingested {added} new render pairs")
        ok = run_dpo_train(epoch + 1)
        if ok:
            epoch += 1
        time.sleep(interval_sec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', type=float, default=RETRAIN_INTERVAL_SEC)
    args = ap.parse_args()
    watch_loop(args.interval)


if __name__ == '__main__':
    main()
