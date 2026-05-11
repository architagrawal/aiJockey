#!/bin/bash
# KTO training convenience wrapper — fires when plan_stats.jsonl has enough rows.
#
# Usage: bash scripts/kto_train.sh [pq_threshold] [out_dir]
#
# KTO advantages over DPO for our regime:
#   - binary pass/fail signal from composite (PQ+CE)/2 vs threshold
#   - unlocks ALL render logs (not just delta-pair subset)
#   - data-efficient at N<200 (Kahneman-Tversky loss-aversion model)
set +e
cd /workspace

PQ_THRESH="${1:-7.0}"
OUT="${2:-/scratch/director_kto_lora}"
JSONL="/scratch/probes/plan_stats.jsonl"

if [ ! -f "$JSONL" ]; then
  echo "ERROR: $JSONL missing — run renders first"
  exit 1
fi

N=$(wc -l < "$JSONL")
if [ "$N" -lt 20 ]; then
  echo "ERROR: only $N rows; need >= 20 for KTO. Run more renders."
  exit 1
fi

echo "[kto] $N rows, pq_thresh=$PQ_THRESH, out=$OUT"

export AIJOCKEY_DIRECTOR_DPO_ENABLE=1
export HF_HOME=/scratch/hf_cache

/opt/venv/bin/python /workspace/scripts/dpo_director.py \
  --jsonl "$JSONL" \
  --base "Qwen/Qwen2.5-7B-Instruct" \
  --out "$OUT" \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 5e-6 \
  --beta 0.1 \
  --lora-r 16 \
  --loss-type kto \
  --pq-pass-thresh "$PQ_THRESH"
