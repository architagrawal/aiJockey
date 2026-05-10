#!/usr/bin/env bash
# Locked-config demo render — deadline-fast variant of teammate's
# render_demo_set.sh. Skips genre presets requiring curated clip pools we
# don't have yet; just renders 5 variations on existing test_user_cache_v6
# with the configurations most likely to ship.
#
# Output: /workspace/demo_locked/demo_{1..5}.{wav,mp3}
# Probes auto-logged via cmd_execute → AIJOCKEY_PROBE_LOG.
#
# Usage:
#   bash scripts/render_demo_locked.sh
#
# Prereq: cohort run NOT in flight (uses GPU). Check with:
#   ps aux | grep -E 'baseline|run_baseline' | grep -v grep
set -e

CACHE="${1:-/workspace/test_user_cache_v6}"
OUT="${2:-/workspace/demo_locked}"
mkdir -p "$OUT"

# Locked config — all improvers ON, target severity ≤0.5.
export AIJOCKEY_PHASE=1 AIJOCKEY_DTYPE=bfloat16
export AIJOCKEY_PHRASE_QUANTIZE=1 AIJOCKEY_STEM_SWAP=1
export AIJOCKEY_CONSTITUTIONAL=1 AIJOCKEY_INSTRUMENTAL_ONLY=1
export AIJOCKEY_USE_DIRECTOR_LLM=1
export AIJOCKEY_RENDER_WORKERS=6 AIJOCKEY_STEM_WORKERS=8
export AIJOCKEY_IMPROVER_ENERGY=1 AIJOCKEY_IMPROVER_OVERLAP=1 AIJOCKEY_IMPROVER_SWAP=1
export AIJOCKEY_PROBE_LOG="${AIJOCKEY_PROBE_LOG:-/scratch/probes/log_demo.jsonl}"
export TRANSFORMERS_VERBOSITY=error
> "$AIJOCKEY_PROBE_LOG"

PROMPTS=(
    "warmup to peak hour, energetic build"
    "after-hours smoky lo-fi"
    "rolling tech-house groove"
    "deep house cooldown"
    "festival peak euphoric drops"
)

for i in "${!PROMPTS[@]}"; do
    n=$((i + 1))
    PROMPT="${PROMPTS[$i]}"
    TL="$OUT/demo_${n}_timeline.json"
    WAV="$OUT/demo_${n}.wav"
    MP3="$OUT/demo_${n}.mp3"
    echo
    echo "=== DEMO $n: $PROMPT ==="
    cd /workspace/src
    python3 main.py plan \
        --cache "$CACHE" --out "$TL" \
        --duration 180 --arc build \
        --use_director --apply_llm_tiers \
        --min_unique_clips 2 --max_clips 8 \
        --prompt "$PROMPT" 2>&1 | tail -3
    python3 main.py execute \
        --timeline "$TL" --cache "$CACHE" --out "$WAV" \
        --improve_max_passes 1 --improve_threshold 0.5 2>&1 | tail -5
    python3 main.py master \
        --in "$WAV" --out "$MP3" --lufs -9 2>&1 | tail -1
done

echo
echo "=== DEMO BATCH SUMMARY ==="
cd /workspace/src
python3 -m probe_log summary 2>&1 | tail -25
echo
echo "Files:"
ls -la "$OUT"/*.mp3 "$OUT"/*.wav 2>/dev/null
