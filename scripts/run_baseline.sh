#!/usr/bin/env bash
# Cohort baseline runner — 4 cohorts × 10 prompts × 2 seeds = 80 rows.
# Cumulative ablation per teammate plan:
#
#   Cohort A:  baseline       (all improvers OFF)
#   Cohort B:  E              (energy-repick only)
#   Cohort C:  E + O          (energy + overlap-shorten)
#   Cohort D:  E + O + S      (all three)
#
# Each cohort writes ~20 rows to $AIJOCKEY_PROBE_LOG with improver_state
# fields stamped per row (probe_log captures env at log time). After all
# cohorts complete, run:
#   python -m probe_log summary --by_cohort
#
# Usage:
#   AIJOCKEY_PROBE_LOG=/scratch/probes/log_cohorts.jsonl \
#     bash scripts/run_baseline.sh /workspace/test_user_cache_v6
set -e
CACHE="${1:-${AIJOCKEY_BASELINE_CACHE:-/workspace/test_user_cache_v6}}"
OUT_BASE="${2:-/workspace/output/cohorts}"
SEEDS="${SEEDS:-2}"
DURATION="${DURATION:-180}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
SRC_DIR="${SRC_DIR:-src}"

# Reset log so the run produces a clean by_cohort summary.
LOG="${AIJOCKEY_PROBE_LOG:-/scratch/probes/log_cohorts.jsonl}"
mkdir -p "$(dirname "$LOG")"
> "$LOG"
# CRITICAL: export so child processes (baseline_renders.py → main.py →
# log_render) write to this path. Without export, log_render falls through
# to its default /scratch/probes/log.jsonl and the cohort log stays empty.
export AIJOCKEY_PROBE_LOG="$LOG"
echo "[cohorts] log -> $LOG (exported)"

# Common env (improver passes enabled — gated per cohort)
export AIJOCKEY_PHASE=1 AIJOCKEY_DTYPE=bfloat16 AIJOCKEY_PHRASE_QUANTIZE=1
export AIJOCKEY_STEM_SWAP=1 AIJOCKEY_CONSTITUTIONAL=1
export AIJOCKEY_INSTRUMENTAL_ONLY=1 AIJOCKEY_USE_DIRECTOR_LLM=1
export AIJOCKEY_RENDER_WORKERS=6 AIJOCKEY_STEM_WORKERS=8
export TRANSFORMERS_VERBOSITY=error

run_cohort() {
    local cohort="$1"
    local energy="$2"
    local overlap="$3"
    local swap="$4"
    local improve_passes="$5"
    local out_dir="$OUT_BASE/cohort_$cohort"
    mkdir -p "$out_dir"
    echo
    echo "=== COHORT $cohort  E=$energy O=$overlap S=$swap  passes=$improve_passes ==="
    AIJOCKEY_COHORT="$cohort" \
    AIJOCKEY_IMPROVER_ENERGY="$energy" \
    AIJOCKEY_IMPROVER_OVERLAP="$overlap" \
    AIJOCKEY_IMPROVER_SWAP="$swap" \
    python3 scripts/baseline_renders.py \
        --cache "$CACHE" \
        --out_dir "$out_dir" \
        --duration "$DURATION" \
        --seeds "$SEEDS" \
        ${PROMPTS_FILE:+--prompts_file "$PROMPTS_FILE"} \
        --use_director --apply_llm_tiers \
        --improve_max_passes "$improve_passes" \
        --src_dir "$SRC_DIR" 2>&1 | tail -30
}

# Cohort A: baseline. All improvers off; no improver passes (default 0).
run_cohort A 0 0 0 0
# Cohort B: energy_repick only, 1 pass.
run_cohort B 1 0 0 1
# Cohort C: energy + overlap.
run_cohort C 1 1 0 1
# Cohort D: all three.
run_cohort D 1 1 1 1

echo
echo "=== SUMMARY ==="
cd "$SRC_DIR"
python3 -m probe_log summary --by_cohort
