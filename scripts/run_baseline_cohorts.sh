#!/usr/bin/env bash
# Cumulative-ablation baseline for the probe → improver loop.
#
# Renders 4 cohorts × 10 prompts × 2 seeds = 80 mixes, each with a different
# improver-knob configuration. Probe log captures severity per render. Use
# `python -m probe_log summary --by-cohort` after to compare distributions.
#
# Cohorts:
#   A: ALL improvers OFF       (baseline)
#   B: ENERGY only ON
#   C: ENERGY + OVERLAP ON
#   D: ENERGY + OVERLAP + SWAP ON
#
# Cumulative (B → C → D adds one knob each) tells you incremental
# contribution. To get per-improver isolated effects, run a follow-up
# round with each knob solo.
#
# Each render also logs the active env knob state into the JSONL row
# (per probe_log.py — verify field includes `improver_state`).
#
# Estimated wall time on MI300X with bf16 + sdpa: ~3-4 hr.
# Disable Beat-This! per cohort if you want a non-BT control.
#
# Usage:
#   bash scripts/run_baseline_cohorts.sh \\
#       --user-pool clips_test \\
#       --duration 120 \\
#       --mix-mode balanced \\
#       --prompts scripts/prompts/listen_test.json \\
#       --seeds 2

set -e

USER_POOL="clips_test"
DURATION=120
MIX_MODE="balanced"
PROMPTS="scripts/prompts/listen_test.json"
SEEDS=2
PROBE_LOG_DIR="${AIJOCKEY_PROBE_LOG_DIR:-/scratch/probes}"
RUN_ID="cohort_$(date +%Y%m%d_%H%M%S)"

while [ $# -gt 0 ]; do
    case "$1" in
        --user-pool)    USER_POOL=$2; shift 2 ;;
        --duration)     DURATION=$2; shift 2 ;;
        --mix-mode)     MIX_MODE=$2; shift 2 ;;
        --prompts)      PROMPTS=$2; shift 2 ;;
        --seeds)        SEEDS=$2; shift 2 ;;
        --run-id)       RUN_ID=$2; shift 2 ;;
        *) echo "unknown arg $1"; exit 1 ;;
    esac
done

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
mkdir -p "$PROBE_LOG_DIR"

if [ ! -f "$PROMPTS" ]; then
    echo "no prompt file at $PROMPTS — generating default"
    mkdir -p "$(dirname "$PROMPTS")"
    cat > "$PROMPTS" <<'JSON'
[
  {"id": "lt01", "prompt": "warmup deep house slow burn", "arc": "build"},
  {"id": "lt02", "prompt": "festival peak euphoric drops", "arc": "peak"},
  {"id": "lt03", "prompt": "after-hours techno hypnotic", "arc": "flat_low"},
  {"id": "lt04", "prompt": "melodic house emotional sunset", "arc": "build"},
  {"id": "lt05", "prompt": "trance peak time euphoria", "arc": "peak"},
  {"id": "lt06", "prompt": "tech-house rolling groove", "arc": "build"},
  {"id": "lt07", "prompt": "drum and bass liquid roller", "arc": "build"},
  {"id": "lt08", "prompt": "minimal techno late night", "arc": "flat_low"},
  {"id": "lt09", "prompt": "trap heavy 808 menacing", "arc": "peak"},
  {"id": "lt10", "prompt": "lofi hip hop chill evening", "arc": "flat_low"}
]
JSON
fi

# Cohort definitions: name + improver knob env exports.
declare -a COHORTS=(
    "A:OFF:AIJOCKEY_IMPROVER_ENERGY=0 AIJOCKEY_IMPROVER_OVERLAP=0 AIJOCKEY_IMPROVER_SWAP=0"
    "B:ENERGY:AIJOCKEY_IMPROVER_ENERGY=1 AIJOCKEY_IMPROVER_OVERLAP=0 AIJOCKEY_IMPROVER_SWAP=0"
    "C:E+O:AIJOCKEY_IMPROVER_ENERGY=1 AIJOCKEY_IMPROVER_OVERLAP=1 AIJOCKEY_IMPROVER_SWAP=0"
    "D:E+O+S:AIJOCKEY_IMPROVER_ENERGY=1 AIJOCKEY_IMPROVER_OVERLAP=1 AIJOCKEY_IMPROVER_SWAP=1"
)

PY=${PYTHON:-/opt/venv/bin/python}
[ -x "$PY" ] || PY=python

echo "=== run $RUN_ID ==="
echo "user_pool: $USER_POOL"
echo "duration:  $DURATION"
echo "mix_mode:  $MIX_MODE"
echo "prompts:   $PROMPTS"
echo "seeds:     $SEEDS"
echo "probe log: $PROBE_LOG_DIR/$RUN_ID.jsonl"
echo

OUT_ROOT="output/cohorts/$RUN_ID"
mkdir -p "$OUT_ROOT"

# Read prompts via python (avoid jq dependency)
N_PROMPTS=$($PY -c "import json,sys;print(len(json.load(open('$PROMPTS'))))")
echo "$N_PROMPTS prompts loaded"
echo

n_done=0
n_fail=0
total=$(( ${#COHORTS[@]} * N_PROMPTS * SEEDS ))
t0=$(date +%s)

for cohort_def in "${COHORTS[@]}"; do
    cohort_name=$(echo "$cohort_def" | cut -d: -f1)
    cohort_label=$(echo "$cohort_def" | cut -d: -f2)
    cohort_env=$(echo "$cohort_def" | cut -d: -f3-)
    cohort_dir="$OUT_ROOT/$cohort_name"
    mkdir -p "$cohort_dir"

    echo "=== cohort $cohort_name ($cohort_label) === [$cohort_env]"

    for i in $(seq 0 $((N_PROMPTS - 1))); do
        prompt=$($PY -c "import json;d=json.load(open('$PROMPTS'));print(d[$i]['prompt'])")
        arc=$($PY -c "import json;d=json.load(open('$PROMPTS'));print(d[$i].get('arc','build'))")
        pid=$($PY -c "import json;d=json.load(open('$PROMPTS'));print(d[$i].get('id',f'p{i:02d}'))")

        for seed in $(seq 1 $SEEDS); do
            out_wav="$cohort_dir/${pid}_seed${seed}.wav"
            log="$cohort_dir/${pid}_seed${seed}.log"

            # Render via CLI. AIJOCKEY_PROBE_LOG points each render's row at
            # the per-cohort JSONL — easier post-hoc bucketing than a single
            # global file.
            env $cohort_env \
                AIJOCKEY_PROBE_LOG="$PROBE_LOG_DIR/${RUN_ID}_${cohort_name}.jsonl" \
                AIJOCKEY_PROBE_COHORT="$cohort_name" \
                AIJOCKEY_RANDOM_SEED="$seed" \
                $PY -m src.main all \
                    --clips "$USER_POOL" \
                    --cache cache \
                    --out "$out_wav" \
                    --prompt "$prompt" \
                    --arc "$arc" \
                    --duration "$DURATION" \
                    --mix_mode "$MIX_MODE" \
                    --use_director 1 \
                    > "$log" 2>&1 \
                && status="ok" || status="fail"

            if [ "$status" = "ok" ]; then
                n_done=$((n_done+1))
            else
                n_fail=$((n_fail+1))
                echo "  ! $cohort_name $pid seed=$seed FAIL — see $log"
            fi
            elapsed=$(( $(date +%s) - t0 ))
            avg=$(( elapsed / (n_done + n_fail) ))
            remain=$(( avg * (total - n_done - n_fail) ))
            echo "  [$cohort_name] $pid seed=$seed: $status (${elapsed}s elapsed, ~${remain}s remaining)"
        done
    done
done

echo
echo "=== done: $n_done ok, $n_fail fail in $(( $(date +%s) - t0 ))s ==="
echo
echo "Per-cohort log files:"
ls -la "$PROBE_LOG_DIR/${RUN_ID}_"*.jsonl 2>/dev/null
echo
echo "Summarize:"
echo "  $PY -m probe_log summary --pattern '$PROBE_LOG_DIR/${RUN_ID}_*.jsonl'"
echo "Or per cohort:"
echo "  for c in A B C D; do echo \"--- \$c ---\"; $PY -m probe_log summary --log $PROBE_LOG_DIR/${RUN_ID}_\$c.jsonl; done"
