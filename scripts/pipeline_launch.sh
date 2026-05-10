#!/usr/bin/env bash
# Boot the full pipeline-parallel stack in a tmux session.
# Run this on the MI300X box once after `git pull` + venv activation.
#
# Reference: docs/phase1_plan.md §15.9.

set -e

SESSION="${AIJOCKEY_TMUX_SESSION:-aijockey}"
SCRATCH="${AIJOCKEY_SCRATCH:-/scratch}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "${SCRATCH}"/{raw,cache,transitions,embed,renders,preferences,models,output,prompts}

cd "${REPO}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session ${SESSION} exists; attach with: tmux attach -t ${SESSION}"
    exit 0
fi

tmux new-session -d -s "${SESSION}" -n dev
for w in s0_download s1_analyze s2_segment s3_embed s4_critic s5_selfplay s7_dpo s8_bridge monitor; do
    tmux new-window -t "${SESSION}" -n "${w}"
done

PY="${PYTHON:-python}"
ENV_PREFIX="AIJOCKEY_PHASE=1 AIJOCKEY_DTYPE=bfloat16 AIJOCKEY_COMPILE=1 AIJOCKEY_FLASH_ATTN=2 AIJOCKEY_OPTIMIZER=lion AIJOCKEY_PHRASE_QUANTIZE=1 AIJOCKEY_STEM_SWAP=1 AIJOCKEY_CONSTITUTIONAL=1 AIJOCKEY_SCRATCH=${SCRATCH}"

tmux send-keys -t "${SESSION}:s0_download"  "${ENV_PREFIX} ${PY} scripts/stage0_download.py --src mixotic,fma_medium,mtg_jamendo --max-per-src 1000" C-m
tmux send-keys -t "${SESSION}:s1_analyze"   "${ENV_PREFIX} ${PY} scripts/stage1_analyze.py --watch ${SCRATCH}/raw" C-m
tmux send-keys -t "${SESSION}:s2_segment"   "${ENV_PREFIX} ${PY} scripts/stage2_segment.py --watch ${SCRATCH}/cache" C-m
tmux send-keys -t "${SESSION}:s3_embed"     "${ENV_PREFIX} ${PY} scripts/stage3_embed.py --watch ${SCRATCH}/cache" C-m
tmux send-keys -t "${SESSION}:s4_critic"    "${ENV_PREFIX} ${PY} scripts/stage4_critic.py --watch ${SCRATCH}/transitions --min-samples 200" C-m
tmux send-keys -t "${SESSION}:s5_selfplay"  "${ENV_PREFIX} ${PY} scripts/stage5_selfplay.py --k 8" C-m
tmux send-keys -t "${SESSION}:s7_dpo"       "${ENV_PREFIX} ${PY} scripts/stage7_dpo.py" C-m
tmux send-keys -t "${SESSION}:s8_bridge"    "${ENV_PREFIX} ${PY} scripts/stage8_bridge.py" C-m
tmux send-keys -t "${SESSION}:monitor"      "${PY} scripts/monitor.py" C-m

echo "launched. attach: tmux attach -t ${SESSION}"
