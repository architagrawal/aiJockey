#!/usr/bin/env bash
# Pre-flight checklist for MI300X pipeline launch.
# Run BEFORE `pipeline_launch.sh` to catch bring-up issues without burning GPU time
# on a doomed launch. Each check exits non-zero on failure.
#
# Usage: bash scripts/preflight.sh
#
# Exit codes:
#   0   all green — safe to launch pipeline
#   1   GPU / ROCm baseline failed
#   2   Python deps missing
#   3   Models not cached / wrong defaults
#   4   Scratch disk too low
#   5   Code-state issue (smoke test fail)

set -e
set -o pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="${AIJOCKEY_SCRATCH:-/scratch}"
MIN_SCRATCH_GB="${MIN_SCRATCH_GB:-200}"
PY="${PYTHON:-python}"

ok() { echo "  ✓ $1"; }
fail() { echo "  ✗ $1" >&2; exit "${2:-1}"; }
heading() { echo; echo "== $1 =="; }

cd "${REPO}"

heading "1. GPU + ROCm baseline"
command -v rocm-smi >/dev/null 2>&1 || fail "rocm-smi not on PATH" 1
rocm-smi --showid >/dev/null 2>&1 || fail "rocm-smi cannot enumerate GPUs" 1
ok "rocm-smi reachable"
${PY} -c "import torch; assert torch.cuda.is_available(), 'torch.cuda.is_available()=False'" || fail "PyTorch can't see GPU" 1
ok "torch.cuda.is_available() = True"
${PY} -c "import torch; print(torch.cuda.get_device_name(0))" | grep -i "instinct\|MI300\|AMD" >/dev/null \
    || fail "GPU name doesn't look like AMD Instinct" 1
ok "GPU = AMD Instinct"

heading "2. Python deps"
for pkg in torch torchaudio transformers demucs librosa pyrubberband fastapi peft trl accelerate datasets; do
    ${PY} -c "import ${pkg}" 2>/dev/null \
        || fail "missing dep: ${pkg} — pip install -r requirements-rocm.txt" 2
    ok "import ${pkg}"
done
# Optional but expected
${PY} -c "import beat_this" 2>/dev/null && ok "import beat_this (optional)" \
    || echo "  ! beat_this missing — librosa downbeat fallback will run (degraded)"
${PY} -c "import bs_roformer" 2>/dev/null && ok "import bs_roformer (optional)" \
    || echo "  ! bs_roformer missing — htdemucs_ft vocals will be used"
${PY} -c "import laion_clap" 2>/dev/null && ok "laion-clap (preferred CLAP backend)" \
    || echo "  ! laion-clap missing — transformers ClapModel fallback will be used"

heading "3. Models cached"
${PY} -c "
from demucs.pretrained import get_model
m = get_model('${AIJOCKEY_DEMUCS_MODEL:-htdemucs_ft}')
print('demucs ok')
" >/dev/null 2>&1 || fail "demucs ${AIJOCKEY_DEMUCS_MODEL:-htdemucs_ft} not cached — run scripts/prefetch_models.py" 3
ok "demucs ${AIJOCKEY_DEMUCS_MODEL:-htdemucs_ft} loadable"
${PY} -c "
from transformers import ClapModel
ClapModel.from_pretrained('laion/clap-htsat-unfused')
print('clap ok')
" >/dev/null 2>&1 || fail "CLAP not cached — run scripts/prefetch_models.py" 3
ok "CLAP cached"
# Director: prefetch is heavy (~14-16GB); skip-check unless explicitly requested
if [ "${PREFLIGHT_CHECK_DIRECTOR:-0}" = "1" ]; then
    ${PY} -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
print('qwen text ok')
" >/dev/null 2>&1 || fail "Qwen2.5-7B tokenizer not cached" 3
    ok "Qwen2.5-7B tokenizer cached"
else
    echo "  ! skipping Director cache check (PREFLIGHT_CHECK_DIRECTOR=1 to enable)"
fi

heading "4. Scratch disk"
if [ ! -d "${SCRATCH}" ]; then
    fail "scratch dir ${SCRATCH} does not exist (set AIJOCKEY_SCRATCH or mkdir)" 4
fi
free_gb=$(df -BG --output=avail "${SCRATCH}" | tail -1 | tr -dc '0-9')
if [ -z "${free_gb}" ] || [ "${free_gb}" -lt "${MIN_SCRATCH_GB}" ]; then
    fail "scratch ${SCRATCH} has ${free_gb}GB free, need >=${MIN_SCRATCH_GB}GB" 4
fi
ok "${SCRATCH} has ${free_gb}GB free"
for d in raw cache transitions embed renders preferences models output prompts; do
    mkdir -p "${SCRATCH}/${d}"
done
ok "scratch subdirs ready"

heading "5. Code state"
${PY} -c "
import sys
sys.path.insert(0, 'src')
import director, planner, execute, samples, constitutional
print('imports ok')
" >/dev/null 2>&1 || fail "src/ module imports failed" 5
ok "src/ imports clean"
${PY} -m pytest tests/test_phrase_quantize.py tests/test_constitutional.py tests/test_transition_mapping.py tests/test_samples_gating.py -q --no-header 2>&1 | tail -3
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    fail "unit tests failing on this branch — fix before pipeline launch" 5
fi
ok "unit tests pass"

heading "6. Env knobs (informational)"
for k in AIJOCKEY_PHASE AIJOCKEY_DTYPE AIJOCKEY_FLASH_ATTN AIJOCKEY_COMPILE \
         AIJOCKEY_DEMUCS_MODEL AIJOCKEY_DEMUCS_OVERLAP AIJOCKEY_BEAT_THIS \
         AIJOCKEY_BS_ROFORMER AIJOCKEY_RENDER_WORKERS AIJOCKEY_STEM_WORKERS \
         AIJOCKEY_BATCH_CLAP HF_DIRECTOR_MODEL HF_HOME; do
    val="${!k:-<unset>}"
    echo "  ${k} = ${val}"
done

echo
echo "==========================================="
echo "PRE-FLIGHT GREEN. Safe to launch pipeline:"
echo "  bash scripts/pipeline_launch.sh"
echo "==========================================="
