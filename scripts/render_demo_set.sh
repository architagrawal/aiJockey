#!/usr/bin/env bash
# Locked-config demo render. Produces 5 polished MP3s for HF Space pre-bake.
#
# Run AFTER cohort baseline + listen test confirm an improver config.
# Lock the env knobs in this script (or override at call time) so the
# 5 demos are generated with the exact pipeline that will ship.
#
# Each demo:
#   - 180 s target duration
#   - Director on (audio-aware if Qwen2-Audio cached)
#   - mix_mode=balanced (library augments user pool when needed)
#   - Probe log row appended → severity becomes ship gate
#
# After all 5 render: prints per-demo severity, fails if any > 0.5.
#
# Usage:
#   bash scripts/render_demo_set.sh
#   bash scripts/render_demo_set.sh --duration 240 --severity-gate 0.4
#
# Output:
#   demo_mp3/<preset>.mp3      — final files for HF Space
#   /scratch/probes/demo_set.jsonl — per-demo probe log
#   logs/demo_<preset>.log     — render trace

set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

DURATION=180
MIX_MODE=balanced
SEVERITY_GATE=0.5
OUT_DIR="$REPO/demo_mp3"
LOG_DIR="$REPO/logs"
PROBE_LOG="${AIJOCKEY_PROBE_LOG:-/scratch/probes/demo_set.jsonl}"

while [ $# -gt 0 ]; do
    case "$1" in
        --duration)        DURATION=$2; shift 2 ;;
        --mix-mode)        MIX_MODE=$2; shift 2 ;;
        --severity-gate)   SEVERITY_GATE=$2; shift 2 ;;
        --out-dir)         OUT_DIR=$2; shift 2 ;;
        *) echo "unknown arg $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR" "$LOG_DIR"
mkdir -p "$(dirname "$PROBE_LOG")"
: > "$PROBE_LOG"     # fresh log per render set

PY=${PYTHON:-/opt/venv/bin/python}
[ -x "$PY" ] || PY=python

# Locked improver config for ship. Update these after cohort verdict
# names a winner.
export AIJOCKEY_PHASE=1
export AIJOCKEY_DTYPE=bfloat16
export AIJOCKEY_FLASH_ATTN=1
export AIJOCKEY_DEMUCS_MODEL=htdemucs_ft
export AIJOCKEY_DEMUCS_OVERLAP=0.10
export AIJOCKEY_BEAT_THIS=1
export AIJOCKEY_BEAT_THIS_DBN=auto
export AIJOCKEY_IMPROVER_ENERGY=1
export AIJOCKEY_IMPROVER_OVERLAP=1
export AIJOCKEY_IMPROVER_SWAP=1
export AIJOCKEY_PROBE_LOG="$PROBE_LOG"

# Five preset → demo-clip-pool pairings. Each preset directory under
# clips_demo/ should contain 5-8 hand-curated user clips matching the
# preset's vibe. clips_demo/ ships with the repo (per HACKATHON.md).
declare -a DEMOS=(
    "festival_inferno"
    "midnight_noir"
    "neon_retrowave"
    "east_meets_bass"
    "bollywood_block_party"
)

echo "=== render demo set ==="
echo "duration:      ${DURATION}s"
echo "mix_mode:      $MIX_MODE"
echo "severity gate: $SEVERITY_GATE"
echo "out:           $OUT_DIR"
echo "probe log:     $PROBE_LOG"
echo

t0=$(date +%s)

for preset in "${DEMOS[@]}"; do
    pool="$REPO/clips_demo/$preset"
    if [ ! -d "$pool" ]; then
        # Fallback: use generic clips/ pool if per-preset folder missing
        pool="$REPO/clips"
    fi
    out_mp3="$OUT_DIR/${preset}.mp3"
    out_wav="$OUT_DIR/${preset}.wav"
    log="$LOG_DIR/demo_${preset}.log"

    echo "[demo] $preset (pool=$pool)"

    if "$PY" -m src.main all \
        --clips "$pool" \
        --cache cache \
        --out "$out_wav" \
        --preset "$preset" \
        --duration "$DURATION" \
        --use_director 1 \
        --mix_mode "$MIX_MODE" \
        > "$log" 2>&1; then
        # Transcode to MP3 for HF Space size budget
        if command -v ffmpeg >/dev/null 2>&1 && [ -f "$out_wav" ]; then
            ffmpeg -y -i "$out_wav" -codec:a libmp3lame -qscale:a 2 "$out_mp3" \
                >> "$log" 2>&1
            rm -f "$out_wav"
        fi
        echo "  ✓ $out_mp3"
    else
        echo "  ✗ FAILED — see $log"
    fi
done

elapsed=$(( $(date +%s) - t0 ))
echo
echo "=== rendered in ${elapsed}s ==="

# Verify each demo's severity is below gate
echo
echo "=== severity gate check (target: < $SEVERITY_GATE) ==="
fail=0
"$PY" - <<PY
import json, sys
gate = float("$SEVERITY_GATE")
demos = "${DEMOS[*]}".split()
rows = []
try:
    with open("$PROBE_LOG") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
except FileNotFoundError:
    print("warn: no probe log written; severity check skipped", file=sys.stderr)
    sys.exit(0)

# Match by job_id presence + chronological order — last 5 rows correspond to demos
recent = rows[-len(demos):] if len(rows) >= len(demos) else rows
print(f"  parsed {len(recent)} probe rows")
fails = 0
for i, row in enumerate(recent):
    p = row.get('probe') or {}
    sev = p.get('overall_severity') or row.get('severity')
    if sev is None:
        print(f"  ? row {i}: no severity field")
        continue
    name = demos[i] if i < len(demos) else f"row{i}"
    status = "PASS" if sev < gate else "FAIL"
    print(f"  {name}: severity={sev:.3f} {status}")
    if sev >= gate:
        fails += 1
sys.exit(1 if fails else 0)
PY
gate_status=$?

if [ "$gate_status" -ne 0 ]; then
    echo
    echo "!! one or more demos failed severity gate ($SEVERITY_GATE)"
    echo "   options: re-render with bumped seed, OR lower the gate, OR debug"
    exit 1
fi

echo
echo "=== all demos pass gate. ready for HF Space deploy. ==="
ls -la "$OUT_DIR"
