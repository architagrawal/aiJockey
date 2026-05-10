#!/usr/bin/env bash
# Run BEFORE taking a slim snapshot. Strips ~40+ GB of re-fetchable model
# weights, downloaded raw audio, and ephemeral artifacts so the snapshot
# is small. Keeps the irreplaceable bits: cache JSONs, stems, checkpoints,
# probe logs, source maps.
#
# Modes:
#   slim (default) — aggressive cleanup, drops HF caches + raw + intermediate
#                    expected after-size: ~10-20 GB on /
#   fat            — light cleanup (logs + apt + tmp only); ~70 GB after
#
# Run inside the droplet (NOT inside the container — host-level cleanup):
#   bash scripts/clean_for_snapshot.sh           # slim
#   bash scripts/clean_for_snapshot.sh --fat     # fat
#
# After resume from the slim snapshot, run:
#   docker exec rocm python /workspace/scripts/prefetch_models.py
#
# Reference: scripts/snapshot_manifest.md for the full inventory.
set -euo pipefail

MODE="slim"
case "${1:-}" in
    --fat) MODE="fat" ;;
    --slim|"") MODE="slim" ;;
    *) echo "unknown arg: $1 (use --slim or --fat)"; exit 1 ;;
esac

echo "=== mode: $MODE ==="
echo
echo "=== before ==="
df -h / | tail -1
echo

# ---------------------------------------------------------------------------
# 1. HF model caches (slim only — fat keeps for fast restart)
# ---------------------------------------------------------------------------
if [ "$MODE" = "slim" ]; then
    echo "purging HF caches (Qwen ~30GB, CLAP ~1.7GB, Demucs ~80MB)..."
    docker exec rocm bash -c '
      rm -rf /root/.cache/huggingface/hub/models--Qwen--*
      rm -rf /root/.cache/huggingface/hub/models--laion--clap-*
      rm -rf /root/.cache/huggingface/hub/models--facebook--musicgen-*
      rm -rf /root/.cache/huggingface/hub/models--CPJKU--beat_this*
      rm -rf /root/.cache/torch/hub/checkpoints/*
      rm -rf /root/.cache/pip
      rm -rf /scratch/hf_cache 2>/dev/null
      echo cleaned-container-caches
    ' 2>/dev/null || echo "container not running (ok)"
fi

# ---------------------------------------------------------------------------
# 2. Raw downloaded audio (slim only — re-derivable via S0 / IA fetcher)
# ---------------------------------------------------------------------------
if [ "$MODE" = "slim" ]; then
    echo "removing raw downloaded audio + intermediate stage data..."
    docker exec rocm bash -c '
      rm -rf /scratch/raw
      rm -rf /scratch/transitions
      rm -rf /scratch/renders
      rm -rf /scratch/output
      echo cleaned-scratch-derived
    ' 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 3. Ephemeral test outputs + scratchpads (always)
# ---------------------------------------------------------------------------
echo "removing test scratchpads + render artifacts..."
docker exec rocm bash -c '
  rm -rf /workspace/test_user_cache /workspace/test_user_cache_v*
  rm -rf /workspace/test_user_clips /workspace/test_clips_lib
  rm -rf /workspace/test_cache /workspace/test_cache_lib
  rm -rf /workspace/test_out /workspace/test_out_lib /workspace/test_out_v*
  rm -rf /workspace/clips_par_test /workspace/clips_smoke
  rm -rf /workspace/cache_par_seq /workspace/cache_par_par /workspace/cache_smoke
  rm -rf /workspace/output/live
  # log files (regenerated next launch). preserve probe logs at /scratch/probes/
  rm -rf /workspace/logs/*.log
  rm -f /workspace/*.log
  rm -rf /workspace/.pytest_cache
  find /workspace -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
  echo cleaned-workspace
' 2>/dev/null || true

# Pre-container-era artifacts on host
rm -rf /root/clips_upload /root/dj_sets_mp3 /root/test_clips 2>/dev/null || true
rm -f /root/test_out_*.wav /root/critic.npz 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. Datasets (slim only — yt-dlp / Mixotic re-fetchable; expensive but doable)
# ---------------------------------------------------------------------------
if [ "$MODE" = "slim" ]; then
    docker exec rocm bash -c '
      rm -rf /workspace/datasets/dj_sets_mp3
      rm -rf /workspace/datasets/dj_sets
      rm -rf /workspace/datasets/critic.npz
    ' 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 5. APT + system caches (always)
# ---------------------------------------------------------------------------
apt-get clean -y 2>/dev/null || true
rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* 2>/dev/null || true
journalctl --vacuum-time=1h 2>/dev/null || true

# ---------------------------------------------------------------------------
# Verification: confirm we KEPT the irreplaceable bits
# ---------------------------------------------------------------------------
echo
echo "=== verifying preserved artifacts ==="
docker exec rocm bash -c '
  for path in /cache /workspace/checkpoints /workspace/samples \
              /workspace/clips_demo /scratch/preferences /scratch/probes \
              /scratch/embed /scratch/models; do
    if [ -d "$path" ] || [ -f "$path" ]; then
      sz=$(du -sh "$path" 2>/dev/null | cut -f1)
      echo "  ✓ $path ($sz)"
    fi
  done
' 2>/dev/null || true

echo
echo "=== after ==="
df -h / | tail -1
echo
echo "Now:"
echo "  1. shutdown the droplet (panel or 'shutdown -h now')"
echo "  2. snapshot via AMD Dev Cloud panel"
echo "  3. once snapshot complete: destroy droplet to stop billing"
