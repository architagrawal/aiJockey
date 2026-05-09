#!/usr/bin/env bash
# Run BEFORE taking a slim snapshot. Strips ~30+ GB of re-fetchable model
# weights and ephemeral artifacts from boot disk so the snapshot is small.
#
# Run inside the droplet (NOT inside the container — host-level cleanup):
#   bash scripts/clean_for_snapshot.sh
#
# After resume from the snapshot, run:
#   docker exec rocm python /workspace/aiJockey/scripts/prefetch_models.py
#
set -euo pipefail

echo "=== before ==="
df -h / | tail -1

# 1. HF model caches inside container (Qwen 30 GB, Demucs 80 MB, CLAP 1.7 GB)
echo "purging HF caches inside rocm container..."
docker exec rocm bash -c '
  rm -rf /root/.cache/huggingface/hub/models--Qwen--*
  rm -rf /root/.cache/huggingface/hub/models--laion--clap-*
  rm -rf /root/.cache/huggingface/hub/models--facebook--musicgen-*
  rm -rf /root/.cache/torch/hub/checkpoints/*
  rm -rf /root/.cache/pip
  echo cleaned-container
' 2>/dev/null || echo "container not running (ok)"

# 2. Re-uploadable / ephemeral
echo "removing uploaded clip stagings + test outputs..."
rm -rf /root/clips_upload /root/dj_sets_mp3 /root/test_clips
rm -f  /root/test_out_*.wav /root/critic.npz
docker exec rocm bash -c '
  rm -rf /workspace/aiJockey/datasets/dj_sets_mp3
  rm -rf /workspace/aiJockey/datasets/dj_sets
  rm -rf /workspace/aiJockey/datasets/critic.npz
  rm -rf /workspace/aiJockey/test_out
  rm -rf /workspace/aiJockey/test_out_lib
  rm -rf /workspace/aiJockey/test_out_v*
  rm -rf /workspace/aiJockey/test_clips_lib /workspace/aiJockey/test_clips
  rm -rf /workspace/aiJockey/test_cache /workspace/aiJockey/test_cache_lib
  rm -rf /workspace/aiJockey/output/live
  rm -rf /workspace/aiJockey/clips_par_test /workspace/aiJockey/cache_par_seq /workspace/aiJockey/cache_par_par
  rm -rf /workspace/aiJockey/clips_smoke /workspace/aiJockey/cache_smoke
  rm -f  /workspace/aiJockey/*.log
  echo cleaned-workspace
' 2>/dev/null || true

# 3. apt cache
apt-get clean -y || true
rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* 2>/dev/null || true

# 4. journald logs (small but tidy)
journalctl --vacuum-time=1h 2>/dev/null || true

echo "=== after ==="
df -h / | tail -1
echo
echo "Now: clean shutdown then take snapshot from DO panel."
echo "  shutdown -h now"
