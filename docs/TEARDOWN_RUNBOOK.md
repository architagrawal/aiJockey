# Teardown runbook — credit-out

When GPU credits run out (current: ~5 hr remaining as of session 2026-05-11),
run this checklist to preserve session artifacts without paying for idle GPU.

## 1. Save artifacts to local laptop (~5 min)

```bash
# Curated gallery + best outputs
scp -i ~/.ssh/aijockey_mi300x -r \
  root@165.245.135.121:/workspace/output/iter_refined_v3 \
  root@165.245.135.121:/workspace/output/iter_refined_v4 \
  root@165.245.135.121:/workspace/output/iter_refined_v5 \
  root@165.245.135.121:/workspace/output/grid_sweep_v2 \
  root@165.245.135.121:/workspace/output/_final_top10 \
  ./local_output/

# Trained reward heads / DPO ckpts
scp -i ~/.ssh/aijockey_mi300x \
  root@165.245.135.121:/scratch/mert_reward.pt \
  root@165.245.135.121:/scratch/vampnet_ft_v4/coarse.pth \
  root@165.245.135.121:/scratch/vampnet_dpo_v3.pth \
  ./local_artifacts/

# Plan-stats JSONL (DPO/KTO training data)
scp -i ~/.ssh/aijockey_mi300x \
  root@165.245.135.121:/scratch/probes/plan_stats.jsonl \
  ./local_artifacts/

# VampNet bridges (145 wavs, ~500 MB)
scp -i ~/.ssh/aijockey_mi300x -r \
  root@165.245.135.121:/cache/vampnet_bridges \
  ./local_artifacts/
```

## 2. Snapshot cache metadata (skip stems, 50+ GB)

```bash
ssh root@165.245.135.121 'docker exec rocm bash -c "
  cd /cache && tar czf /workspace/cache_metadata.tar.gz \
    *.json *.npz *.audiobox_slices.json *.mert_pred.json *.stem_audiobox.json \
    2>/dev/null
  du -h /workspace/cache_metadata.tar.gz
"'
scp -i ~/.ssh/aijockey_mi300x root@165.245.135.121:/workspace/cache_metadata.tar.gz ./local_artifacts/
```

Restore cost on next GPU spin-up: ~5 min (extract + Audiobox+Mel-Band reload from HF).

## 3. Stop services, verify nothing important running

```bash
ssh root@165.245.135.121 'docker exec rocm bash -c "
  pgrep -af uvicorn
  pgrep -af ngrok
  pgrep -af python | head
"'
```

## 4. Push final git state

```bash
ssh root@165.245.135.121 'docker exec rocm bash -c "
  cd /workspace && git status --short
"'
# Any uncommitted local changes → grab + commit before destroy.
```

## 5. Destroy droplet

DigitalOcean Console → droplet → Destroy. Confirms via name.

Alt: `doctl compute droplet delete <id>` after `pip install python-digitalocean`.

## 6. Verify HF Space still works without backend

HF Space at `lablab-ai-amd-developer-hackathon/aijockey` polls backend
at `issue-slingshot-bobsled.ngrok-free.dev`. When droplet dead, ngrok
tunnel dies, Space shows backend-unavailable. Either:
  - Update Space README to "demo offline" banner
  - Or stop the Space too (Settings → Pause)

## 7. Next GPU spin-up checklist

When credits reload / new droplet:

```bash
# 1. Spin droplet at DigitalOcean (gpu-mi300x)
# 2. SSH + start rocm container per AGENTS.md
# 3. git clone https://github.com/architagrawal/aiJockey /workspace
# 4. Restore cache metadata
tar xzf cache_metadata.tar.gz -C /cache/
# 5. Restore trained ckpts
mkdir -p /scratch && cp mert_reward.pt vampnet_ft_v4/ vampnet_dpo_v3.pth /scratch/
# 6. Re-fetch HF models (Audiobox, Mel-Band, Qwen2.5, VampNet) via warmup
# 7. Restart uvicorn per docs/deploy_runbook.md
# 8. Re-establish ngrok with same domain (issue-slingshot-bobsled.ngrok-free.dev)
```

## State as of 2026-05-11

- 50+ commits on `best-output-pipeline` branch
- ~35 src modules, ~15 scripts
- 35+ env flags toggleable
- 145 VampNet bridges in /cache
- 331 stem-audiobox sidecars
- 83 mert_pred sidecars
- 54 audiobox_slices sidecars
- MERT-95M reward head trained (n=32)
- VampNet finetune trained (n=4, 2 epochs)
- VampNet DPO trained (8 pref pairs, 2 epochs)
- Live demo at HF Space `lablab-ai-amd-developer-hackathon/aijockey`
