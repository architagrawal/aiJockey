# Demo deploy runbook

Steps to ship the public demo after cohort baseline + listen test pass.
Critical path only. Backlog (S5/S7 self-play, BS-Roformer, model swaps) runs
post-deploy, not gating.

## Prerequisites (gate)

Before starting deploy:

- [ ] Cohort baseline result: cohort D mean severity ≤ 0.5 (loose) or ≤ 0.3 (target)
- [ ] Listen test: 5+ blind ranks, improver-on > improver-off in majority
- [ ] `improver_state` env knobs locked into a config (e.g. all three ON, or specific subset)
- [ ] Last commit on `best-output-pipeline` includes the locked config as default
- [ ] Tests green: `pytest tests/ -q`

If any of these fails → loop back to triage, not forward to deploy.

## Step 1 — Final demo render (~30 min)

5 demo MP3s with the locked pipeline. Curated user-clip pools per genre.

```bash
# On droplet, inside rocm container
cd /workspace
for preset in festival_inferno midnight_noir neon_retrowave east_meets_bass bollywood_block_party; do
    /opt/venv/bin/python -m src.main all \
        --clips clips_demo/$preset \
        --cache cache \
        --out /workspace/demo_mp3/${preset}.mp3 \
        --preset $preset \
        --duration 180 \
        --use_director 1 \
        --mix_mode balanced 2>&1 | tee /workspace/logs/demo_${preset}.log
done

# Verify each demo's probe log row severity
python -m probe_log summary --log /scratch/probes/latest.jsonl --tail 5
```

Acceptance: each demo's severity ≤ 0.5 AND human-listen pass.

If any demo's severity > 0.5 or sounds wrong → re-render with seed bumped, or fall
back to the previous best demo MP3 from the snapshot.

## Step 2 — Server hardening (~15 min)

```bash
# In container, on the host where /generate runs
export SERVER_KEY=$(openssl rand -hex 24)
echo "$SERVER_KEY" > /workspace/SERVER_KEY.txt    # for HF Space secret
export AIJOCKEY_JOB_TIMEOUT_SEC=900               # 15 min hard cap
export AIJOCKEY_USE_DIRECTOR_LLM=1
export HF_HOME=/scratch/hf_cache

# Smoke /generate end-to-end via curl
curl -X POST -H "X-Key: $SERVER_KEY" \
    -F "files=@clips_test/clip1.wav" \
    -F "files=@clips_test/clip2.wav" \
    -F "files=@clips_test/clip3.wav" \
    -F "preset=festival_inferno" \
    -F "duration=120" \
    -F "mix_mode=balanced" \
    http://localhost:8000/generate -o /tmp/test_render.wav
# Expect: 200 OK, .wav file written, X-Probe header present
```

Acceptance: `curl -I http://localhost:8000/health` returns 200 AND
`/generate` smoke returns audio with X-Probe severity ≤ 0.5.

## Step 3 — Tunnel (~5 min)

ngrok free static domain (per STATUS — `issue-slingshot-bobsled.ngrok-free.dev`).

```bash
ngrok http 8000 --domain=issue-slingshot-bobsled.ngrok-free.dev &
sleep 3
curl -s https://issue-slingshot-bobsled.ngrok-free.dev/health | jq .
```

Acceptance: tunnel URL responds with `{"status": "ok", ...}` from the laptop.

## Step 4 — HF Space deploy (~30 min)

Push `space/` directory to HF Space repo. Set secrets:

| Secret | Value | Purpose |
|--------|-------|---------|
| `MI300X_URL` | `https://issue-slingshot-bobsled.ngrok-free.dev` | Backend pointer |
| `MI300X_KEY` | `$SERVER_KEY` from step 2 | Auth |
| `ADMIN_PW` | strong random | Try-It tab gate |

```bash
cd /workspace/space
git remote -v   # verify HF remote
git add .
git commit -m "deploy: locked v1 pipeline"
git push hf main
```

Wait for HF Space build (~3-5 min). Then test from browser:

1. Public demo tab: 5 pre-baked MP3s play
2. Try It tab (with ADMIN_PW): upload 3 clips, generate, audio returned

Acceptance: both tabs work end-to-end through the tunnel.

## Step 5 — Snapshot (~10 min)

```bash
# On the droplet host (NOT inside container)
bash /root/aijockey/scripts/clean_for_snapshot.sh

# Then on AMD Dev Cloud dashboard:
# 1. Stop droplet (do NOT destroy yet)
# 2. Create snapshot named "aijockey-demo-v1-2026-05-10"
# 3. Verify snapshot completes (~5-10 min depending on size)
```

Acceptance: snapshot listed in dashboard, status "available".

## Step 6 — Destroy droplet (~1 min)

After snapshot confirmed:

```bash
# Stop billing immediately. Restore via snapshot when needed.
# AMD Dev Cloud dashboard: Destroy droplet.
```

Cost stops the moment droplet destroyed. Snapshot retains state at ~$7/mo.

## Step 7 — Verify ship

- [ ] HF Space URL accessible publicly
- [ ] Demo tab plays 5 mixes
- [ ] Try It tab works through ADMIN_PW
- [ ] Snapshot named + tagged
- [ ] Droplet destroyed (zero compute cost)
- [ ] STATUS.md updated with "demo-v1 shipped on 2026-05-10"
- [ ] Slack/email teammate the public URL

## Rollback

If anything breaks post-deploy:

1. **Backend unreachable** (tunnel down) — restart droplet from snapshot, repoint ngrok, repush HF Space secret.
2. **Renders fail** — check `/scratch/probes/*.jsonl` for severity spike, rerun cohort triage. Either revert to last good commit OR bump improver knobs back to defaults.
3. **HF Space crashes** — check Space logs in HF dashboard, redeploy from `space/` repo.
4. **Snapshot restore needed** — see `scripts/snapshot_manifest.md` for restore procedure.

## Post-deploy backlog (parallel, not gating)

Once demo is live, kick these on a fresh droplet from snapshot:

- S2 segment + S3 embed pipeline (foundation for self-play)
- S5 self-play with K-render + critic rerank → preference pairs
- S7 ORPO LoRA Director training
- CLAP head v2 + mix critic v2 retraining
- BS-Roformer activation when vocal complaints surface

Each of these improves demo-v2 quality. None blocks demo-v1 ship.

## Cost estimate

| Phase | Time | Cost |
|-------|-----:|-----:|
| Cohort triage | 60 min | $2 |
| Listen test | 60 min | (laptop) |
| Demo render | 30 min | $1 |
| Server harden + tunnel | 20 min | $0.66 |
| HF Space deploy | 30 min | (free tier) |
| Snapshot | 10 min | $7/mo persist |
| Total to ship | ~3-4 hr | ~$5 + $7/mo storage |

After destroy: $0/hr compute until next snapshot restore.
