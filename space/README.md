---
title: AiJockey
emoji: 🎧
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Open-source AI DJ. 5 wildly different mixes, same engine.
hf_oauth: true
hf_oauth_scopes:
  - openid
  - profile
---

# AiJockey HF Space

Public demo + password-gated live generation backed by AMD MI300X.

## Required secrets (Settings → Variables and secrets)

| Name | Type | Value |
|------|------|-------|
| `ADMIN_PW`   | secret | strong password for Try It tab |
| `MI300X_URL` | secret | `https://issue-slingshot-bobsled.ngrok-free.dev` |
| `MI300X_KEY` | secret | shared secret matching `SERVER_KEY` on droplet |

## API limits (keep Space UX aligned)

These mirror [server/api.py](../server/api.py):

| Constraint | Value |
|------------|--------|
| Clip count | 2–8 |
| Each file | ≤ 25 MB; extensions `.wav` `.mp3` `.flac` `.m4a` `.ogg` |
| Requested duration | 30–600 s (slider max may be shorter without **Use sample library**) |
| Concurrent jobs | up to `AIJOCKEY_INFLIGHT_MAX` (default 4); GPU stages (analyze + Director) serialize via `_gpu_lock`. `503` when slot pool full; retry after ~2 min |
| Job wall clock | backend default **1200 s** (`AIJOCKEY_JOB_TIMEOUT_SEC`) |
| Downloads | MP3 default; WAV/FLAC optional (Try It dropdown) |

**AGPL-3.0** — forks / network-hosted variants must remain open source under compatible terms.

## Repo layout

```
space/
  app.py              # this Gradio app
  README.md           # this file
  demo_mp3/           # 5 pre-baked mixes (uploaded after droplet pre-bake)
    festival_inferno.mp3
    midnight_noir.mp3
    neon_retrowave.mp3
    east_meets_bass.mp3
    bollywood_block_party.mp3
```

## Deploy

```bash
huggingface-cli login              # use the write token
huggingface-cli repo create aijockey --type space --space_sdk gradio
git clone https://huggingface.co/spaces/Archit00/aijockey hf-space
cp app.py README.md hf-space/
cp -r demo_mp3 hf-space/
cd hf-space && git add -A && git commit -m "init" && git push
```

## License

AGPL-3.0. See repo root.
