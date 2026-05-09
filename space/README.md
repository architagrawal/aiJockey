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
---

# AiJockey HF Space

Public demo + password-gated live generation backed by AMD MI300X.

## Required secrets (Settings → Variables and secrets)

| Name | Type | Value |
|------|------|-------|
| `ADMIN_PW`   | secret | strong password for Try It tab |
| `MI300X_URL` | secret | `https://issue-slingshot-bobsled.ngrok-free.dev` |
| `MI300X_KEY` | secret | shared secret matching `SERVER_KEY` on droplet |

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
