#!/usr/bin/env bash
# Boot script for AMD MI300X droplet (PyTorch 2.6 + ROCm 7.0 image).
# Run after first SSH-in. Idempotent: safe to re-run.
#
# Required env vars before running (or pass inline):
#   GITHUB_REPO     e.g. https://github.com/architagrawal/aiJockey.git
#   SERVER_KEY      shared secret for FastAPI auth (set in HF Space secret too)
#   NGROK_TOKEN     ngrok authtoken
#   NGROK_DOMAIN    e.g. issue-slingshot-bobsled.ngrok-free.dev
#   DO_API_TOKEN    DigitalOcean API token (for self-destroy on idle)
#   DROPLET_ID      this droplet's ID (from DO panel)
#   IDLE_MINUTES    default 15
#   HARD_CAP_HOURS  default 4

set -euo pipefail

: "${GITHUB_REPO:=https://github.com/architagrawal/aiJockey.git}"
: "${SERVER_KEY:?SERVER_KEY required}"
: "${NGROK_TOKEN:?NGROK_TOKEN required}"
: "${NGROK_DOMAIN:?NGROK_DOMAIN required}"
: "${DO_API_TOKEN:?DO_API_TOKEN required}"
: "${DROPLET_ID:?DROPLET_ID required}"
: "${IDLE_MINUTES:=15}"
: "${HARD_CAP_HOURS:=4}"

WORK=/root/aijockey

echo "=== [1/6] system packages ==="
apt-get update -qq
apt-get install -y -qq ffmpeg git curl tmux htop unzip rubberband-cli

echo "=== [2/6] clone repo ==="
if [ ! -d "$WORK/.git" ]; then
  git clone "$GITHUB_REPO" "$WORK"
else
  (cd "$WORK" && git pull --rebase)
fi
cd "$WORK"

echo "=== [3/6] python deps ==="
# pre-built image already has PyTorch+ROCm. Add aiJockey extras.
pip install --quiet --upgrade pip
pip install --quiet \
  demucs madmom librosa soundfile pyrubberband \
  transformers accelerate huggingface_hub \
  fastapi uvicorn python-multipart \
  pyloudnorm "numpy<1.24" scipy
pip install --quiet --no-deps audiocraft || true

echo "=== [4/6] ngrok ==="
if ! command -v ngrok >/dev/null; then
  curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz -o /tmp/ngrok.tgz
  tar xzf /tmp/ngrok.tgz -C /usr/local/bin
fi
ngrok config add-authtoken "$NGROK_TOKEN"

echo "=== [5/6] idle + hard-cap watchers ==="
cat >/usr/local/bin/aijockey-idle-destroy.sh <<EOF
#!/usr/bin/env bash
LAST=\$(cat /tmp/aijockey-last 2>/dev/null || date +%s)
NOW=\$(date +%s)
if [ \$((NOW - LAST)) -gt $((IDLE_MINUTES*60)) ]; then
  curl -sS -X DELETE -H "Authorization: Bearer $DO_API_TOKEN" \
    "https://api.digitalocean.com/v2/droplets/$DROPLET_ID"
fi
EOF
chmod +x /usr/local/bin/aijockey-idle-destroy.sh
date +%s > /tmp/aijockey-last
( crontab -l 2>/dev/null | grep -v aijockey-idle-destroy; \
  echo "* * * * * /usr/local/bin/aijockey-idle-destroy.sh >>/var/log/aijockey-idle.log 2>&1" \
) | crontab -

# hard cap: destroy after HARD_CAP_HOURS no matter what
at "now + ${HARD_CAP_HOURS} hours" <<EOF || true
curl -sS -X DELETE -H "Authorization: Bearer $DO_API_TOKEN" "https://api.digitalocean.com/v2/droplets/$DROPLET_ID"
EOF

echo "=== [6/6] launch services ==="
# kill any old session
tmux kill-session -t aijockey 2>/dev/null || true
tmux new-session -d -s aijockey -n api  "cd $WORK && SERVER_KEY=$SERVER_KEY IDLE_FILE=/tmp/aijockey-last python -m uvicorn server.api:app --host 0.0.0.0 --port 8000 2>&1 | tee /var/log/aijockey-api.log"
sleep 3
tmux new-window  -t aijockey   -n tunnel "ngrok http --domain=$NGROK_DOMAIN 8000 --log=stdout 2>&1 | tee /var/log/aijockey-tunnel.log"

echo
echo "=== READY ==="
echo "API:    http://localhost:8000/health"
echo "Public: https://$NGROK_DOMAIN/health"
echo "Tail:   tmux attach -t aijockey"
echo "Idle:   destroy after ${IDLE_MINUTES}min, hard cap ${HARD_CAP_HOURS}h"
