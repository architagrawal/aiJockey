#!/usr/bin/env bash
# Example: log /health responses for synthetic monitoring (cron every 5 min).
# Replace URL and optionally add Authorization if you terminate TLS elsewhere.
URL="${AIJOCKEY_HEALTH_URL:-https://YOUR-TUNNEL.example.com/health}"
curl -sS -m 15 -o "/tmp/aijockey-health.log" -w "%{http_code}" "$URL" | logger -t aijockey-health
