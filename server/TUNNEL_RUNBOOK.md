# Free tunnel URL rotation (HF Space → AMD backend)

When you use **free-tier** tunnels (ngrok free, etc.), the public **HTTPS URL can change** whenever the tunnel process restarts.

## Symptoms

- Hugging Face “Try It” tab returns “Backend unreachable” or `502` despite AMD instance running.

## Fix (every rotation)

1. SSH to the AMD host (or wherever `uvicorn` runs).
2. Restart the tunnel; copy the **new** `https://…` forwarding URL.
3. In HF Space **Settings → Repository secrets** update:
   - `MI300X_URL` → new URL (no trailing slash; use `https://…` origin only).
   - `MI300X_KEY` unchanged unless you also rotated `SERVER_KEY` on the server.
4. Open the Space → **Try It** → **Check backend** → run a tiny health check (`GET /health`).

Expected brief downtime equals tunnel restart + HF secret propagation (usually seconds to a minute).

## Paid / stable ingress

Use a paid ngrok/cloudflared **reserved hostname** or a tiny VPS reverse proxy so `MI300X_URL` stays constant.
