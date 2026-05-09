# Deploy AiJockey API on AMD Developer Cloud (MI300X / ROCm)

For **coding agents** (Claude Code, Cursor, etc.): read repo root [**AGENTS.md**](../AGENTS.md) first for architecture, env vars, `/generate` contract, and deploy pitfalls.

This is the **GPU worker** behind the Hugging Face Space. The Space only proxies uploads; all Demucs, CLAP, optional Director LLM, and mixing run here.

## One-time on the instance

```bash
git clone <your-repo> && cd aiJockey
python -m venv .venv && source .venv/bin/activate   # Linux
pip install -r requirements-rocm.txt
pip install fastapi uvicorn python-multipart
export SERVER_KEY='generate-a-long-random-secret'
export AI_DEVICE=cuda                         # ROCm uses the cuda API in PyTorch
export AIJOCKEY_JOB_TIMEOUT_SEC=1200         # optional, default 20 min
```

## Run the API

Use **tmux** so SSH drops don’t kill long jobs:

```bash
tmux new -s aijockey
cd aiJockey
source .venv/bin/activate
python -m uvicorn server.api:app --host 0.0.0.0 --port 8000
```

Expose port `8000` through **ngrok**, **cloudflared**, or a reverse proxy. Put the HTTPS origin in the Space secret `MI300X_URL` and set `MI300X_KEY` equal to `SERVER_KEY`.

## Director LLM (optional load)

- Default model: `HuggingFaceTB/SmolLM2-360M-Instruct` (overridable with `HF_DIRECTOR_MODEL`).
- Disable to save cold-start time: `export AIJOCKEY_USE_DIRECTOR_LLM=0` (deterministic JSON fallback).

## Observability (v1)

- JSON lines on stdout: `job_id`, `stage`, `ms` — grep or ship to your log stack later.
- `GET /health` — limits, disk free space estimate, `pipeline_locked`, `concurrent_denied_total`.
- `GET /ready` — lightweight readiness.

## Billing

Tear down the AMD instance when idle (hourly billing). See repo root [README.md](../README.md).

## Prometheus (deferred)

Metrics endpoint is not enabled in v1; extend `server/api.py` when you outgrow log-only ops.
