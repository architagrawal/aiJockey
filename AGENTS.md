# Guidance for coding agents (Claude Code, Cursor, etc.)

This document summarizes **recent architecture changes** and gives a **repeatable deploy playbook** so you can ship updates without re-reading the whole repo.

## What this stack does

AiJockey is an **offline-first DJ pipeline**: upload a pool of tracks → analyze (Demucs stems, beats, key, CLAP) → beam-search plan → execute transitions → master → encoded download.

**Inference-only HF integration (no custom model training):**

| Role | Implementation |
|------|----------------|
| **Director** | [`src/director.py`](src/director.py): small HF **instruct** LM produces JSON (`arc`, `text_prompt`, budgets, **`transition_tiers`** `major`\|`minor`, optional **`accent_hints`**). Disabled with `AIJOCKEY_USE_DIRECTOR_LLM=0` → deterministic fallback JSON. |
| **Tier → DSP** | [`src/transition_mapping.py`](src/transition_mapping.py): maps LLM tiers to concrete `transition_in` dicts consumed by [`src/execute.py`](src/execute.py). Planner’s rule-based technique choice is **replaced** after `plan()` for junctions ≥ 1. |
| **Genre / cohesion** | [`src/planner.py`](src/planner.py): `compute_pool_coherence()` from CLAP; high coherence tightens key/tempo weights (`PlannerConfig.pool_coherence`, `same_genre_tight_mix`). Timeline **`meta.max_stretch_ratio`** caps rubberband BPM ratio in execute (stricter when pool is heterogeneous). |
| **Accents** | Optional `accent_hint` on timeline entries; execute overlays samples at transition exits via `_overlay_accent_hint`. |
| **Adaptive master** | [`src/master.py`](src/master.py): trims hot peaks and eases glue compression when integrated loudness is already high. |
| **Optional generative bridge** | [`src/gen_bridge.py`](src/gen_bridge.py): stub only; enable later with `AIJOCKEY_MUSICGEN` — **not** wired into execute by default. |

## Deployment topology

```
Browser → Hugging Face Space (Gradio: space/app.py)
       → HTTPS tunnel URL (MI300X_URL secret)
       → AMD GPU VM: uvicorn server.api:app :8000
```

Heavy work (**never** on the Space CPU for real generation): analyze, optional Director LLM, plan, execute, master, ffmpeg encode.

**Human docs:**

- [`server/DEPLOY_AMD.md`](server/DEPLOY_AMD.md) — ROCm venv, SERVER_KEY, uvicorn, Director env
- [`server/TUNNEL_RUNBOOK.md`](server/TUNNEL_RUNBOOK.md) — what to do when free tunnel URLs rotate
- [`space/README.md`](space/README.md) — Space secrets (`ADMIN_PW`, `MI300X_URL`, `MI300X_KEY`) and UX limits

## Environment variables (AMD API host)

| Variable | Required | Default | Notes |
|----------|----------|---------|--------|
| `SERVER_KEY` | **yes** | — | Must match Space secret `MI300X_KEY`; sent as header `X-Key` on `/generate`. |
| `AI_DEVICE` | no | `cuda` | On ROCm PyTorch still use `cuda` string. |
| `AIJOCKEY_JOB_TIMEOUT_SEC` | no | `1200` | Wall-clock ceiling for one job (asyncio `wait_for` around the worker thread). Returns **504** if exceeded. |
| `AIJOCKEY_USE_DIRECTOR_LLM` | no | `1` | Set `0` to skip HF LLM load (faster cold start; fallback JSON). |
| `HF_DIRECTOR_MODEL` | no | `HuggingFaceTB/SmolLM2-360M-Instruct` | First request downloads weights; ensure disk and HF access. |
| `IDLE_FILE` | no | `/tmp/aijockey-last` | Touched on requests; optional external idle monitor. |

Optional: `HF_HOME` / `TRANSFORMERS_CACHE` if the instance has a large disk for model cache.

## How to run the API (from repo root)

```bash
python -m venv .venv && source .venv/bin/activate   # Linux / AMD
pip install -r requirements-rocm.txt   # or requirements-cpu.txt for smoke tests
pip install fastapi uvicorn python-multipart
export SERVER_KEY='...'
python -m uvicorn server.api:app --host 0.0.0.0 --port 8000
```

**Smoke checks**

```bash
curl -s "${URL}/health" | jq .
curl -s "${URL}/ready" | jq .
```

**Single-job behavior:** only one `/generate` runs at a time (`threading.Lock`). Second client gets **503** with `Retry-After: 120`. `GET /health` exposes `pipeline_locked` and `concurrent_denied_total`.

## `/generate` contract (for Spaces and clients)

**Method:** `POST /generate`  
**Auth:** header `X-Key: <SERVER_KEY>`  

**Multipart form (main fields):**

- `files`: 2–8 audio files; max **25 MB** each; extensions: wav, mp3, flac, m4a, ogg
- `preset`: key in `server.api.PRESETS` (e.g. `festival_inferno`) *or* supply custom `prompt` + `arc`
- `duration`: seconds, **30–600** (effective max may be lower without `use_library`)
- `use_library`: `true`/`false` — merges pre-analyzed clips from `clips/` + `cache/` if present
- `lufs`: float (e.g. `-9` club)
- `export_format`: `mp3` | `wav` | `flac`
- Optional: `prompt`, `arc`, `seed`

**Response:** binary audio; headers **`X-Job-Id`**, optional **`X-Ingest-Warnings`** (e.g. low MP3 bitrate).

**Failure codes:** `401` bad key, `400` validation, `503` busy, `504` timeout, `500` pipeline error.

## Hugging Face Space wiring

1. Set secrets: `MI300X_URL` (no trailing slash), `MI300X_KEY` (= `SERVER_KEY`), `ADMIN_PW` for Try It tab.
2. [`space/app.py`](space/app.py) POSTs to `{MI300X_URL}/generate` with **1200 s** client timeout; handles **503** (busy) and **504** (server timeout).
3. After changing tunnel URL, update `MI300X_URL` only (see tunnel runbook).

## Observability (v1)

- **Structured logs:** JSON lines on stdout via [`server/joblog.py`](server/joblog.py) (`job_id`, `stage`, `ms`, …).
- **Health:** `/health` includes limits, `disk_free_gb`, lock state; `/ready` for cheap probes.
- **Cleanup:** background task removes most per-job files shortly after success; see `server/api.py` `_delayed_cleanup`.
- **Cron example:** [`scripts/cron_health_example.sh`](scripts/cron_health_example.sh).

## Files to read when changing behavior

| Concern | Files |
|---------|--------|
| API surface, timeouts, lock, encoding | `server/api.py` |
| LLM Director schema / fallback | `src/director.py` |
| Tier → technique list | `src/transition_mapping.py` |
| Beam search + coherence + timeline `meta` | `src/planner.py` |
| Stretch cap + accents | `src/execute.py` |
| Loudness safety | `src/master.py` |
| Space UX | `space/app.py`, `space/README.md` |
| Local all-in-one Gradio | `app.py` |

## Common deploy mistakes

1. **`SERVER_KEY` unset** on AMD → API returns 500 “server key not configured”.
2. **Space `MI300X_KEY` ≠ `SERVER_KEY`** → 401.
3. **Tunnel URL rotated** (free tier) → Space “unreachable” until `MI300X_URL` updated.
4. **Two concurrent users** → second gets 503; expected for one GPU.
5. **Missing system deps** on Linux: `ffmpeg`, `rubberband-cli` (for pyrubberband), Demucs/ROCm stack per `requirements-rocm.txt`.
6. **First Director request slow** — model download + load; set `AIJOCKEY_USE_DIRECTOR_LLM=0` for demos if needed.

## Tests

- [`tests/test_transition_mapping.py`](tests/test_transition_mapping.py) — pure-Python tier mapping (no GPU).

When changing tier names or techniques, update `transition_mapping.py` and this test file together.
