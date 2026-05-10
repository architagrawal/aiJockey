# Guidance for coding agents (Claude Code, Cursor, etc.)

This document summarizes **recent architecture changes** and gives a **repeatable deploy playbook** so you can ship updates without re-reading the whole repo.

## What this stack does

AiJockey is an **offline-first DJ pipeline**: upload a pool of tracks → analyze (Demucs stems, beats, key, CLAP) → beam-search plan → execute transitions → master → encoded download.

**Inference-only HF integration (no custom model training):**

| Role | Implementation |
|------|----------------|
| **Director** | [`src/director.py`](src/director.py): HF **instruct** LM (text default Qwen2.5-7B / audio default Qwen2-Audio-7B; auto-routes by `audio_clip_paths` presence) produces JSON: `arc`, `set_narrative`, `narrative_notes`, `text_prompt`, `surprise_budget`, `callback_budget`, **`transition_tiers`**, **`transition_intents`** (per-junction WHY: `breath`/`build_tension`/`drop_payoff`/`genre_jump`/`callback`/`smooth_continue`/`cooldown`), optional **`accent_hints`**. Tier vocab: full = `minor`\|`major`\|`drop`\|`cut`\|`loop`. Phase 1 (`AIJOCKEY_PHASE=1`, default) restricts to `minor`\|`major`\|`drop`; `cut`/`loop` downgrade to `major`. Pool inventory ([`src/pool_intelligence.py`](src/pool_intelligence.py)) injected into prompt so Director sees what each clip IS (USER vs LIB, genre, BPM, section, energy). Disabled with `AIJOCKEY_USE_DIRECTOR_LLM=0` → deterministic fallback JSON (still emits narrative + intents). |
| **Pool intelligence** | [`src/pool_intelligence.py`](src/pool_intelligence.py): `tag_clip` per-clip metadata, `cluster_pool` (genre, bpm_band) groups, `coherence_score` 0..1, `summary_table` markdown for Director, `diagnose` returns verdict (`tight`/`mixed_navigable`/`disparate`) + narrative_advice + bpm_spread. Persists in `card.json` next to rendered mix. |
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
| `HF_DIRECTOR_MODEL` | no | text path: `Qwen/Qwen2.5-7B-Instruct`; audio path: `Qwen/Qwen2-Audio-7B-Instruct` | Default branches on whether `audio_clip_paths` are passed (e.g. `/generate` user uploads always pass them). First request downloads weights. Earlier `Qwen/Qwen3-8B-Instruct` default reverted — that ID does not exist on HF (verified 401). Real Qwen3 text models: `Qwen/Qwen3-4B-Instruct-2507` (small) or `Qwen/Qwen3-235B-A22B-Instruct-2507` (MoE, needs QLoRA on 192 GB VRAM). Set explicitly to test newer family. |
| `AIJOCKEY_PHASE` | no | `1` | `1` = Phase A polish (3-tier vocab, sample whitelist, stem-swap on); `2` = full vocab + meme samples. |
| `AIJOCKEY_PHRASE_QUANTIZE` | no | `1` | Snap segment boundaries to clip downbeats grid. |
| `AIJOCKEY_STEM_SWAP` | no | `1` | Stem-additive overlap path (eliminates subtractive vocal-mute phasing). |
| `AIJOCKEY_CONSTITUTIONAL` | no | `1` | Hard musical-rule validator + auto-repair on violations. |
| `AIJOCKEY_INSTRUMENTAL_ONLY` | no | unset | Per-request flag set by `/generate` `instrumental_only=true`. Drops vocals stem from full mix everywhere, not just overlap. Save/restore around request scope. |
| `AIJOCKEY_DEMUCS_MODEL` | no | `htdemucs_ft` | Demucs checkpoint name (`htdemucs`, `htdemucs_ft`, `mdx_extra_q`, ...). `_ft` is fine-tuned variant of `htdemucs` — same arch, ~0.5 dB SDR cleaner stems. |
| `AIJOCKEY_DEMUCS_OVERLAP` | no | `0.10` | Demucs window overlap. Lower = faster, slight artifact risk. Demucs default was `0.25`. |
| `AIJOCKEY_STEM_WORKERS` | no | `4` | Parallel rubberband workers per render_segment (= stem count). |
| `AIJOCKEY_RENDER_WORKERS` | no | `2` | Parallel timeline-segment renders. Each ~200MB PCM peak; raise on big-RAM hosts. |
| `AIJOCKEY_RB_COMBINED` | no | `1` | Single-call rubberband (`--tempo` + `--pitch` in one subprocess). Falls back to two-pass on failure. |
| `AIJOCKEY_RUBBERBAND_BIN` | no | `rubberband` | Path to `rubberband` CLI binary. |
| `AIJOCKEY_BATCH_CLAP` | no | `1` | Stage 1 batched CLAP forward across pending clips. |
| `AIJOCKEY_CAPTION_BATCH` | no | `8` | Stage 3 caption batch size for Qwen2-Audio. |
| `AIJOCKEY_DTYPE` | no | `bfloat16` | Mixed precision dtype for training/inference. |
| `AIJOCKEY_COMPILE` | no | `1` | torch.compile on model forward paths. |
| `AIJOCKEY_COMPILE_MODE` | no | `default` | `default`/`reduce-overhead`/`max-autotune`. ROCm 6.0 HIP graphs less stable — `default` safer. |
| `AIJOCKEY_FLASH_ATTN` | no | `1` | HF transformers `attn_implementation`: `0`=eager, `1`=sdpa, `2`=flash_attention_2. **`2` requires ROCm-CK fork of flash-attn — stock `pip install flash-attn` does NOT build on ROCm.** |
| `AIJOCKEY_QLORA` | no | `0` | 4-bit QLoRA load for big-model fine-tune. Needs `bitsandbytes-rocm` on AMD. |
| `AIJOCKEY_INT8` | no | `0` | 8-bit weight quant for inference. Needs `bitsandbytes-rocm` on AMD. |
| `AIJOCKEY_OPTIMIZER` | no | `lion` | `lion`/`sophia`/`adamw8bit`/`adamw`. |
| `AIJOCKEY_PREF_METHOD` | no | `orpo` | Preference training method: `orpo`/`dpo`/`kto`. |
| `AIJOCKEY_SCRATCH` | no | `/scratch` | Pipeline scratch root (S0–S9 stage queues). |
| `AIJOCKEY_BEAT_THIS` | no | `1` | Beat-This! GPU joint beats+downbeats. Falls back to madmom DBN (preferred) or librosa heuristic. |
| `AIJOCKEY_BEAT_THIS_DBN` | no | `auto` | `auto` picks DBN postprocessor when madmom importable, else minimal (with sanity guards). |
| `AIJOCKEY_BEAT_THIS_CKPT` | no | `final0` | Beat-This! checkpoint name. |
| `AIJOCKEY_BEAT_THIS_TEMPO_CAP` | no | `220.0` | Reject beat_this output above this BPM (sanity guard against minimal postprocessor over-detection). |
| `AIJOCKEY_BS_ROFORMER` | no | `0` | Opt-in vocal stem swap to BS-Roformer / Mel-Band Roformer. Requires `_CKPT` path. |
| `AIJOCKEY_BS_ROFORMER_CKPT` | no | `` | Path to BS-Roformer vocals checkpoint (.ckpt/.pth). |
| `AIJOCKEY_BATCH_DEMUCS` | no | `1` | Stage 1 batched Demucs across N clips per forward (lifts MI300X GPU util from 4% → 95%). |
| `AIJOCKEY_DEMUCS_BATCH` | no | `4` | Demucs micro-batch size for `Analyzer.stems_batch`. |
| `AIJOCKEY_CLAP_CHUNK` | no | `32` | Chunk size for `get_audio_embedding_batch` to avoid OOM on 100+-clip batches. |
| `AIJOCKEY_PROBE_LOG` | no | `/scratch/probes/log.jsonl` | Per-render JSONL append path for probe data + DPO accumulator. Falls back to `./probes_log.jsonl` on laptop. |
| `AIJOCKEY_IMPROVER_ENERGY` | no | `1` | Probe-driven planner energy re-pick action (writes `_improver_repick` flag, planner re-picks via `repick_energy_constrained_segments`). |
| `AIJOCKEY_IMPROVER_OVERLAP` | no | `1` | Probe-driven overlap-shorten action (halves `transition_in.bars` when phase_delta > 0.25). |
| `AIJOCKEY_IMPROVER_SWAP` | no | `1` | Probe-driven vocal-suppress technique swap action (drum_break/filter_fade/eq_swap when xcorr > 0.45). |
| `AIJOCKEY_COHORT` | no | `` | Human-readable cohort label stamped into probe_log row for A/B bucketing (e.g. `A`/`B`/`C`/`D`). |
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
- `mix_mode`: `tight` (user clips only) | `balanced` (default; library fills 30% headroom) | `exploratory` (heavier library augmentation)
- `library_role` (optional): `any` | `fill_gaps` | `warmup_outro` | `bridges_only`
- `instrumental_only`: `true` (default) — drops vocals stem from full mix everywhere; toggle off for vocal mixes
- `lufs`: float (e.g. `-9` club)
- `export_format`: `mp3` | `wav` | `flac`
- Optional: `prompt`, `arc`, `seed`

**Audio Director caveats** (auto-engaged when files uploaded):
- Default model: `Qwen/Qwen2-Audio-7B-Instruct` (~14 GB first-run download)
- Hard cap: model hears **first 6 user clips × 30 s window each** (memory bound). Pools >6 clips: extra clips planned via metadata only, not audibly grounded
- Override with `HF_DIRECTOR_MODEL=Qwen/Qwen2.5-7B-Instruct` for text-only path (no audio understanding, faster cold start)

**Response:** binary audio; headers **`X-Job-Id`**, **`X-Mix-Mode`**, **`X-Clips-Used`** (JSON `{user_count, library_count, library_ids[]}`), optional **`X-Ingest-Warnings`** (e.g. low MP3 bitrate).

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
