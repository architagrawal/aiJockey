# AiJockey — Status, Progress, Plan, Bugs

Snapshot of what works, what's broken, what's next.
Last updated: 2026-05-09 (post live MI300X smoke + render-path stabilization on `best-output-pipeline`).

---

## Session update — render path stabilization + pool intelligence + mix_mode UX (2026-05-09 evening)

Branch `best-output-pipeline` taken from "scaffolded but broken on real clips" → "ships intelligible mixes from arbitrary user clips on MI300X". 9 critical render bugs fixed (overlap math, tier propagation, segment min-length, htdemucs_ft compile crash). Director wiring now accepts pool inventory + emits set narrative + per-junction intent. Library augmentation gets semantic mix_mode UX with CLAP-similarity picks.

### MI300X bring-up — done

- Snapshot `aijockey-slim` (109.7 GB atl1) pulled into a fresh AMD Developer Cloud MI300X 1× ($1.99/hr).
- Container env: `rocm:latest` with PyTorch 2.9 nightly + ROCm 7.0, /workspace bound to /root/aijockey, /cache (snapshot's 6.2 GB analyzed pool of 103 clips) bound in.
- Deps installed: `lion-pytorch peft trl datasets accelerate laion-clap pyrubberband demucs h5py torchlibrosa ftfy starlette anyio pydantic fastapi<0.115`.
- Stock `flash-attn` not built (ROCm — opt-in via CK fork only). `bitsandbytes` not installed (vanilla CUDA-only; bnb-rocm fork required for QLoRA/INT8).
- 14/14 unit tests pass on GPU.
- 10+ smoke renders proven end-to-end (analyze → plan → execute → master).

### Render path bugs fixed (live-discovered + repaired)

| Commit | Bug | Fix |
|--------|-----|-----|
| `5b4f9cd` | `tier_to_technique` discarded `tier` field → constitutional couldn't see `drop` tier | propagate tier into returned technique dict |
| `5b4f9cd` | Planner `pick_segment` last-resort returned 0.5–2 s segments → consumed entirely by overlap | `enforce_min_segment_length()` extends short segments to N bars via downbeats |
| `759aca7` → `9b74485` | Overlap math shadowed: `apply_transition` clamped `overlap_n` for vocal mute but `crossfade_transition`/`eq_swap_transition` recomputed overlap from raw `bars` arg → consumed full segment | shadow `bars = overlap_n // bar_samples` so primitives receive clamped value |
| `0c844ef` | Half-segment overlap still consumed too much → output stuck at longest-segment length | clamp = min(shorter_side / 3, 8 bars) |
| `6c4ba97` | min_bars=16 over-extended segments planner had legitimately picked short → 238 s for 120 s target | min_bars=8, only patch truly tiny |
| `7c486cb`, `be9a128` | `estimate_max_transitions_for_pool` returned 64 for 103-clip pool → LLM context overflow → fallback fired | cap at 16, formula `max(8, duration/20)` |
| `4b592ad` | Tokenizer `max_length=2048` truncated chat-template tail with pool inventory injected → LM continued user text instead of answering | bump to 8192 |
| `aeb75ac` | `htdemucs_ft` (BagOfModels) crashes after `torch.compile` — wrapper hides `.segment` attr → `apply_model` raises `AttributeError` | skip `maybe_compile` for `BagOfModels`; bf16 autocast still applies |
| `f68b62b` | Default `Qwen3-8B-Instruct` returned `RepositoryNotFoundError` (model does not exist on HF) | revert to `Qwen2.5-7B-Instruct`; document real Qwen3 IDs (`Qwen3-4B-Instruct-2507`, `Qwen3-235B-A22B-Instruct-2507`) |

After all fixes: render duration matches plan ±10 %, Director runs without fallback, audio output not abrupt within constraints of pool coherence.

### Pool intelligence + Director narrative (`a71876b`)

`src/pool_intelligence.py` (new, ~200 lines):

- `tag_clip(meta)` — per-clip dict {clip_id, source, genre, bpm, bpm_band, key, section_top, energy, duration, has_vocals}
- `cluster_pool()` — group by (genre, bpm_band)
- `coherence_score()` — mean cosine sim of CLAP embeddings to pool centroid (0..1)
- `summary_table()` — markdown table of pool with USER/LIB column for Director consumption
- `diagnose()` — verdict (`tight` / `mixed_navigable` / `disparate`) + narrative_advice + bpm_spread + cluster summary
- `pick_coherent_subset()` — auto-curate when pool too wide (deferred wiring)

Director (`src/director.py`):

- `SYSTEM_PROMPT_PHASE1` rewritten — workflow demands `set_narrative`, per-junction `transition_intents`, honest `narrative_notes` when pool disparate. 7 intent categories: `breath` / `build_tension` / `drop_payoff` / `genre_jump` / `callback` / `smooth_continue` / `cooldown`. 3 reasoning examples.
- `run_director(clips_meta=...)` injects pool inventory + diagnose() output into LLM prompt. Director reasons about what pool actually contains.
- `_sanitize_out` validates set_narrative + transition_intents (defaults inferred per tier when LLM omits).
- `_fallback_director` also produces narrative + intents for deterministic path.
- Per-junction accent cap (≤2 in Phase 1) at sanitize time.
- Style-RAG few-shot block prepended when `/scratch/embed/` index populated.

Planner (`src/planner.py`):

- `apply_llm_transition_tiers_to_timeline(transition_intents=...)` propagates intent into `transition_in['intent']`.
- Constitutional-violation penalty in `plan_n_best` (rejects subtract 0.5 from candidate score).
- Planner-stage phrase quantize (defense in depth above execute-stage).

`src/main.py` saves `card.json` next to rendered mix with: set_narrative, narrative_notes, arc, pool diagnostic, per-junction (tier, intent) plan.

### mix_mode + library augmentation (`bd30458`)

UX evolution: replace raw `library_ratio` knob with semantic mode.

- `/generate` form fields: `mix_mode` (`tight` / `balanced` (default) / `exploratory`) + `library_role` (optional advanced: `any` / `fill_gaps` / `warmup_outro` / `bridges_only`).
- `tight` forces `use_library=False` regardless of toggle.
- `lib_count_for_mode(mode, user_count, user_total_dur, target_duration)` — sweet spot moves with all four. balanced caps at LIBRARY_MAX_PICK//2, exploratory at LIBRARY_MAX_PICK.
- `_user_pool_centroid()` + `library_clip_paths_clap()` — CLAP-cosine retrieval against library cache, BPM ±15 % filter. Replaces alphabetical `library_clip_paths()` when CLAP cache available; alphabetical fallback when not.
- User clips tagged `source='user'` in their analyzed cache JSON; library clips tagged `source='library'` on link (rewrites symlink to concrete file before edit).
- Constitutional new rule `check_user_clip_floor` — warns if any source=='user' clip missing from timeline (severity warn, planner re-pick rather than reject).
- Director system prompt has USER-VS-LIB POLICY section: every USER clip MUST appear at least once; LIB supports as bridges/warmup/cooldown.
- Response headers: `X-Job-Id`, `X-Mix-Mode`, `X-Clips-Used` (`{user_count, library_count, library_ids[]}`), optional `X-Ingest-Warnings`.
- jlog `library_pick` event with mode/role/user-count/lib-count for observability.

### Smoke quality progression (test1.wav + test2.wav user pool)

| Run | Render | Director | Verdict |
|-----|--------|----------|---------|
| smoke.wav | 35.7 s / 90 s target | off | render duration shortfall — bars consumption bug |
| smoke_d10.wav | 160 s / 120 s target | Qwen2.5-7B | post overlap fix |
| smoke_long.wav | 279 s / 360 s target | Qwen2.5-7B | longer narrative space, all-minor (overcautious on disparate library pool) |
| test_3min_v2.wav | 136 s / 180 s target | **fallback** (Qwen3-8B 401) | DSP good, no narrative |
| **test_3min_v3.wav** | **136 s / 180 s target** | **Qwen2.5-7B (working)** | narrative: "navigate disparate pool as a curated journey rather than forcing build", per-junction intents present, htdemucs_ft + stem-swap + phrase-quantize all active |

### Self-critique loop — designed, not built

Discussed 1.5-pass design (audio critic → rule-based improver → surgical re-render). Decision: defer to post-Beat-This! since most issues a critic would catch are upstream artifacts (downbeat heuristic). Build order:

1. **Beat-This!** — fixes phrase-grid drift (downbeat heuristic currently `beats[::4]`); ~30 % of post-render issues vanish
2. Cheap audio probes (RMS env, xcorr, spectro diff) — numpy-only, catches 70 % of artifacts at 1 % cost of LLM critic
3. Wire CriticV2 (S4 trained) into `/generate` as global score gate
4. Rule-based improver: 3-5 issue types → deterministic timeline edits
5. Audio-LLM critic (Qwen2-Audio JSON output) as **fallback**, not primary; opt-in flag
6. Surgical re-render with cascade (segment + 2 transitions)
7. Skip 2nd critique pass

---

## Session update — code review + perf + ROCm hardening (2026-05-09 morning)

Branch `best-output-pipeline` reviewed end-to-end. 9 correctness bugs, 6 ROCm compat issues, 9 perf wins, audio-Director wiring, and 2 model swaps applied. All syntax-clean. Render path independent of training pipeline — usable on user clips without library/self-play.

### Correctness fixes
| File | Bug | Fix |
|------|-----|-----|
| `server/api.py:549` | `AIJOCKEY_INSTRUMENTAL_ONLY` env mutation outside try/finally — crash leaves process stuck | save/restore around `_run_generate_sync` call |
| `scripts/stage7_dpo.py:65` | non-atomic append race with S5 → duplicate DPO pairs | rewrite-with-atomic_write + dedupe pre-existing rows |
| `src/execute.py:431,441` | `sum(generator)` → `int 0` on empty stems → `stem_swap_transition(0,...)` numpy crash | use `_stem_sum(...,('drums','bass','other'))`; None-guard |
| `src/execute.py:227` | `inst[:,-(n+r):-n]` ramp slice unguarded → shape mismatch | clamp ramp_n by inst/vox actual lengths |
| `scripts/pipeline/common.py:85` | `watch()` adds `seen` after yield → consumer crash drops file forever | only mark seen when no done_marker; with marker, marker is source of truth |
| `src/director.py:333,403` | system prompt sent twice (concat + system role) → 2048-token truncation | drop concat at both call sites |
| `src/training/augment.py:51` | `speed_perturb` was no-op (returned input + new_sr) | actual `librosa.resample` with safe fallback |
| `tests/test_samples_gating.py:44` | Phase2 test silently passes if `airhorns` absent from SYNTHESIZERS | precondition assert + module reset |
| `scripts/pipeline_launch.sh:28` | unquoted `ENV_PREFIX`/`PY` in tmux send-keys breaks on space-paths | `printf %q` everywhere |

### ROCm/AMD compat
| Concern | Fix |
|---------|-----|
| `attn_implementation='flash_attention_2'` default kills model load on stock ROCm wheel | `efficiency.py:hf_attn_implementation()` default → `'sdpa'`; FA2 opt-in via `AIJOCKEY_FLASH_ATTN=2` after building ROCm-CK fork |
| `torch.compile(mode='reduce-overhead')` brittle on ROCm 6.0 HIP graphs | default → `'default'`; `AIJOCKEY_COMPILE_MODE` override |
| `int8_quant_config()` always returned BNB config (CUDA-only) | env-gated by `AIJOCKEY_INT8` |
| `pipeline_launch.sh` set `AIJOCKEY_FLASH_ATTN=2` | downgraded to `=1` (sdpa) |
| `transformers>=4.30` too loose for Qwen2-Audio + ORPO | bumped to `>=4.43.0` (bump again to `>=4.51.0` when adopting Qwen3 family) |
| `requirements-rocm.txt` missing training deps | added `accelerate`, `peft`, `trl`, `datasets`, `lion-pytorch`; documented `bitsandbytes-rocm` + ROCm flash-attn as opt-in |

### Performance (already in branch + this session)
- **`load_stems` parallel I/O** — 4 stem WAVs concurrently via ThreadPoolExecutor (~3× I/O step)
- **`render_segment` stem-parallel rubberband** — 4 stems stretched/pitched concurrently (pyrubberband releases GIL) → ~3-4× CPU util on hot path
- **Timeline render parallelism** — `_RENDER_WORKERS=2` default, env-tunable; ~2× wall on N-segment mix
- **Free-as-you-go memory** — `rendered[i-1] = None` after consumed; ~200MB/segment reclaimed
- **Single-call rubberband** — `_rubberband_combined` calls binary once with `--tempo` + `--pitch` flags; cuts subprocess spawns + tempfile WAV roundtrips in half. Falls back to two-pass on failure. Disable: `AIJOCKEY_RB_COMBINED=0`
- **Demucs perf** — `overlap=0.10` (was 0.25, ~15% faster); bf16 autocast on GPU (~1.5-2× throughput on MI300X); compile mode default → `default` (ROCm-safe)
- **cuDNN/MIOpen autotune** — `cudnn.benchmark=True` + `set_float32_matmul_precision('high')` at execute module init
- **Director inference** — `inference_mode()` instead of `no_grad()`; bf16 dtype on bf16-capable GPUs (auto-detect via `torch.cuda.is_bf16_supported()`)
- **CLAP batched embeddings** — new `get_audio_embedding_batch(audios)` in `clap_wrapper.py`; `Analyzer.analyze` accepts `precomputed_clap=`; S1 `process_batch` pre-loads 48k mono for all pending clips, single batched CLAP forward, then per-clip demucs/beats. ~Nx CLAP step speedup. Disable: `AIJOCKEY_BATCH_CLAP=0`
- **Caption batched** — `_Captioner.caption_batch(paths)` runs single Qwen2-Audio forward across N clips; per-row decode via attention_mask sums; ~5-8× throughput at batch 8. Tunable `--caption-batch-size` / `AIJOCKEY_CAPTION_BATCH`

### Audio-aware Director wired into request path
- Previously: `_call_qwen2audio` existed but `/generate` never passed audio paths → text Director only
- Now: `server/api.py:566` and `app.py:105` pass `audio_clip_paths=[saved_paths]`; Director hears every user upload (capped at 6 internally, 30s window each)
- `HF_DIRECTOR_MODEL` defaults to `Qwen/Qwen2-Audio-7B-Instruct` when audio paths present, else text Director

### Model swaps (free OSS upgrades)
| Component | Old | New | Effort | Knob |
|-----------|-----|-----|--------|------|
| Text Director default | `Qwen2.5-7B-Instruct` | **`Qwen3-8B-Instruct`** | 1 line | `HF_DIRECTOR_MODEL` |
| Stem separator default | `htdemucs` | **`htdemucs_ft`** (~0.5 dB SDR cleaner) | 1 line | `AIJOCKEY_DEMUCS_MODEL` |

### Recommended next swaps (not yet applied)
| # | Swap | Effort | Reason |
|---|------|--------|--------|
| 1 | madmom → **Beat-This!** | ~30 lines | librosa fallback `downbeats=beats[::4]` is silently corrupting phrase quantize; joint beats+downbeats from transformer = direct fix to abruptness complaints |
| 2 | htdemucs_ft → **BS-Roformer / Mel-Band Roformer** for vocals | ~80 lines | vocals SDR ~9 → ~11 dB; cleaner stem-swap + mashup; license MIT |
| 3 | CLAP → **MuQ** for music similarity (retrieval only) | ~100 lines | MuQ Apache 2.0 (MERT is CC-BY-NC); marginal vs CLAP — defer until library retrieval matters |
| 4 | Qwen2-Audio → **Qwen2.5-Omni-7B** | ~50 lines | only when audio Director becomes critical path |

### New env knobs introduced this session
```
AIJOCKEY_STEM_WORKERS      1-4    (default 4)   parallel rubberband across stems
AIJOCKEY_RENDER_WORKERS    1-8    (default 2)   parallel timeline-segment renders
AIJOCKEY_DEMUCS_OVERLAP    0-0.5  (default 0.10) demucs window overlap
AIJOCKEY_DEMUCS_MODEL      str    (default htdemucs_ft) demucs checkpoint
AIJOCKEY_RB_COMBINED       0|1    (default 1)   single-call rubberband
AIJOCKEY_RUBBERBAND_BIN    path   (default 'rubberband') CLI binary path
AIJOCKEY_COMPILE_MODE      str    (default 'default') torch.compile mode
AIJOCKEY_INT8              0|1    (default 0)   int8 inference (needs bnb-rocm)
AIJOCKEY_BATCH_CLAP        0|1    (default 1)   S1 batched CLAP
AIJOCKEY_CAPTION_BATCH     int    (default 8)   S3 caption batch size
```

### Architectural note — user/library mix direction
Branch supports `use_library` boolean. Recommended UX evolution (not yet implemented):
- Replace explicit ratio with semantic mode: `tight` / `balanced` / `exploratory`
- Backend computes effective lib_count from `(mode, user_count, total_user_dur, target_duration)`
- Hard floor: every user clip MUST appear in timeline at least once (planner reserved slots)
- Library picks via CLAP-similarity to user pool centroid (style-RAG already plumbed)
- Response telemetry: `clips_used: {user: N, library: M, library_ids: [...]}`

`server/api.py` partially has `lib_count_for_mode()` helper. Wire to `mix_mode` form field next.

### MI300X bring-up order (when moving to GPU)
1. `rocm-smi` + `python -c "import torch; print(torch.cuda.is_available())"` — baseline
2. `pip install -r requirements-rocm.txt`
3. `python scripts/00_rocm_sanity.py` — must pass
4. Smoke test: single `/generate` end-to-end on 3 user clips (forces audio Director download ~16GB first time)
5. `pytest tests/` — regression
6. Profile serial vs parallel (vary `AIJOCKEY_RENDER_WORKERS`); watch `rocm-smi` for VRAM headroom <80%
7. Tune knobs (overlap, FLASH_ATTN, COMPILE)
8. `bash scripts/pipeline_launch.sh` only after smoke passes
9. Optional: ROCm flash-attn CK fork → `AIJOCKEY_FLASH_ATTN=2`; bnb-rocm → `AIJOCKEY_QLORA=1`

---

## Phase A polish branch — completed before MI300X spin-up

- **Foundation fixes**:
  - Phrase quantization (planner-stage + execute-stage defense in depth) — [src/planner.py](src/planner.py), [src/execute.py](src/execute.py)
  - Stem-additive overlap (eliminates phasey vocal-mute artifact, STATUS bug #6) — [src/execute.py](src/execute.py)
  - Humanized accent FX (±8 ms timing jitter, ±8 % velocity, deterministic seed) — [src/execute.py](src/execute.py)
  - Sample lib gating (PHASE1_ALLOWED_TYPES whitelist; meme/airhorn pollution removed) — [src/samples.py](src/samples.py)
- **Phase 1 scope enforcement**:
  - 3-tier vocab (`minor`/`major`/`drop`) — Director downgrades cut/loop to major
  - 3 arc presets (`build`/`peak`/`flat_low`) — full vocab restored when AIJOCKEY_PHASE=2
  - Director SYSTEM_PROMPT_PHASE1 with section-aware drop rule + accent budget
  - Director-level accent cap (≤2 per junction)
  - API: min 3 clips Phase 1, BPM-band post-analyze warning, instrumental_only=True default
- **Hard musical rules**: [src/constitutional.py](src/constitutional.py) validator (phrase-grid, drop-section, breakdown-pair, bpm-drift, key-compat, accent-budget) + repair() downgrade. Wired into execute + N-best ranker.
- **Style-RAG**: Director retrieves nearest CLAP-similar real DJ transitions from /scratch/embed index, prepends as in-context examples — [src/style_rag.py](src/style_rag.py)
- **Efficiency hooks**: [src/training/efficiency.py](src/training/efficiency.py) — bf16, torch.compile, Flash-Attn 2, QLoRA, LoRA, Lion/Sophia/AdamW8bit, ORPO/DPO/KTO trainer factory, int8 inference. Retrofitted into analyze.py.
- **Augmentation utils**: [src/training/augment.py](src/training/augment.py) — AugChain (pitch/stretch/gain/codec roundtrip), SpecAugment, mixup. ~5-10× effective dataset.
- **Pipeline-parallel scripts (S0–S9 + monitor + launcher)**: [scripts/stage*.py](scripts/) — all stages have real bodies (no stubs). Mixotic + FMA + MTG-Jamendo URL listers, batch-mode analyze, novelty-curve transition segmentation, Qwen2-Audio captioner, CLAP-feature critic v2 with multi-task heads, K-render self-play with critic rerank, ORPO LoRA Director with director.json text pairs, MusicGen-Small bridge fine-tune with EnCodec recon loss, K=64 final batch render.
- **Seed prompts**: [scripts/prompts/selfplay.json](scripts/prompts/) (30) + final.json (20) ready to feed S5/S9.
- **Tests**: 14/14 torch-free unit tests pass (phrase quantize, constitutional, transition mapping Phase 1/2 vocab gates).
- **Doc sync**: AGENTS.md tier vocab + 12 env vars; HACKATHON.md Slide 4 reflects current pipeline.

Branch ready for MI300X. Pull with `git pull origin best-output-pipeline` then `bash scripts/pipeline_launch.sh`.

---

---

## Done in current session

### Infrastructure

- **AMD Developer Cloud** account + billing + API token + SSH keys.
- **GPU droplet** `aijockey-mi300x` (MI300X 1×, 192 GB VRAM, 240 GB RAM, 720 GB NVMe boot, 5 TB NVMe scratch). Per-second billing, $1.99/hr.
- **Container env**: PyTorch 2.6 / ROCm 7.0 image with running `rocm` Docker container at `/workspace`. PyTorch 2.9 nightly + ROCm 7.0.51831 inside. CUDA-API mapped to ROCm — code uses `torch.cuda.is_available()` unchanged.
- **Snapshot**: 113 GB at `aijockey-prebake-v1`, ~$6.78/mo. Captures clip pool, cache, all checkpoints, 25-set DJ corpus, meme library.
- **Idle/hard-cap auto-destroy** scaffolded in `server/boot.sh` (DO API token + droplet ID required).

### Data

- **Curated clip pool**: ~103 audio files spanning EDM, house, techno, trance, dubstep, future-bass, dnb, chillstep, hip-hop, trap, bollywood, punjabi, ghazal, lofi, ambient, cinematic, synthwave, retrowave, disco, drill, Hindustani classical instrumental (sitar, bansuri, sarod, tabla, violin), Indian fusion. Stored in `clips/` (gitignored, ~5 GB on droplet).
- **Meme/SFX library**: 34 short one-shots in `samples/memes/` (airhorns, vine-boom, scratches, dhol/tabla hits, party_horn, drama chord, etc.). Manifest auto-built. Wired into `SampleBank`.
- **DJ-set corpus**: 25 long-form sets (~20 GB raw → ~2.5 GB mp3 192k) in `datasets/dj_sets_mp3/`. Genres mirror clip pool. Used for mix-critic positives.

### Pipeline / DSP

- **Analyze**: parallelized via multiprocessing (`--workers N`). Demucs htdemucs stems on GPU. CLAP via `transformers` (`laion/clap-htsat-unfused`). librosa beat-track fallback (madmom Python 3.10 incompatible — patched by switching to librosa). Backfilled `vocal_activity` per section.
- **Planner**: beam search + multi-tier scorer; pool coherence; CLAP compat head projection; arc shapes; text-prompt CLAP cosine bias; min_unique_clips hard-enforced.
- **N-best rerank** (`plan_n_best`): generates 3-5 candidate timelines varying surprise/beam, reranks by heuristic (CLAP coherence + vocal-collision + duration match + bpm strain).
- **Vocal-aware planning**: `pick_segment(prefer_instrumental=True)` for transition boundaries.
- **Dual-side vocal suppression in `execute.py`**: outgoing + incoming vocals zeroed during overlap window for crossfade/eq_swap/filter_fade/echo_out, ramped in/out post-overlap.
- **Execute**: 15 transition techniques, rubberband stretch (capped), accent_hint overlay, sample bank with synth fallback.
- **Master**: HP30 → multiband + glue compression → LUFS norm → true-peak limiter. Adaptive (eases compression on already-hot inputs).

### LLM Director (multi-stage)

- **Director model**: Qwen2-Audio-7B-Instruct (multimodal — actually hears each clip's first 30 s before producing JSON plan). Fallback to Qwen2.5-7B-Instruct (text only).
- **5-tier transition vocabulary**: `minor`, `major`, `drop`, `cut`, `loop`. Each tier maps to genuinely distinct DSP techniques (no eq_swap collapse).
- **Director system prompt**: Tomorrowland-grade drama policy + tier-distribution rules per arc + 4 few-shot examples. Outputs arc, refined prompt, surprise/callback budgets, transition_tiers, accent_hints, same_genre_tight_mix.
- **Accent hints**: Director can request riser / impact / snare_roll / sweep / hihat_roll FX overlays at specific junctions; execute lays them down via SampleBank.

### Trained custom weights (on MI300X)

- **Tier 1 classifier** (`checkpoints/technique_classifier.pt`) — MLP over synthetic transition features. Replaces rule-based technique pick (when wired in).
- **Tier 1.5 CLAP compat head** (`checkpoints/clap_compat_head.pt`) — InfoNCE-trained 512→128 projection. Val accuracy 0.844. Used in planner to rank pair compatibility in DJ-mix space.
- **Mix critic** (`checkpoints/mix_critic.pt`) — small CNN over log-mel; real-DJ-set windows vs random-splice. Val acc 0.775. Currently NOT used in rerank (limited reliability).
- **Tier 2 RLHF / DPO**: not yet trained. Pairwise preference labels would unlock this.
- **MusicGen / Stable Audio bridge**: stub only (`src/gen_bridge.py`); audiocraft install fails on ROCm — deferred.

### Frontend / API

- **FastAPI server** (`server/api.py`): `/health`, `/ready`, `/library/info`, `/demos/{slug}.mp3`, `/generate`. Multipart uploads, X-Key auth, idle ping, structured `joblog` lines, in-flight cache by content hash. Director LLM gated by `AIJOCKEY_USE_DIRECTOR_LLM`. Tier mapping gated by `AIJOCKEY_APPLY_LLM_TIERS`.
- **HF Space Gradio app** (`space/app.py`): public Demo tab (5 pre-baked MP3s), password-gated Try It tab (upload 2-8 clips, 5 vibe presets, length slider min-30s/max=Σ-clip-duration, library toggle, advanced overrides for prompt/arc/seed/LUFS).
- **CLI** (`src/main.py`): `analyze`/`plan`/`execute`/`master`/`eval`/`all` with `--workers`, `--n_best`, `--use_director`, `--apply_llm_tiers`, `--compat_head`.
- **Deploy docs**: `server/DEPLOY_AMD.md`, `server/TUNNEL_RUNBOOK.md`, `AGENTS.md`.

### Tests

- `tests/test_transition_mapping.py` — unit coverage for tier→technique mapping.

---

## Generated mixes (this session)

| # | Pool | Director | Tiers | Key change | Outcome |
|---|------|----------|-------|------------|---------|
| Pre-baked 5 | 100 clip lib | none | none | initial demos | Acceptable per critic, mixed user verdict |
| v1-v3 | 2 test | none | none | vocal suppression tuning | Degenerate (1-clip output) |
| v4 | 2 test | none | none | + planner fix (multi-clip) | 2 clips, echo_out, 75 s |
| v5 | 10 mixed | none | none | + N-best | 4 clips, varied techniques |
| v6 | 10 mixed | none | none | + dual vocal suppression + vocal-aware planner | 6-entry, callback structure |
| v7 | 2 test | none | none | full latest stack | 2 clips, echo_out, 75 s |
| v9 | 2 test | SmolLM-360M | minor only | LLM Director | Director regressive — boring eq_swap only |
| v10 | 10 mixed | none | none | minimal flags + arc=build | User-acceptable; varied transitions |
| v11 | 10 mixed | SmolLM | minor | Director on big pool | Worse than v10 |
| v12 | 10 mixed | Qwen2.5-7B | mostly minor | Qwen text Director | Slight improvement |
| v13 | 10 mixed | Qwen2.5-7B + better prompt | minor | + Tomorrowland prompt | Same as v12 |
| v14 | 10 mixed | Qwen2-Audio-7B | minor | Multimodal Director hears clips | Audio path live, output collapses to eq_swap |
| **v15** | 10 mixed | Qwen2-Audio + 5-tier vocab | drop/cut/major/loop mixed | Final current | Real diversity: drum_break / filter_fade / echo_out / cut |

---

## Current pros

- **End-to-end runs on MI300X** with no code changes vs CUDA reference.
- **Multimodal Director hears clips** before planning. Genuine audio-aware control.
- **5-tier transition vocabulary** produces actual variety per junction.
- **Vocal-collision avoidance** at both planner level (instrumental section pick) and execute level (subtractive vocal mute on overlap).
- **Director-suggested accent FX** layered correctly via sample bank.
- **N-best heuristic rerank** picks longer / more-coherent / lower-collision candidate from N variations.
- **Three custom-trained checkpoints** (Tier 1 classifier, Tier 1.5 CLAP head, mix critic) all live on MI300X.
- **Real DJ-set corpus** (25 sets, 20 GB raw) ready for further training.
- **Snapshot insurance** (~$7/mo) means fast restore — no re-setup cost.
- **Cost discipline**: ~$15-20 spent on droplet so far out of $100 credit.

---

## Current bugs / limitations

### High priority

1. **Render duration shortfall**: timelines plan 180-220 s but `execute.py` writes 60-90 s. Overlap windows (`bars * 4 * beat_dur`) consume each segment by ~12-24 s × N transitions, faster than segments grow. Need: shorter transition bars on shorter segments OR extend segment selection to compensate.
2. **User-perceived audio quality**: even v15 still sounds mechanical to careful listening. Symptoms: jarring clip changes, vocal-suppression phasing artifacts, accent FX sometimes mis-timed.
3. **Mix critic unreliable**: 0.77 val accuracy on real-vs-splice doesn't translate to "sounds good to humans". Codec bias (mp3 positives, wav negatives), short window (8 s), tiny dataset (1.6k samples).
4. **2-clip pool fundamentally limited**: any DJ AI struggles. Fix at API: auto-enable `use_library=true` when ≤3 user clips.

### Medium priority

5. **Eq_swap dominance pre-fix**: pre-v15 tier mapping collapsed major-tier to eq_swap. Fixed in `transition_mapping.py` — now major picks from {filter_fade, drum_break, silence_drop, echo_out}. Validate.
6. **Vocal-suppression artifacts**: subtractive Demucs-stem mute leaves phasey holes since stems aren't bit-perfect. Better: use actual stems instead of full mix during overlap (mix drums+bass+other only).
7. **Transition technique inappropriate for material**: `pitch_bend`, `loop_callback`, `spinback`, `scratch_fill` sound amateurish on full vocal pop songs. Restrict to instrumental sections only.
8. **Director at minor-bias on conservative prompts**: when user says "smooth groove", Qwen correctly picks all-minor → boring. Need: arc-shape sometimes overrides prompt drama level.
9. **Snapshot does NOT capture scratch disk** (5 TB NVMe). Confirm critical artifacts live on boot disk only — they do. Documented.

### Low priority

10. **madmom dropped**: Python 3.10 incompatibility (np.float, np.int removals + `from collections import MutableSequence`). Switched to librosa `beat_track`. Loses downbeat detection precision (heuristic = every 4th beat).
11. **audiocraft install fails on ROCm** (build wheel error). Path B (MusicGen generative bridges) blocked until resolved or replaced with HF inference API.
12. **Cloudflare tunnel needs payment** (free Zero Trust requires card). Switched to ngrok free static domain `issue-slingshot-bobsled.ngrok-free.dev`. Stable URL across droplet restarts.
13. **DigitalOcean credit terms** ambiguous on snapshots ("Add-Ons" exclusion). Snapshot is ~$7/mo — verify on next bill.

---

## Plan: next 8-12 hours

### Tier 1 — fix core quality (4-6 hr)

1. **Investigate and fix render duration shortfall** in `execute.py`. Likely: track actual rendered samples per entry vs planned, adjust transition `bars` dynamically.
2. **Switch vocal-overlap from subtractive to additive stem mix**: during overlap, render only `{drums, bass, other}` of incoming, hold outgoing's stems. No phase artifacts.
3. **Auto-restrict transition vocabulary by material**: if section is vocal-active, never apply pitch_bend/scratch_fill/spinback. Map to filter_fade/drum_break instead.
4. **Director awareness of incoming section type**: pass each clip's section labels (drop/breakdown/verse) to Director so it can pick `drop` tier correctly when next is a drop.

### Tier 2 — preference fine-tune (3-4 hr)

5. **Pairwise preference labeling**: user listens to 30-50 mix pairs (v6/v9/v10/v11/v12/v15 + new variants), picks winner each. Generate `(prompt, mix_A, mix_B, winner)` JSONL.
6. **DPO/ORPO LoRA fine-tune Qwen2-Audio Director** on preference pairs. ~4 hr training on MI300X.
7. **Promote tuned weights** as default Director.

### Tier 3 — UX + deploy (2-3 hr)

8. **Wire FastAPI on droplet via ngrok stable domain**. Test `/generate` end-to-end through tunnel from laptop.
9. **Deploy HF Space** (Archit00/aijockey) with public demo MP3s + password-gated Try It tab. Set MI300X_URL + MI300X_KEY + ADMIN_PW secrets.
10. **Render final 5 demo MP3s** with v15-quality engine. Replace pre-baked MP3s.
11. **Snapshot final state** as `aijockey-demo-ready`. Destroy live droplet to stop billing.

### Tier 4 — research / stretch (post-demo)

12. **Audio caption library** with Qwen2-Audio (per-clip "uplifting techno, 128 BPM, female vocals from 0:30") → enrich planner with semantic tags.
13. **Better mix critic v2**: codec-invariant features (CLAP embedding instead of mel), longer windows (30-60 s), retrain on more sets.
14. **MusicGen generative bridges** when audiocraft install resolved. Prototype 4-bar interpolation between incompatible clips.
15. **Critic-as-rerank**: tuned critic in `plan_n_best`. Render only top-2 candidates fully, rest by audio-feature proxy.

---

## Cost ledger

| Bucket | Spend |
|--------|-------|
| MI300X compute (this session, ~10 hr cumulative) | ~$20 |
| Snapshot storage (113 GB × $0.06/GB/mo) | ~$7/mo |
| HF Space free tier | $0 |
| ngrok free static domain | $0 |
| **Remaining AMD credits** | ~$80 of $100 |

Buffer for remaining work: ~40 hr droplet uptime. Plenty.

---

## Files of note

| File | Purpose |
|------|---------|
| `AGENTS.md` | architecture playbook for AI coding agents |
| `STATUS.md` | this file |
| `README.md` | onboarding |
| `AiJockey.md` | full architecture + code reference |
| `ARCHITECTURE.md` | streaming agent + LLM hybrid target design |
| `HACKATHON.md` | MI300X hosting + pitch outline |
| `LICENSE` | AGPL-3.0 |
| `app.py` | top-level Gradio entrypoint (legacy local) |
| `space/app.py` | HF Space frontend |
| `server/api.py` | FastAPI backend |
| `server/DEPLOY_AMD.md`, `server/TUNNEL_RUNBOOK.md` | deploy runbooks |
| `src/main.py` | CLI orchestrator |
| `src/director.py` | HF instruct-LM + multimodal Director |
| `src/transition_mapping.py` | 5-tier → DSP technique map |
| `src/planner.py` | beam search + N-best rerank + tier application |
| `src/execute.py` | render with vocal-aware DSP transitions |
| `src/master.py` | adaptive mastering chain |
| `src/training/mix_critic.py` | real-vs-splice critic (parallel build) |
| `src/training/clap_finetune.py` | CLAP compat head |
| `src/training/classifier.py` | Tier 1 technique classifier |
| `scripts/prebake_demos.py` | renders 5 demo mixes |
| `scripts/scrape_dj_sets.py` | yt-dlp 25 long-form DJ sets |
| `scripts/compress_dj_sets.py` | wav→mp3 parallel for fast upload |
| `scripts/backfill_vocal_activity.py` | retrofit vocal_activity into existing cache |
| `scripts/download_diverse.py`, `download_indian.py`, `download_classical.py`, `download_memes.py` | clip-pool gap-fillers |

---

## Bottom line

Pipeline is technically rich: end-to-end DJ engine running on AMD MI300X with multimodal Qwen2-Audio Director, three custom-trained heads (CLAP-compat / technique-classifier / mix-critic), a real DJ-set corpus, and a 5-tier dramatic transition vocabulary that produces audibly varied output (v15).

Quality bar still below pro DJ ear. The remaining gap is best closed by:
(a) preference fine-tune on user labels (DPO/ORPO LoRA on Qwen2-Audio),
(b) fixing the render duration shortfall,
(c) replacing subtractive vocal-suppress with stem-mixing,
(d) section-type-aware transition restriction.

12-hour plan above gets us through (a)-(d) plus HF Space deploy and final snapshot. Defensible pitch story: "AMD MI300X-trained Director, 5-tier dramatic vocabulary, vocal-collision-aware planner, real-DJ-corpus-backed quality estimator, $0 hosting via HF Space + ngrok-free."
