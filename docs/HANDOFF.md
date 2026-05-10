# AiJockey — Session Handoff (2026-05-09)

Resume reference for next conversation. Read this first to recover full context.

---

## Where we are

**Branch**: `best-output-pipeline` (head: `9c7a52a`, pushed to origin)

**Live state**: MI300X 1× ($1.99/hr) running at `165.245.135.121`. Container `rocm` mounts `/root/aijockey` → `/workspace`. Snapshot `aijockey-slim` (109.7 GB, atl1) is the boot image — has 103 pre-analyzed clips at `/cache`.

**SSH**: `ssh -i "C:/Users/msi-laptop/.ssh/aijockey_mi300x" root@165.245.135.121`

**Render path**: works end-to-end. test1.wav + test2.wav → `output/gpu_pulls/test_3min_v3.wav` (136 s, Director-driven, narrative + intents present).

---

## What got done this session

### Foundation (`best-output-pipeline` core)
- Phrase quantization (planner + execute)
- Stem-additive overlap (kills phasey vocal-mute artifact)
- Humanized accent FX (±8 ms / ±8 % deterministic seed)
- Sample lib whitelist (no airhorn/meme pollution)
- Constitutional rule validator (phrase-grid, drop-section, breakdown-pair, bpm-drift, key-compat, accent-budget, **user-clip-floor**)
- 3-tier vocab + 3-arc Phase 1 scope
- Min-segment length enforcement + overlap clamp (1/3 of shorter side, 8-bar cap, bars shadowed into transition primitives)
- Tier propagation into `transition_in['tier']` so constitutional sees Director's intent

### Intelligence layer
- `src/pool_intelligence.py` — pool inventory (USER/LIB tag, genre, BPM, section, energy), clusters, coherence score, diagnose verdict + narrative_advice
- Director SYSTEM_PROMPT_PHASE1 — pool inventory injected, demands `set_narrative` + `narrative_notes` + per-junction `transition_intents` (7 categories)
- `apply_llm_transition_tiers_to_timeline(transition_intents=...)` propagates intent
- `card.json` saved next to render with full reasoning trail

### Library augmentation UX
- `mix_mode` enum (`tight` / `balanced` (default) / `exploratory`)
- `library_role` optional (`bridges_only` / `warmup_outro` / `fill_gaps` / `any`)
- `lib_count_for_mode(mode, user_count, user_total_dur, target_duration)` formula
- `library_clip_paths_clap()` — CLAP-cosine retrieval against library cache, BPM ±15 % filter
- Source tagging: user clips get `source='user'`, library clips get `source='library'` in cache JSON
- Constitutional `check_user_clip_floor` warns if any user clip missing from timeline
- Response headers: `X-Mix-Mode`, `X-Clips-Used` JSON `{user_count, library_count, library_ids[]}`

### Performance
- Demucs default `htdemucs_ft` (~0.5 dB SDR cleaner stems) with BagOfModels guard against `torch.compile` crash
- bf16 autocast on Demucs/Director (MI300X native)
- ROCm-safe `torch.compile` mode default (`'default'`, not `'reduce-overhead'`)
- Stem-parallel rubberband via ThreadPoolExecutor (~3-4× CPU util on render)
- Timeline-segment parallel render (`AIJOCKEY_RENDER_WORKERS=2`)
- Free-as-you-go memory: `rendered[i-1] = None` after consumed
- Single-call rubberband (`AIJOCKEY_RB_COMBINED=1`)
- CLAP batched embeddings + Qwen2-Audio batched captions

### Render bug fixes (live-discovered on MI300X)
- 9 commits chained from `5b4f9cd` → `f68b62b` (see STATUS.md table for full list with commit refs)
- Fix highlights: tier propagation, min-segment 8 bars, overlap clamp passed to primitives, Qwen3-8B revert (model doesn't exist on HF), tokenizer max_length 2048 → 8192, mt cap 64 → 16

### Docs
- STATUS.md — top session-update block, 9-bug commit table, smoke progression table
- AGENTS.md — Director row + Pool intelligence row, `/generate` form fields, audio Director caveats, `HF_DIRECTOR_MODEL` corrected
- docs/dj_research.md, docs/phase1_plan.md (earlier in branch)

### Tests
- 14/14 unit tests pass (`tests/test_constitutional.py`, `tests/test_phrase_quantize.py`, `tests/test_transition_mapping.py`)

---

## Open issues (known)

| Issue | Where | Severity |
|---|---|---|
| ~~Downbeat heuristic `beats[::4]`~~ — RESOLVED via Beat-This! integration (`src/beat_this_wrapper.py`); librosa stays as last-resort fallback | `src/analyze.py:beats_and_downbeats` | **resolved** (validate on MI300X) |
| Audio Director path exists but capped at 6 user clips × 30 s window | `src/director.py:_call_qwen2audio` | low — documented in AGENTS.md |
| Qwen3-8B-Instruct ID does not exist on HF | reverted in `f68b62b` | resolved |
| htdemucs_ft + torch.compile = crash | guard added in `aeb75ac` | resolved |
| 2-clip user pool ceiling: render 136 s of 180 s target | inherent to pool size | use library augmentation for longer outputs |
| Pool of 103 cached clips is genre-disparate (20 genres, BPM 57-195, coherence 0.0) | snapshot lib | curate by genre cluster OR user supplies tighter clips |

---

## Next tasks (priority order)

### P0 — DONE this session (validate on MI300X)

**1. Beat-This! integration** — DONE (`src/beat_this_wrapper.py`, wired into `Analyzer.beats_and_downbeats`). `AIJOCKEY_BEAT_THIS=1` (default). Falls through to librosa. Need MI300X smoke to confirm phrase-grid drift gone.

**2. BS-Roformer for vocal stem** — SCAFFOLD DONE (`src/bs_roformer_wrapper.py`, `Analyzer._maybe_swap_vocals`). Opt-in: `AIJOCKEY_BS_ROFORMER=1` + `AIJOCKEY_BS_ROFORMER_CKPT=/path`. drums/bass/other still demucs. Need to download a vocals checkpoint + smoke.

**GPU saturation pass** — DONE. GPU was 4% util; new batched-Demucs path (`stems_batch`), stage1 micro-batching (default 4 clips/forward), CLAP chunking, bumped `_RENDER_WORKERS` 2→6 + `_STEM_WORKERS` 4→8. Need MI300X `rocm-smi` watch on next run to measure lift.

### P1 — UX completion + observability

**3. Wire CriticV2 (`scripts/stage4_critic.py` model) into `/generate` as global score gate**
- Trained checkpoint exists at `checkpoints/mix_critic.pt` per STATUS.md
- Output rendered mix → CriticV2 forward → score in 0..1
- Threshold gate: if score < 0.5, skip ship, flag for re-plan (or fall through with warning)
- Enables future self-critique loop without paying LLM tax

**4. Cheap audio probes** (numpy-only, ~100 lines)
- RMS envelope per stem at junction → energy-mismatch detector
- Cross-correlation vocal stems prev/cur → vocal-bleed detector
- Spectro diff @ overlap window → phasing detector
- 100 ms each, catches 70 % of audio artifacts at 1 % the cost of an audio LLM critic

**5. Test mix_mode + library augmentation end-to-end on MI300X**
- Need real library populated at `/workspace/clips` + `/workspace/cache` (currently empty for library — only user-side test runs done)
- Smoke 3 modes (tight / balanced / exploratory) with 3 user clips + library
- Verify `X-Clips-Used` header populates correctly
- Listen-test difference

### P2 — quality compounding

**6. Rule-based improver + surgical re-render** (only after P0+P1)
- 3-5 issue-type → deterministic timeline edit map (e.g. `vocal_collision @ junction_5 → set instrumental_only=true for that overlap`)
- Cascade re-render: changing segment N invalidates transitions N-1→N AND N→N+1, so re-render = segment + 2 transitions
- Confidence threshold on critic findings (≥ 0.7 to act)

**7. Audio LLM critic as fallback** (only when symbolic + cheap probes empty AND CriticV2 below threshold)
- Qwen2-Audio with structured JSON output `{junction, issue_type, severity, confidence, suggested_fix}`
- ~30-60 s latency tax per critique (realistic, not the 5-10 s teammate estimated)
- Opt-in flag, never default

**8. Style-RAG corpus build** (when training pipeline is wanted)
- Run `scripts/stage3_embed.py` over a real DJ-set corpus to populate `/scratch/embed/clap.npy` + `clap_index.json` + `captions.json`
- Director already reads via `style_rag.few_shot_block_for_director` — currently empty index = no retrieval. Populate to activate.

### P3 — deferred / optional

- Qwen3-4B-Instruct-2507 swap (newer family, smaller — quality regression possible vs Qwen2.5-7B)
- Qwen3-235B-A22B-Instruct-2507 (MoE, 22B active, needs QLoRA on 192 GB)
- Per-genre LoRA Director adapters
- DPO loop (S5+S7 pipeline) if pre-render still produces consistently fixable mistakes
- MuQ for music similarity (when library retrieval ranking matters)
- Qwen2.5-Omni-7B (only when audio Director becomes critical path)

### Demo / shipping items

- Curate a clean 20-30 clip library by genre cluster at `/workspace/clips` (not the 103-clip dispar /cache snapshot)
- Render 5 demo mixes using mix_mode=balanced + curated library
- HF Space deploy with mix_mode UI knob
- `aijockey-demo-ready` snapshot of final state, destroy live droplet

---

## Cost ledger

- $80 of $100 AMD credits remaining at session start
- ~10 hr droplet uptime this session ≈ $20
- Remaining budget ~$60 ≈ 30 hr droplet time

---

## How to resume in a new conversation

1. **Read this file first** — `docs/HANDOFF.md`
2. **Check branch**: `git log --oneline best-output-pipeline ^main | head -5` should show `9c7a52a` at top (or later)
3. **Verify GPU live**: `ssh -i "C:/Users/msi-laptop/.ssh/aijockey_mi300x" root@165.245.135.121 "docker ps"` should show `rocm` container running. If droplet destroyed, see [server/DEPLOY_AMD.md](../server/DEPLOY_AMD.md) for re-spin from snapshot.
4. **Verify tests pass**: `python -m pytest tests/test_constitutional.py tests/test_phrase_quantize.py tests/test_transition_mapping.py -q` should be `14 passed`.
5. **Listen to test_3min_v3.wav**: at `output/gpu_pulls/test_3min_v3.wav` — that's the current quality baseline (Director-driven, htdemucs_ft, all DSP fixes active).
6. **Read STATUS.md top block** — most up-to-date code/commit map.
7. **Pick next task from P0 above** (Beat-This! is highest leverage).

---

## Key files to know

| File | Role |
|---|---|
| `src/director.py` | LLM Director with pool-aware system prompt + narrative + intents |
| `src/pool_intelligence.py` | Pool tagging, clustering, diagnose, summary table |
| `src/constitutional.py` | Hard musical rules + repair |
| `src/execute.py` | Render path with phrase-quantize + min-segment + overlap clamp + stem-swap |
| `src/analyze.py` | Demucs (htdemucs_ft default) + CLAP + librosa beats |
| `server/api.py` | `/generate` endpoint with mix_mode + library augmentation |
| `src/main.py` | CLI orchestrator (analyze / plan / execute / master) |
| `tests/` | 14 unit tests, all passing |
| `STATUS.md` | Current truth |
| `AGENTS.md` | Deploy playbook + env vars |
| `docs/dj_research.md` | DJ domain research grounding |
| `docs/phase1_plan.md` | Original plan (earlier in branch) |

---

## Smoke test command (verify everything works)

```bash
ssh -i "C:/Users/msi-laptop/.ssh/aijockey_mi300x" root@165.245.135.121 "docker exec rocm bash -c '
cd /workspace
export AIJOCKEY_PHASE=1 AIJOCKEY_DTYPE=bfloat16 AIJOCKEY_FLASH_ATTN=1
export AIJOCKEY_PHRASE_QUANTIZE=1 AIJOCKEY_STEM_SWAP=1 AIJOCKEY_CONSTITUTIONAL=1
export AIJOCKEY_INSTRUMENTAL_ONLY=1 AIJOCKEY_USE_DIRECTOR_LLM=1
export TRANSFORMERS_VERBOSITY=error
cd src
python3 main.py plan --cache /workspace/test_user_cache --out /workspace/output/smoke_timeline.json \
  --duration 180 --arc build --use_director --apply_llm_tiers \
  --min_unique_clips 2 --max_clips 2 \
  --prompt \"3 minute mix\"
python3 main.py execute --timeline /workspace/output/smoke_timeline.json \
  --cache /workspace/test_user_cache --out /workspace/output/smoke_raw.wav
python3 main.py master --in /workspace/output/smoke_raw.wav \
  --out /workspace/output/smoke.wav --lufs -9
cat /workspace/output/card.json
'"
```

Expected: `card.json` contains real `set_narrative` (not deterministic fallback), narrative_notes admits pool limitations, transition_plan has tier+intent pairs.

If `_fallback: True` in director.json → Director failed to load. Likely env config issue.
