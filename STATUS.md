# AiJockey — Status, Progress, Plan, Bugs

Snapshot of what works, what's broken, what's next.
Last updated: 2026-05-09 (mid-session — AMD MI300X live).

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
