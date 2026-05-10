# AiJockey — Session Handoff (2026-05-10)

**Resume reference for next session. Read this first.**

---

## Where we are RIGHT NOW

- **Single branch**: `best-output-pipeline @ df1593e` (tier1-upgrades merged + deleted)
- **Live droplet**: MI300X at `165.245.135.121`. Container `rocm`. SSH:
  `ssh -i C:/Users/msi-laptop/.ssh/aijockey_mi300x root@165.245.135.121`
- **Library**: 158 clips at `/cache` (113 base + 55 Incompetech instrumentals downloaded this session)
- **Pre-staged user pools**: `/workspace/user_set` (4), `/workspace/user_genres/{chillstep,dnb,dubstep,future_bass}` (6 each), `/workspace/test_user_clips` (2)
- **Models loaded**: Beat-This + Demucs htdemucs_ft + Mel-Band Roformer (`/cache/models/MelBandRoformer.ckpt`, 913 MB) + CLAP + Qwen2.5-7B Director + Audiobox Aesthetics

---

## What landed this session

### 1. Director — tier diversity + organic guidance (src/director.py)
- Post-LLM enforcement: `AIJOCKEY_DIRECTOR_TIER_ENFORCE=1` caps minor at `MAX_MINOR_PCT=0.75` (configurable). Promotes interior minors → major when LLM picks all-safe.
- SYSTEM_PROMPT_PHASE1 strengthened with Tomorrowland narrative-driven structure (opener → climb → first peak → valley → bigger peak → callback → outro).
- Vocal-aware tier placement guidance.
- force_drop + auto_accent kept behind env knobs (default off — felt artificial per user feedback).

### 2. Vocal_guard — relaxed per dj_research §6 (src/execute.py)
Three-tier gate replacing single AGGRESSIVE_VS_VOCALS list:
- **SHREDDERS** (always reject when VA>0.30): pitch_bend, bpm_warp, tape_stop, chop, scratch_fill, beat_juggle, loop_roll, loop_tighten, spectral_hold, spinback, forward_spin
- **HEAVY** (reject when VA>0.55): drum_replace, kickless_swap, snare_buildup, build_riser_drop, punch_in
- **ARTIFACT_PRONE** (NEW — reject when VA>0.15, downgrade to gentler equivalent): drum_replace→drum_break, bass_swap→long_crossfade, instrumental_swap→mashup, kickless_swap→highs_swap

### 3. Restricted-mode whitelist expanded 12→30 (src/restricted_mode.py)
Added all vocal-safe transitions per dj_research: short/long_crossfade, bass_swap, highs_swap, frequency_blend, highpass_sweep_in, band_filter_sweep, punch_in, kickless_swap, drum_replace, instrumental_swap, acapella_drop, reverb_wash, harmonic_overlay, riser_overlay, impact_overlay, snare_buildup, build_riser_drop. Time/pitch/per-sample warps still safely excluded.

### 4. Vocal-phrase boundary snap (src/execute.py + src/vocal_phrase.py)
`snap_segment_end_to_vocal_silence()` runs after phrase-quantize. Per junction, scans vocals stem RMS in [end-1.5, end+5.0], finds earliest 0.25s+ window below -36dB, snaps end there. Vocals never cut mid-phrase. `AIJOCKEY_VOCAL_END_SNAP=0` disables.

### 5. Segment cap for set rotation (src/planner.py)
- `min_segment_seconds: 30→18` + `max_segment_seconds: 32` (NEW)
- Net: 5min set goes from 5-8 entries (no rotation) → 11-16 entries (A-B-C-A-D callbacks)

### 6. Tomorrowland multi-peak arc (src/planner.py)
`ARC_PRESETS['tomorrowland'] = [0.35, 0.55, 0.80, 0.95, 0.55, 0.45, 0.70, 0.95, 1.0, 0.85, 0.60, 0.35]` — opener → first peak → valley → bigger peak → cooldown.

### 7. Tape saturation mastering (src/master.py)
Mid-band tanh asymmetric soft-clip after multi-band compression. `AIJOCKEY_MASTER_TAPE_SAT=1` + `AIJOCKEY_MASTER_TAPE_DRIVE=0.6` defaults. Marginal Audiobox lift.

### 8. Library expansion +55 clips
55 Incompetech CC-BY tracks downloaded + analyzed. VA mean **0.017** (super instrumental) — fills the vocal-balance gap in the prior library. Sources direct mp3 wget, no bot block.

### 9. UI preset system (server/preset.py — NEW)
Drop-in `apply_preset(mode, vocals, style, arc, mix_mode, advanced, base_prompt)` returns `(env_overrides, cli_overrides)`. Plus `compose_cli_args()` and `PRESET_SCHEMA` + `ADVANCED_SCHEMA` for frontend auto-render.

Mode presets:
- **mashup**: tape sat off, strict vocal_guard (0.30/0.45), arc=build, callbacks=2, no segment cap. v6-style PQ ~7.84.
- **dj_set**: tape sat on (drive 0.3), relaxed vocal_guard (0.40/0.60), arc=tomorrowland, callbacks=4, segment cap 28s. v9/v12-style with peaks.

Vocals presets: `on` (default) / `off` (instrumental_only) / `dim` (-6dB, needs ~5 LOC in execute.py).

Style presets: festival_inferno, midnight_noir, neon_retrowave, east_meets_bass, bollywood_block_party (5 CLAP-text + arc shortcuts).

### 10. /generate endpoint wired with preset (server/api.py)
- New form fields: `mode`, `vocals`, `style`, `advanced_json`, `sample_clip_ids`
- `apply_preset()` dispatched, env_overrides + cli_overrides flow into subprocess
- New endpoints: `/preset_schema` (frontend auto-render), `/sample_clips` (load-demo button)
- Phase 1 arc fallback: tomorrowland → peak when not in current phase

### 11. Audiobox Aesthetics critic (Tier 1 C)
Wired into /generate post-master. 4-axis score (PQ/PC/CE/CU) emitted as `X-Audiobox` response header + jlog row. Disable: `AIJOCKEY_AUDIOBOX_AESTHETICS=0`.

### 12. Mel-Band Roformer vocal stems (Tier 1 A)
Wrapper at src/mel_band_roformer_wrapper.py + 913MB ckpt at /cache/models/. Activate: `AIJOCKEY_MEL_BAND_ROFORMER=1` + `AIJOCKEY_MEL_BAND_ROFORMER_CKPT`.

---

## Demo gallery — 10 curated mixes

Path: `c:/Archit/MyCoding/Development/AiJockey/output/featured/` + `README.md`

| # | file | PQ | role |
|---|---|---|---|
| 1 | userset_v6_t1.mp3 | **7.84** | mashup gold standard |
| 2 | mega_cross_genre.mp3 | **7.64** | flagship 6-min Tomorrowland cross-genre journey |
| 3 | genre_chillstep.mp3 | 7.60 | pure chillstep set |
| 4 | aug_dnb.mp3 | 7.58 | dnb + library bridges |
| 5 | genre_dnb.mp3 | 7.57 | pure DnB high-energy |
| 6 | aug_chillstep.mp3 | 7.51 | chillstep + bridges |
| 7 | genre_future_bass.mp3 | 7.42 | future bass uplifting |
| 8 | userset_v12.mp3 | 7.23 | dj_set + VA gate (best variety+quality balance) |
| 9 | userset_v9_t1.mp3 | 7.15 | dj_set raw variety (A/B with v12) |
| 10 | genre_dubstep.mp3 | 7.14 | dubstep peaks |

---

## Audiobox Aesthetics — measured ceilings

Across 12+ render iterations (v2 → v12 + 9 genre demos):

- **PQ ceiling** = 7.84 (v6_t1 mashup). Adding variety always costs PQ (more transitions = more artifact accumulation).
- **CE peak** = 7.22 (v5 dj_set with strong arc).
- **DJ-set quality recovery via VA gate**: v9 (7.15) → v12 (7.23) — kept 11 unique transitions while clamping artifact-prone techs on vocal junctions.
- **Pool quality dominates everything**: picker upgrades > stem model > DSP-level (DTW, harmonic_cap). Mel-Band activation: lateral on Audiobox aggregate.

---

## Active env knobs (production defaults)

```
AIJOCKEY_PHASE=1
AIJOCKEY_USE_DIRECTOR_LLM=1
AIJOCKEY_VOCAL_GUARD=1
AIJOCKEY_VOCAL_GUARD_THR=0.30
AIJOCKEY_VOCAL_GUARD_THR_HEAVY=0.55
AIJOCKEY_VOCAL_GUARD_THR_QUALITY=0.15
AIJOCKEY_VOCAL_END_SNAP=1
AIJOCKEY_VOCAL_SAFE_STRETCH=1
AIJOCKEY_VOCAL_RAMP_BEATS=4
AIJOCKEY_DIRECTOR_TIER_ENFORCE=1
AIJOCKEY_DIRECTOR_MAX_MINOR_PCT=0.75
AIJOCKEY_MASTER_TAPE_SAT=1
AIJOCKEY_MASTER_TAPE_DRIVE=0.6
AIJOCKEY_PHRASE_QUANTIZE=1
AIJOCKEY_STEM_SWAP=1
AIJOCKEY_CONSTITUTIONAL=1
AIJOCKEY_CANDIDATE_PICKER=1
AIJOCKEY_ALL_IN_ONE=1
AIJOCKEY_MEL_BAND_ROFORMER=1
AIJOCKEY_AUDIOBOX_AESTHETICS=1
AIJOCKEY_DTW_ALIGN=1
AIJOCKEY_HARMONIC_OVERLAP_CAP=1
```

---

## Teammate's 5 ship tasks for demo (~1.5-2 hr)

1. **Add `tomorrowland` to `PHASE1_ARCS`** (`server/api.py:60`) — 1 LOC. Unblocks dj_set arc differentiation.
2. **Verify `apply_preset()` wire works in /generate** end-to-end on droplet — already merged, smoke test only.
3. **Add `/preset_schema` endpoint reachable from frontend** — already in api.py, frontend just needs to fetch + render.
4. **Frontend mode/vocals/style toggles in `space/app.py`** (Gradio) — radios + dropdown wired to form POST.
5. **Demo button: pre-fill `dj_set` + 4 user clips + tomorrowland + festival_inferno** — load-demo button + sample_clips endpoint already shipped. Just wire UI handler.

---

## Known issues / TODO (post-demo)

- **Phase 1 arc whitelist** doesn't include 'tomorrowland' yet. Preset asks for it → falls back to peak. Fix: add to PHASE1_ARCS list. (1 LOC.)
- **vocal_dim mode**: preset.py exposes it but execute.py needs ~5 LOC to honor `AIJOCKEY_VOCAL_DIM_DB`.
- **Audiobox-feedback rerank** wired in library_picker_score.py but call site missing in /generate post-master. Closed-loop quality improvement waiting.
- **Multi-Director sampling** (temp>0, 3 plans, pick best) not shipped — biggest single-shot quality lift remaining.

---

## Critical paths

| component | file |
|---|---|
| Preset logic | `server/preset.py` |
| /generate handler | `server/api.py` |
| Frontend | `space/app.py` |
| Mode env mappings | `server/preset.py:_MODE_PRESETS` |
| Director | `src/director.py` |
| Vocal guard | `src/execute.py` (line ~689) |
| Segment cap | `src/planner.py PlannerConfig` |
| Tomorrowland arc | `src/planner.py ARC_PRESETS` |
| Mastering | `src/master.py` |
| DJ research source | `docs/dj_research.md` |
| Demo mp3s | `output/featured/` (10 curated) |

---

## Quick smoke test on droplet

```bash
ssh -i C:/Users/msi-laptop/.ssh/aijockey_mi300x root@165.245.135.121
docker exec rocm bash -lc 'cd /workspace && git pull origin best-output-pipeline'
docker exec rocm curl -s http://localhost:PORT/preset_schema | head
docker exec rocm curl -s http://localhost:PORT/sample_clips | head
```

Render direct CLI smoke (works without UI):
```bash
docker exec rocm python3 /workspace/src/main.py plan \
  --cache /workspace/cache_user_set --out /tmp/tl.json --duration 240 \
  --arc build --use_director --apply_llm_tiers \
  --prompt "test" --callbacks 2 --reuse_cooldown 5 --max_clips 12
```
