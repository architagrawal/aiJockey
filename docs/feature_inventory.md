# Feature inventory — exposed knobs

Reference for what `/generate` accepts and what `space/app.py` surfaces.
Source of truth: `server/preset.py` (`PRESET_SCHEMA`, `ADVANCED_SCHEMA`).

## Tier 1 — exposed in HF Space UI today

| Knob | Param | Values | Default | UI control |
|------|-------|--------|---------|-----------|
| Mode | `mode` | `mashup` \| `dj_set` | `dj_set` | radio |
| Vocals | `vocals` | `on` \| `off` | `on` | radio |
| Style | `style` / `preset` | 5 PRESETS keys | `festival_inferno` | dropdown |
| Mix mode | `mix_mode` | `tight` \| `balanced` \| `exploratory` | `balanced` | radio |
| Use library | `use_library` | bool | `false` | checkbox |
| Length | `duration` | 30–600s | 120 | slider |
| Loudness | `lufs` | -14 to -7 | -9 | dropdown |
| Custom prompt | `prompt` | freeform | — | textbox (advanced) |
| Custom arc | `arc` | phase1: build/peak/flat_low | — | dropdown (advanced) |
| Seed | `seed` | int | — | number (advanced) |

## Tier 2 — in `ADVANCED_SCHEMA`, not yet UI-surfaced

JSON dict via `advanced_json` form param. Backend dispatches to env / cli via
`preset._ADV_TO_ENV` and `_ADV_TO_CLI`.

| Knob | Type | Default | Effect |
|------|------|---------|--------|
| `tape_sat` | bool | true | mastering tape saturation |
| `tape_drive` | float 0–1 | 0.6 | drive amount |
| `mel_band` | bool | true | Mel-Band Roformer for vocals stem |
| `dtw_align` | bool | true | sub-sample beat snap |
| `use_director` | bool | true | LLM director on/off |
| `tier_enforce` | bool | true | enforce tier diversity |
| `max_minor_pct` | float 0.5–1.0 | 0.75 | max % minor tier transitions |
| `force_drop` | bool | false | force at least one drop |
| `auto_accents` | bool | false | director adds accent FX |
| `vocal_guard` | bool | true | vocal-collision check |
| `vocal_guard_thr` | float 0.2–0.5 | 0.30 | strict threshold |
| `vocal_guard_thr_heavy` | float 0.4–0.7 | 0.55 | heavy-action threshold |
| `vocal_end_snap` | bool | true | snap segment end past vocal phrase |
| `vocal_safe_stretch` | bool | true | cap stretch to preserve vocal pitch |
| `vocal_ramp_beats` | int 2–8 | 4 | vocal ducking ramp length |
| `phrase_quantize` | bool | true | quantize segments to phrase grid |
| `stem_swap` | bool | true | stem-aware overlap |
| `candidate_picker` | bool | true | scored junction picker |
| `all_in_one` | bool | true | All-In-One structure analyzer |
| `audiobox_critic` | bool | true | post-render quality score |
| `constitutional` | bool | true | hard rule validator |
| `n_best` | int 1–8 | 1 | render N plans, pick best |

## Mode → env mapping

`mashup`:
- `AIJOCKEY_MASTER_TAPE_SAT=0`
- `AIJOCKEY_DIRECTOR_MAX_MINOR_PCT=0.85` (more minors OK = smoother)
- `AIJOCKEY_VOCAL_GUARD_THR=0.30` (strict)
- `min_segment_seconds=28`, `max_segment_seconds=0` (long camps)
- arc → `build`, callbacks=2, reuse_cooldown=5

`dj_set`:
- `AIJOCKEY_MASTER_TAPE_SAT=1`, `_TAPE_DRIVE=0.3`
- `AIJOCKEY_DIRECTOR_MAX_MINOR_PCT=0.65` (force majors)
- `AIJOCKEY_VOCAL_GUARD_THR=0.40` (looser)
- `min_segment_seconds=18`, `max_segment_seconds=28` (rotation)
- arc → `peak` (Phase 1 substitute for tomorrowland), callbacks=4, reuse_cooldown=1

## Out-of-scope for hackathon

- 22 power-user toggles in `ADVANCED_SCHEMA` — not UI-surfaced. Reachable via
  `advanced_json` POST field, or wait for JSON-driven form auto-render.
- Vocals=`dim` (-6dB background mix) — listed in schema, needs ~5 LOC in
  `execute.py` + new env knob. Defer post-demo.
- CLI-only flags (`--callbacks`, `--reuse_cooldown`, `--n_best`,
  `--no_user_floor`) — driven via env from preset module instead.

## Endpoints

- `GET /preset_schema` → `{"presets": PRESET_SCHEMA, "advanced": ADVANCED_SCHEMA}`.
  Frontend can fetch + render form dynamically without hard-coding option lists.
- `POST /generate` accepts all Tier 1 fields above + optional `advanced_json`.
- `GET /jobs/{id}/timeline` returns segment-source attribution for color-coded
  timeline rendering.
