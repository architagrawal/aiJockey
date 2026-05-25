# AiJockey Showcase

Curated AI-generated DJ sets. Top PQ-scored outputs from the AiJockey pipeline.

All tracks PQ 7.61 (peak quality score, current ceiling). MP3 192kbps.

## Tracks

| # | File | Source | Aesthetic | Build |
|---|------|--------|-----------|-------|
| 1 | `01_PQ7.61_userset_dj_set_midnight_noir_tomorrowland.mp3` | userset DJ set | midnight noir | tomorrowland |
| 2 | `03_PQ7.61_chill_dj_set_festival_inferno_tomorrowland.mp3` | chill DJ set | festival inferno | tomorrowland |
| 3 | `05_PQ7.61_userset_mashup_midnight_noir_build.mp3` | userset mashup | midnight noir | build |
| 4 | `08_PQ7.61_chill_dj_set_midnight_noir_tomorrowland.mp3` | chill DJ set | midnight noir | tomorrowland |
| 5 | `10_PQ7.61_userset_dj_set_festival_inferno_build.mp3` | userset DJ set | festival inferno | build |

## Pipeline

Generated via VampNet + MERT reward + DPO-finetuned transitions. Effect routing through fx_orchestrator (mutex groups, per-set FX budget). 145-bridge transition catalog.

See repo root for full system.
