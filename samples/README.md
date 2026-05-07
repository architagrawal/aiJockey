# Sample Bank

Two layers, used by `src/samples.py`:

1. **Real samples** (this directory) — curated CC0/public-domain audio
2. **Synth fallback** (`src/synth_fx.py`) — procedurally generated, always available

If no real sample matches a request, the synth layer fills in. If you never add
real samples, the system still works — it just uses synth FX everywhere.

## Real-sample manifest schema

`samples/manifest.json`:

```json
[
  {"file": "impacts/sub_drop.wav",       "type": "impacts",     "length_beats": 2,  "bpm": "agnostic"},
  {"file": "impacts/reverse_crash.wav",  "type": "impacts",     "length_beats": 1,  "bpm": "agnostic"},
  {"file": "risers/white_noise_4bar.wav","type": "risers",      "length_beats": 16, "bpm": 128},
  {"file": "risers/synth_riser_8bar.wav","type": "risers",      "length_beats": 32, "bpm": 128},
  {"file": "sweeps/downsweep_2bar.wav",  "type": "sweeps",      "length_beats": 8,  "bpm": "agnostic"},
  {"file": "vinyl/spinback_fx.wav",      "type": "vinyl",       "length_beats": 2,  "bpm": "agnostic"},
  {"file": "snare_rolls/roll_4bar.wav",  "type": "snare_rolls", "length_beats": 16, "bpm": 128},
  {"file": "hihat_rolls/hat_2bar.wav",   "type": "hihat_rolls", "length_beats": 8,  "bpm": 128},
  {"file": "airhorns/horn_1.wav",        "type": "airhorns",    "length_beats": 4,  "bpm": "agnostic"}
]
```

Required: `file`, `type`. Optional: `length_beats`, `bpm` ("agnostic" or number), `key`.

`samples.SampleBank.get_fx()` picks the closest match by BPM + length.

## Available FX types (real OR synth fallback)

| Type | Used by transition | Synth available |
|------|-------------------|-----------------|
| `impacts` | silence_drop re-entry | ✅ `synth_fx.impact()` |
| `sub_drops` | silence_drop sub layer | ✅ `synth_fx.sub_drop()` |
| `risers` | loop_tighten build | ✅ `synth_fx.riser_uplift()` |
| `sweeps` | filter_fade accent | ✅ `synth_fx.downsweep()` |
| `snare_rolls` | drum_break lead-in | ✅ `synth_fx.snare_roll()` |
| `hihat_rolls` | eq_swap energy lift | ✅ `synth_fx.hihat_roll()` |
| `airhorns` | loop_tighten high-energy drop | ✅ `synth_fx.airhorn()` |
| `vinyl` | spinback FX | ✅ `synth_fx.vinyl_stop()` |

All 8 types work out-of-the-box via synth. Add real samples to upgrade quality.

## Free CC0 sources for real samples

- https://cymatics.fm/pages/free-download-vault
- https://freesound.org (filter Creative Commons 0)
- https://99sounds.org

## Why the hybrid approach

- **Synth always works** — no downloads needed for first run
- **Real samples beat synth** for impacts/risers/airhorns where character matters
- **Synth wins** for tempo-flexible FX (right BPM auto-generated)
- **No licensing risk** in repo (only synth ships with code)
