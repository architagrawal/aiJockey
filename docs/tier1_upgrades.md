# Tier-1 model upgrades — wire-in guide

Three pretrained-model swaps from the post-cohort research pass. All
shipped as standalone wrappers under `src/`. None modifies existing
files. Each is opt-in via env knob, gracefully degrades when deps or
checkpoints missing.

| Module | Replaces | License | Effort to wire |
|--------|----------|---------|----------------|
| `all_in_one_wrapper.py` | Beat-This! + librosa MFCC segmentation | MIT | ~5 lines in `src/analyze.py` |
| `mel_band_roformer_wrapper.py` | htdemucs_ft vocals stem | MIT | ~3 lines in `src/analyze.py` |
| `audiobox_aesthetics.py` | CriticV2 / supplementary critic | research-friendly | ~5 lines in `server/api.py` |

## A. All-In-One Music Structure Analyzer

### What it does
Single transformer pass returns: tempo, beats, downbeats, AND functional
section labels (`intro / outro / break / bridge / inst / solo / verse /
chorus`). Replaces three current modules (Beat-This! + librosa
segmentation + heuristic energy labeling) with one model call.

### Install
```bash
pip install allin1
# First call downloads ~500MB checkpoint into ~/.cache/all-in-one/
```

### Wire-in (`src/analyze.py`)

In `Analyzer.beats_and_downbeats()`, add at the top:

```python
def beats_and_downbeats(self, wav: torch.Tensor, audio_path: str | None = None) -> tuple[float, list[float], list[float]]:
    # All-In-One first when enabled — joint beats+downbeats+sections in one pass.
    if audio_path:
        try:
            from all_in_one_wrapper import beats_and_downbeats as aio_bd, enabled as aio_enabled
            if aio_enabled():
                out = aio_bd(audio_path, device=self.device)
                if out is not None:
                    return out
        except Exception:
            pass
    # ... existing Beat-This! / madmom / librosa fallback chain ...
```

Add `audio_path` to the call site (`analyze_clip`):

```python
tempo, beats, downbeats = self.beats_and_downbeats(wav, audio_path=path)
```

Same pattern for `sections()` — call `all_in_one_wrapper.sections_for_clip(path)`
when enabled, fall through on None.

### Activate
```bash
export AIJOCKEY_ALL_IN_ONE=1
pytest tests/test_all_in_one_wrapper.py -q   # smoke
```

### Why it matters
- Director's `drop only on drop-section` rule becomes empirically grounded
  (`chorus`, `solo`, `bridge` labels available directly).
- Single GPU forward pass per clip — cheaper than running Beat-This! +
  librosa segmentation separately.

---

## B. Mel-Band Roformer for vocal stems

### What it does
Replaces the htdemucs_ft vocal stem with a Mel-scale Roformer. SDR
~11.93 dB vs htdemucs_ft ~9 dB → cleaner stem-swap, less phase residue
on mashup transitions. Drums/bass/other still come from htdemucs_ft
(Mel-Band published checkpoints are vocals-only).

### Install
```bash
pip install bs-roformer   # provides MelBandRoformer class

# Fetch a published checkpoint, e.g. UVR Mel-Band Roformer vocals:
mkdir -p /scratch/checkpoints
wget -O /scratch/checkpoints/mel_band_roformer_vocals.ckpt \
    https://huggingface.co/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/resolve/main/MelBandRoformer.ckpt
```

### Wire-in (`src/analyze.py`)

`Analyzer.stems()` already calls `_maybe_swap_vocals()` from the
existing `bs_roformer_wrapper`. Switch the import in
`_maybe_swap_vocals()` to prefer Mel-Band when its env enabled:

```python
def _maybe_swap_vocals(self, wav, stems):
    # Mel-Band Roformer first (better SDR), then BS-Roformer, then demucs default.
    for mod_name in ('mel_band_roformer_wrapper', 'bs_roformer_wrapper'):
        try:
            mod = __import__(mod_name)
            if mod.enabled():
                new_vox = mod.vocals_from_wav(wav, sr=SR, device=self.device)
                if new_vox is not None:
                    ref_len = min(s.shape[-1] for s in stems.values())
                    stems['vocals'] = new_vox[:, :ref_len]
                    return stems
        except Exception:
            continue
    return stems
```

### Activate
```bash
export AIJOCKEY_MEL_BAND_ROFORMER=1
export AIJOCKEY_MEL_BAND_ROFORMER_CKPT=/scratch/checkpoints/mel_band_roformer_vocals.ckpt
```

### Why it matters
- Cleaner vocal stem → cleaner stem-swap transitions (currently the
  largest source of "phasing residue" flagged by audio_probes).
- Direct upgrade — branch already has BS-Roformer scaffold, this adds the
  Mel-Band path next to it.

---

## C. Meta Audiobox Aesthetics — reference-free critic

### What it does
Pretrained 4-axis quality scorer from Meta. Returns Production Quality,
Production Complexity, Content Enjoyment, Content Usefulness. No ground
truth needed, just the rendered mix. Beats CLAP score for perceptual
quality (CLAP measures semantic alignment, not quality).

### Install
```bash
# Standalone package (preferred)
pip install audiobox-aesthetics

# Or via HF transformers (fallback path the wrapper handles)
# pip install transformers torch torchaudio librosa
```

### Wire-in (`server/api.py`)

After the render lands and `X-Probe` header is attached, add:

```python
from audiobox_aesthetics import score, severity_proxy
aes = score(rendered_path)
if aes:
    headers['X-Aesthetics-PQ'] = f"{aes['PQ']:.2f}"
    headers['X-Aesthetics-CE'] = f"{aes['CE']:.2f}"
    sev = severity_proxy(aes)
    if sev is not None:
        headers['X-Aesthetics-Severity'] = f"{sev:.3f}"
```

Wire into `probe_log.py` log row too — gives DPO a second-opinion signal:

```python
row['aesthetics'] = aes if aes else None
```

### Activate
```bash
export AIJOCKEY_AUDIOBOX_AESTHETICS=1
```

### Why it matters
- Replaces unreliable CriticV2 (val acc 0.77, codec bias) with zero
  training cost. Pretrained, open weights.
- Probes catch deterministic artifacts (energy / phase / xcorr).
  Audiobox catches subjective polish (mastering, spatialization,
  dynamics) that probes miss.
- Gives DPO a richer reward signal than aggregate severity.

---

## Combined activation profile

For a maximalist run:

```bash
export AIJOCKEY_ALL_IN_ONE=1
export AIJOCKEY_MEL_BAND_ROFORMER=1
export AIJOCKEY_MEL_BAND_ROFORMER_CKPT=/scratch/checkpoints/mel_band_roformer_vocals.ckpt
export AIJOCKEY_AUDIOBOX_AESTHETICS=1
```

Estimated cumulative impact (per research summary in NEXT.TXT):
- ~30% additional severity reduction from All-In-One ground-truth section
  labels improving Director plans.
- ~+2 dB SDR on vocals stem from Mel-Band Roformer → cleaner stem-swap.
- Aesthetics critic adds quality signal independent of probes — surfaces
  failures probes can't see.

## Testing strategy

Each module has a smoke test under `tests/test_tier1_wrappers.py`:

```bash
pytest tests/test_tier1_wrappers.py -q
```

Tests confirm:
1. Modules importable with deps absent (returns None, doesn't crash).
2. `enabled()` reflects env state correctly.
3. Stub mocks of the underlying libraries return shape-correct data.

Real model loads are skipped in CI (would download multi-GB checkpoints).
On the droplet, run end-to-end smoke separately:

```bash
docker exec rocm AIJOCKEY_ALL_IN_ONE=1 \
    /opt/venv/bin/python -c "
from all_in_one_wrapper import analyze_audio_path
r = analyze_audio_path('/cache/stems/some_clip/drums.wav')
print(r['tempo'], len(r['beats']), len(r['downbeats']), len(r['sections']))
"
```

## When to ship each

| Trigger | Module to enable |
|---------|------------------|
| Cohort severity stuck above 0.5 | A (All-In-One) — section labels improve Director planning |
| Stem-swap transitions sound phasey on real ears | B (Mel-Band Roformer) — cleaner vocals |
| Need second-opinion critic before deploy | C (Audiobox) — aesthetic gating |
| All three above | All — diminishing returns past A+B for now |

## Rollback

Each module is fully gated by env. Disable with:

```bash
unset AIJOCKEY_ALL_IN_ONE
unset AIJOCKEY_MEL_BAND_ROFORMER
unset AIJOCKEY_AUDIOBOX_AESTHETICS
```

Or just remove the env vars from `pipeline_launch.sh`. Source code reverts
automatically to existing Beat-This! / htdemucs_ft / probes-only path.
