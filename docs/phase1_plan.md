# AiJockey — Best-Output Production Plan

Consolidated plan from `docs/dj_research.md` recommendations, mapped to `STATUS.md` bugs, current code, and existing architecture docs.

**Goal**: produce the best possible offline-rendered DJ mixes from current MI300X capacity. Cold-start, latency, and live serving are NOT optimization targets — pre-rendered MP3s are the deliverable. Quality per render is the only metric.

**Two parallel tracks**:
1. **Phase A polish (§1–§9c)** — narrow scope + three quality fixes (phrase-quantize, stem-swap, humanize) + sample lib gating. Foundation that removes failure classes.
2. **GPU saturation expansion (§13)** — DPO-tune Director, K=32 best-of render, multi-model bridges, per-junction Director calls, deep ensemble analysis, style-RAG. Spends the 30+ hr of unused MI300X capacity on quality compounding.

> **Naming note**: `ARCHITECTURE.md` uses Phase A → E (A = current rule-based, B–E = trained / streaming targets). This document is **Phase A polish + accelerated dip into Phase B** (DPO + bridge gen). The "Phase 1 / Phase 2" terminology used below refers to **scope buckets within Phase A**, not the architectural phases.

---

## Doc reconciliation snapshot

State across docs (read 2026-05-09):

| Doc | Status | Authoritative for |
|---|---|---|
| `STATUS.md` | **current truth** (mid-session, MI300X live, v15) | Bugs, what works, cost ledger, file map |
| `ARCHITECTURE.md` | target architecture (Phase A→E) | Long-term north star, component decomposition |
| `HACKATHON.md` | original demo script | Demo flow, slide outline (15 techniques — pre-tier-vocab era) |
| `AGENTS.md` | **STALE** — describes 2-tier Director (`major\|minor`) | Deploy playbook, env vars (still valid); tier vocab (out of date) |
| `README.md` | onboarding | Dev workflow, AMD GPU pattern |
| `AiJockey.md` | full architecture reference | Code-level detail (1834 lines; not re-read here) |
| `docs/dj_research.md` | research grounding | DJ practice, mix forms, sound design |
| `docs/phase1_plan.md` | this doc | Phase A polish action plan |

**Drift to fix in Block D**: `AGENTS.md` tier table must be updated from `major\|minor` to current `minor/major/drop/cut/loop` (or to the Phase A polish 3-tier subset — see §2 below).

---

## Guiding principle

**Narrow the corpus → narrow the bug surface.** Every restriction here removes a class of failure. Demo wins on consistency, not coverage. Post-demo expansion picks up the rest of the research backlog.

---

## 1. Scope — what Phase 1 supports

### 1.1 Genre

**In**: house, tech-house, deep house, techno, melodic techno, trance, progressive, chillstep, lo-fi (instrumental), synthwave, retrowave, future bass (instrumental builds).

**Out (Phase 2+)**: hip-hop, trap, drill, DnB, dubstep, pop, rock, ballads, Bollywood/Punjabi vocal, Hindustani classical.

**Why**: 4-on-floor + ~110–130 BPM = predictable phrase grid, reliable beat tracking, clean stem separation, transitions match prior art.

### 1.2 Vocals

**Default**: instrumental mode. Demucs strips vocals; system mixes `drums + bass + other` only.

**Optional**: vocals-on toggle in advanced UI. Planner picks instrumental sections (intro/breakdown/outro) for junctions; vocals only play in segment body, never across overlaps.

**Demo MP3s**: instrumental mode only.

### 1.3 BPM range

Accept **100–135 BPM**. Outside = reject with clear error. No BPM jumps >5% within a single mix.

### 1.4 Inputs

| Constraint | Value |
|---|---|
| Min clips | 3 (auto-augment from library if user gives <3) |
| Max clips | 10 |
| Clip duration | 45 s – 6 min |
| Sample rate | any → resample to 44.1 kHz |
| Channels | mono up-mixed, stereo passthrough |
| Format | mp3, wav, flac, m4a |

### 1.5 Output

| Spec | Value |
|---|---|
| Length | 90–150 s default, slider 30–300 s |
| Junctions | 3–5 per mix |
| LUFS | −9 |
| True-peak | −1.0 dBTP |
| Format | MP3 192k (demo), WAV (download) |
| Arc presets | `build`, `peak`, `flat_low` only — `rollercoaster` deferred to Phase 2 |

### 1.6 Mix form

**Beatmix / blend only.** Mashup, megamix, scratch, open-format BPM-jumps → Phase 2.

---

## 2. Transition vocabulary — collapse 5 tiers → 3

Current 5-tier (`minor/major/drop/cut/loop`) is good design but `cut`/`loop` techniques (`pitch_bend`, `spinback`, `scratch_fill`, `loop_callback`) misfire on real material (STATUS bug #7).

### Phase 1 vocab

| Tier | Techniques | Material constraint |
|---|---|---|
| **minor** (smooth blend, low drama) | `eq_swap`, `volume_blend` | Always safe |
| **major** (audible structural change) | `filter_fade`, `drum_break`, `echo_out` | Always safe |
| **drop** (climax) | `build_riser_drop` | Both sides must be instrumental sections |

**Total**: 6 techniques. Director restricted to picking from these 3 tiers only.

**Cut / loop tiers**: code stays, gated off via env flag `AIJOCKEY_PHASE = 1`. Re-enable in Phase 2.

---

## 3. Quality fixes — the three that matter

These are the cheapest path from v15 ("audibly varied but mechanical") to "sounds good." Ranked by impact-per-hour.

### 3.1 Phrase quantization — biggest single win

**Problem**: junctions placed mid-phrase = chaos regardless of DSP polish (research §2, §5 fix #1).

**Fix**:
- After beat-track, group beats ×4 = bars, ×8 = phrase (32 beats).
- For every junction, snap `out_time` and `in_time` to nearest **8-bar phrase boundary** (16 in long segments).
- Hard requirement: incoming track's first downbeat lands on outgoing's phrase boundary.

**Where**: `src/planner.py` (junction selection), `src/execute.py` (cuepoint resolution).

**Side benefit**: predictable overlap windows fix STATUS bug #1 (render duration shortfall). 8-bar overlaps at 120 BPM = exactly 16 s, computable up-front.

### 3.2 Stem-aware role swap during overlap

**Problem**: subtractive vocal mute leaves phasey holes (STATUS bug #6). Two basslines / two kicks fight (research §2 frequency ownership).

**Fix**:
- During overlap window: outgoing plays only `melody + other` stems, incoming brings `drums + bass`. Then full handoff on phrase 1.
- Replaces full-mix overlap with stem-role overlap. No subtractive vocal mute needed — vocal stem just isn't played during overlap.

**Where**: `src/execute.py` overlap renderer. Already on Tier 1 plan in STATUS.md.

**Bonus**: enables a primitive form of §6 mashup. Free.

### 3.3 Humanization on accent FX

**Problem**: risers/impacts/snare-rolls fire on grid → "robot" feel (research §7.4 timing).

**Fix**: ±8 ms timing jitter, ±8 % velocity variance per accent FX trigger. ~10 lines.

**Where**: `src/execute.py` accent overlay.

---

## 4. Director constraints (Phase 1 prompt changes)

Update `src/director.py` SYSTEM_PROMPT:

- Tier vocabulary: `minor / major / drop` only (drop `cut`, `loop`).
- Hard rule: `drop` tier requires both segments to be instrumental sections — Director receives section labels per clip.
- Arc options: `build`, `peak`, `flat_low` only.
- Drama policy: same as current, but tier distribution caps drop count at 1–2 per mix (avoid drop spam).

---

## 5. Diagnosis hooks (defensive checks before render)

From research §5. Implement as planner-stage validators that reject bad plans:

1. **Off-phrase junction** → snap to grid (§3.1 above; defensive after).
2. **Kick clash** → if both sides have kick during overlap, force stem-swap (§3.2).
3. **Key clash** → out of scope Phase 1 (no Camelot detection yet). Mitigate via `same_genre_tight_mix` heuristic.
4. **Section role mismatch** → forbid `drop`-out-of-`drop`, forbid `breakdown`-into-`breakdown` (energy crater). Reroute via tier downgrade.
5. **Tier collapse** → already enforced (distinct DSP per tier).
6. **Over-effecting** → cap accent FX at 2 per junction.

---

## 6. What is explicitly OUT of Phase 1

Write down so we don't drift:

- Genres: hip-hop, DnB, dubstep, pop/rock, vocal-Bollywood, Indian classical
- Tiers: `cut`, `loop` (and their techniques: pitch_bend, spinback, scratch_fill, loop_callback)
- Mix forms: mashup, megamix, scratch, live remix, open format
- Camelot / key-aware harmonic mixing
- BPM jumps >5 %
- Generative bridges (MusicGen / Stable Audio) — audiocraft blocked anyway
- Sound design / synthesis layer
- Roles beyond DJ (composer/arranger/mixing-engineer/mastering automation)
- Custom FX synthesis (granular freeze, sidechain pumping) — Phase 2

---

## 7. Execution plan — 8–12 hour window

Maps to STATUS.md `Tier 1` and `Tier 3` blocks. `Tier 2` (DPO fine-tune) deferred until Phase 1 ships clean.

### Block A — quality fixes (4–5 hr)

1. **Phrase quantization** in planner + execute. Snap junctions to 8-bar boundaries. Verify against rendered output (overlay grid markers in debug).
2. **Stem-swap overlap** in execute.py. Swap outgoing-melody + incoming-drums-bass for the overlap window. Remove subtractive vocal mute path.
3. **Humanization** on accent FX. ±8 ms / ±8 % jitter.
4. **Render duration audit** — verify phrase-quantization fixes shortfall; if not, dynamic transition-bars sizing.

### Block B — scope enforcement (2 hr)

5. `AIJOCKEY_PHASE` env flag. Phase=1 → Director vocab restricted, cut/loop tiers gated off, only 3 arc presets exposed.
6. API input validation: BPM band, clip count, duration, format. Clear error responses.
7. Vocals toggle in API + UI (default off = instrumental mode).
8. Director prompt rewrite for Phase 1 vocab + section-aware drop rule.

### Block C — demo + deploy (2–3 hr)

9. Re-render 5 demo MP3s with Phase A polish engine. Curate clips that fit the genre band (§1.1). Replace pre-baked MP3s in `space/`.
10. HF Space deploy ([HACKATHON.md](../HACKATHON.md) flow). Set MI300X_URL, MI300X_KEY, ADMIN_PW secrets.
11. Smoke test through ngrok stable domain (`issue-slingshot-bobsled.ngrok-free.dev` per STATUS.md).
12. Snapshot final state as `aijockey-demo-ready`. Destroy live droplet to stop billing (per STATUS.md cost discipline).

### Block D — guardrails + doc sync (1 hr)

13. Diagnosis-hook validators in planner (§5 above). Reject bad plans before render.
14. Test pass: [tests/test_transition_mapping.py](../tests/test_transition_mapping.py) updated for 3-tier vocab.
15. **Sync stale docs**: update [AGENTS.md](../AGENTS.md) Director tier table to current vocab. Update [HACKATHON.md](../HACKATHON.md) Slide 4 "what's AI" — drop "madmom" (now librosa per STATUS.md), drop "Camelot" (deferred to Phase 2).
16. Update [STATUS.md](../STATUS.md) Tier 1 plan checkboxes as items complete.

---

## 8. Done-criteria for Phase 1

| Check | Pass condition |
|---|---|
| Demo MP3s | 5 fresh renders, all 90–150 s, all instrumental-band genre |
| Mechanical-feel test | Subjective: junctions don't jolt; accents don't grid-snap-feel |
| Render duration | Planned ≈ rendered (±10 %) |
| Vocal artifacts | No phasey holes (stem-swap path verified) |
| Failure modes | BPM/clip-count/duration violations return clean errors |
| Tier diversity | Each demo uses ≥2 distinct techniques |
| Deploy | HF Space live, ngrok tunnel stable, snapshot taken |

---

## 9. Phase 2 backlog (parking lot)

Order roughly by user-visible value:

1. **Re-enable cut + loop tiers** with material-aware gating (only on instrumental segments, only when phrase-aligned).
2. **Camelot key detection + harmonic compatibility scoring** in planner.
3. **Live mashup form**: full vocal-of-A over drums-of-B for N bars.
4. **DPO/ORPO LoRA fine-tune** of Qwen2-Audio Director on user preference labels.
5. **Open-format / BPM-jump** support via echo-out reroute.
6. **Wider genre support**: hip-hop, DnB, dubstep — needs tier-vocab expansion + new beat-track handling.
7. **Sidechain pumping** during forced kick overlap.
8. **Granular freeze** on outgoing's last bar as transitional pad.
9. **Reverb-tail bridging** at cut junctions.
10. **Megamix arc** — dense-cut macro mode.
11. **Generative bridges** (MusicGen) once audiocraft / ROCm install resolved.
12. **Mix critic v2** — codec-invariant CLAP-based, longer windows, retrained on bigger DJ corpus.

---

## 9b. Quality fixes — code-level detail

Section 3 named the three fixes. This section is the implementation map. Recon showed lots of scaffolding already exists — Phase 1 is mostly *wiring*, not new code.

### 9b.1 Phrase quantization

**Existing scaffolding**:
- [src/phrase.py:10](../src/phrase.py#L10) — `snap_to_phrase(t_sec, downbeats, bars_per_phrase=16)` already implemented.
- [src/phrase.py:22](../src/phrase.py#L22) — `detect_phrase_length` (16 vs 32 bar autocorr).
- [src/restricted_mode.py:66](../src/restricted_mode.py#L66) — wrapper. Hidden behind restricted-mode flag, NOT used by main planner.

**Edits required**:

1. **Planner** ([src/planner.py](../src/planner.py)) — after junction selection, snap both `out_time` and `in_time` to nearest 8-bar phrase boundary using each clip's `downbeats`. Reject candidate if snap moves time >2 bars (clip too short for boundary; pick different boundary).

   ```python
   from phrase import snap_to_phrase
   out_time = snap_to_phrase(out_time, clip_a['downbeats'], bars_per_phrase=8)
   in_time  = snap_to_phrase(in_time,  clip_b['downbeats'], bars_per_phrase=8)
   ```

2. **Execute** ([src/execute.py](../src/execute.py)) — make overlap window an integer phrase count (e.g. 8 bars at 120 BPM = exactly 16.0 s), not `bars × beat_dur` ad-hoc. Side benefit: fixes STATUS bug #1 (render duration shortfall).

3. **Analyze** ([src/analyze.py](../src/analyze.py)) — already imports `detect_phrase_length`. Verify per-clip phrase length is stored in cache; if not, store. Planner consumes this so 16-bar and 32-bar clips both align correctly.

4. **Validation** — planner-stage assertion that snapped times are on phrase boundaries. Debug logger for grid markers.

**Effort**: ~2 hr. **Risk**: low.

---

### 9b.2 Stem-aware role swap

**Existing scaffolding**:
- Stems loaded per segment ([src/execute.py:89](../src/execute.py#L89)).
- Stems dict: `drums`, `bass`, `vocals`, `other`.
- Subtractive vocal mute already at [src/execute.py:263-274](../src/execute.py#L263-L274) — to be REPLACED.

**Edit** — overlap renderer in [src/execute.py](../src/execute.py). Pseudocode:

```python
# during overlap window of N bars between prev (outgoing) and cur (incoming)
overlap_samples = int(phrase_bars * 4 * beat_dur * sr)

# outgoing keeps melody (other), drums faded to silence, vocals dropped
prev_other = prev['stems']['other'][:, -overlap_samples:]
prev_drums = prev['stems']['drums'][:, -overlap_samples:] * fade_out_curve
prev_overlap = prev_other + prev_drums
# NO subtractive vocal mute — vocals just not played in overlap

# incoming brings drums + bass, melody held until overlap end
cur_drums = cur['stems']['drums'][:, :overlap_samples] * fade_in_curve
cur_bass  = cur['stems']['bass'][:, :overlap_samples]  * fade_in_curve
cur_overlap = cur_drums + cur_bass
# cur 'other' silent during overlap, enters on phrase 1 post-overlap
```

**Why**: frequency-ownership rule (research §2) — outgoing owns mids (melody), incoming owns lows (drums+bass). No fight per band. No phase artifacts from subtractive mute.

**Edge cases**:
- Techno clips with no real "melody" → use `other + filtered drums tail`.
- Stems don't sum to mix exactly; that's fine, role-split is the goal.

**Effort**: ~3 hr. **Risk**: medium — fade curves need to be smooth.

**Validation**: A/B render same plan with old vs new overlap path. New version: no phasey holes, no double-kick mush.

---

### 9b.3 Humanization on accent FX

**Existing scaffolding**:
- Accent overlay at [src/execute.py:184](../src/execute.py#L184) `_overlay_accent_hint`.
- Sample bank in [src/samples.py](../src/samples.py).

**Edit** — inside `_overlay_accent_hint`:

```python
import random

def _overlay_accent_hint(out, cur, sample_bank, target_bpm, beat_dur):
    ah = cur['entry'].get('accent_hint')
    if not ah:
        return out
    sample = sample_bank.get(ah['fx_category'])
    if sample is None:
        return out

    base_offset = int(ah['beats'] * beat_dur * sr)

    # humanization — only for non-locked categories
    locked_categories = {'risers'}  # risers must align to phrase
    if ah['fx_category'] not in locked_categories:
        # seed for reproducibility
        rng = random.Random((cur['entry'].get('clip_id'), ah.get('junction_index', 0)))
        jitter_ms = rng.uniform(-8, 8)
        velocity = rng.uniform(0.92, 1.08)
    else:
        jitter_ms = 0
        velocity = 1.0

    pos = base_offset + int(jitter_ms * sr / 1000)
    end = min(pos + sample.shape[-1], out.shape[-1])
    out[..., pos:end] += sample[..., :end-pos] * velocity
    return out
```

**Rules**:
- `risers` locked to grid (must hit phrase 1).
- `impacts`, `snare_rolls`, `hihat_rolls`, `sweeps` get jitter.
- Per-genre tuning deferred to Phase 2 (techno tighter, house looser).
- RNG seeded by `(clip_id, junction_index)` for deterministic reproducibility across same-input renders.

**Effort**: ~30 min. **Risk**: low — isolated function.

---

### 9b.4 Order of operations

| # | Fix | Hr | Risk | Why this order |
|---|---|---|---|---|
| 1 | Humanization | 0.5 | low | Cheapest, builds confidence, isolated change |
| 2 | Phrase quantization | 2 | low | Functions already exist, just wire in |
| 3 | Stem-swap overlap | 3 | medium | Biggest blast radius on render path; do last |

**Total**: ~5.5 hr. Fits Block A in §7.

---

### 9b.5 Existing infra to reuse (don't reinvent)

| File | What's there | Phase 1 use |
|---|---|---|
| [src/phrase.py](../src/phrase.py) | `snap_to_phrase`, `detect_phrase_length` | Wire into planner + execute |
| [src/restricted_mode.py:66](../src/restricted_mode.py#L66) | Snap wrapper | Reference pattern |
| [src/execute.py:89](../src/execute.py#L89) | Stems loaded per segment | Use directly in overlap renderer |
| [src/execute.py:184](../src/execute.py#L184) | `_overlay_accent_hint` | Add jitter inside |
| [src/camelot.py](../src/camelot.py) | Key detection | Defer to Phase 2 |

Phase 1 = wiring + small edits in existing files. Zero new modules required.

---

## 9c. Sample / FX library — pollution fix

### Problem

Existing lib at [samples/memes/](../samples/memes/) holds 34 wav files. Two buckets:

- **DJ-useful (~5)**: `riser_short`, `subdrop`, `impact_boom`, `record_scratch`, `vinyl_rewind`
- **Meme/novelty (~29)**: `vine_boom`, `airhorn`, `bruh`, `mlg_hitmarker`, `oof`, `damn`, `sad_violin`, `dramatic_chord`, `crowd_cheer`, `party_horn`, `siren`, `thunder`, `tabla_hit`, etc.

Director emits `accent_hint{fx_category: impacts}`; [src/samples.py:102](../src/samples.py#L102) `_best_match` scores by BPM + length only, no aesthetic tag. `vine_boom` and `impact_boom` both classified as `impacts` → bank picks whichever fits length → **meme drops into a tech-house mix**. That's the pollution.

### Three root causes

1. **Aesthetic OOD** — meme samples are not in the DJ-mix distribution Director was prompted/trained against.
2. **Bank routing is blind** — no filter between meme and DJ FX inside same category.
3. **Director over-triggers accents** — fun-looking lib encourages accent spam.

### Phase 1 fix — keep good, gate bad

Combine three small edits:

#### Edit 1 — curate manifest

Reduce [samples/manifest.json](../samples/manifest.json) to the 5 DJ-useful entries. Meme files stay on disk (Phase 2 will reintroduce as deliberate callbacks), just not visible to SampleBank loader.

#### Edit 2 — defense-in-depth allow-list

[src/samples.py](../src/samples.py) `SampleBank.__init__` gets `allowed_types` param. Even if manifest had garbage, accent path can only pull whitelisted categories.

```python
PHASE1_ALLOWED_TYPES = {
    'risers', 'impacts', 'sweeps', 'snare_rolls', 'hihat_rolls', 'subdrops'
}

def __init__(self, samples_dir='samples', allowed_types=None):
    self.samples_dir = Path(samples_dir)
    self.allowed_types = allowed_types
    self.bank = {}
    self._load()

def _load(self):
    ...
    for entry in manifest:
        if self.allowed_types and entry['type'] not in self.allowed_types:
            continue
        ...
```

Phase 1 caller (in `execute.py` setup) constructs with `allowed_types=PHASE1_ALLOWED_TYPES`. Phase 2 callers can omit param to access full lib.

#### Edit 3 — Director prompt restraint

[src/director.py](../src/director.py) SYSTEM_PROMPT addendum:
- "Use accent_hints sparingly. Max 2 per junction."
- "Prefer no accent over a forced accent."
- "Only request accents at structurally significant junctions (`major` or `drop` tiers)."

This caps quantity. Combined with Edit 2 capping quality, pollution is fully gated.

### What gets deferred to Phase 2

- **Meme/novelty samples** stay in repo at `samples/memes/`, off the accent path.
- **Callback-budget mechanism** (Director's `callback_budget` field) becomes the explicit meme-trigger path.
- **New arc preset** `party_meme` (Phase 2) sets `callback_budget=1-2`, enables meme samples via a separate `surprise_callback` event class.
- **Synth fallback** ([src/synth_fx.py](../src/synth_fx.py)) covers any whitelisted category that has no real sample.

### Effort

- Edit 1: 10 min (manifest curation).
- Edit 2: 20 min (init param + filter + Phase 1 wiring).
- Edit 3: 5 min (prompt edit).

**Total ~35 min.** Slot inside Block B (scope enforcement).

### Validation

Render 3 demos with new gating. Confirm:
- Zero meme triggers across all junctions.
- Director still produces `accent_hint` entries for risers/impacts/snare_rolls.
- Synth fallback fires when no real sample matches whitelisted type at requested length/BPM.

---

## 9d. Architecture-doc alignment

How Phase A polish maps onto [ARCHITECTURE.md](../ARCHITECTURE.md) component table:

| ARCHITECTURE.md component | Phase A current | Phase A polish change |
|---|---|---|
| Director | `planner.py` beam search + `director.py` Qwen2-Audio | Restrict tier vocab to 3 (`minor/major/drop`); section-aware drop rule; cap accent_hints at 2/junction |
| DJ LLM core | none (direct execute) | unchanged — Phase B+ work |
| Decoder | rubberband + numpy mix | unchanged |
| Selector | `pick_segment` + Tier 1 classifier | unchanged in Phase A polish (operates within scope-restricted clip set) |
| Mixer | `transitions.py` 15 techniques | Phase A polish uses **6 techniques** out of 15. Other 9 gated off via `AIJOCKEY_PHASE=1` flag. Stem-swap overlap path replaces subtractive vocal mute |
| Critic | `eval.py` post-hoc | unchanged — mix critic remains unused in rerank (STATUS.md notes 0.77 val acc → unreliable) |
| Pool index | `cache/*.json + CLAP npz` | unchanged |
| Sample bank | `samples.py + synth_fx.py` + 34-file meme lib | Whitelist filter to 6 DJ-FX categories; meme lib gated off |

**No architecture changes.** Phase A polish is purely **narrowing scope + wiring existing scaffolding** ([src/phrase.py](../src/phrase.py), [src/restricted_mode.py](../src/restricted_mode.py), Demucs stems already loaded).

---

## 9e. Cost + risk context

Per [STATUS.md](../STATUS.md) cost ledger:

| Bucket | Status |
|---|---|
| Remaining AMD credits | ~$80 of $100 |
| Droplet rate | $1.99/hr |
| Phase A polish budget | ~5.5 hr fix work + ~3 hr demo/deploy = ~8.5 hr at $1.99/hr ≈ **$17** |
| Buffer | ~30 hr droplet uptime remaining for Phase 2 / training |

**Risk register**:

| Risk | Likelihood | Mitigation |
|---|---|---|
| Stem-swap fade curves audibly worse than current | medium | A/B render same plan both paths; revert if worse |
| Phrase quantization drops too many candidate junctions | low | Snap-window tolerance ±2 bars; fall back to nearest valid boundary |
| Humanization jitter on `risers` causes phase-1 miss | low (gated) | Risers locked-to-grid by category whitelist |
| Director still picks unsuitable techniques despite 3-tier restriction | medium | Add planner-stage validator (§5 hooks) — reject + retry |
| HF Space deploy breaks ngrok tunnel | medium | Per [server/TUNNEL_RUNBOOK.md](../server/TUNNEL_RUNBOOK.md), only update `MI300X_URL` secret |
| User uploads out-of-scope genre during demo | high | API input validator (§1.4) returns clean error; fall back to library auto-augment |

---

## 13. GPU saturation track — best-output offline rendering

### 13.1 Reframe

**Original framing** (§1–§10): ship narrow demo, runtime/cold-start matters.

**New framing** (this section): *demo is pre-rendered MP3s*. Cold-start irrelevant. One render can take 30 min. **Optimize for output quality, not throughput.**

Per [STATUS.md](../STATUS.md): MI300X 192 GB VRAM, current pipeline uses ~20 GB (≈10 % utilization). $80 of $100 credits remain ≈ 40 hr droplet uptime. Polish work alone consumes 8.5 hr → ~31 hr unused capacity worth ~$60. Spend it on quality.

### 13.2 What changes when "demo speed" stops mattering

| Constraint | Demo-mode | Offline-best-output mode |
|---|---|---|
| Director model size | 7B (fits cold start) | **72B** (Qwen2.5-72B-Instruct or Qwen2-Audio-72B) |
| Director calls per mix | 1 (whole-plan) | **Per-junction** with full context window |
| N-best K | 3–5 (timeline-only score) | **K=16–32 full audio renders** |
| Bridge generation | none / synth | **Multi-model ensemble** (Stable Audio Open + MusicGen-Large + AudioLDM2) |
| Analysis passes | 1 stem model + 1 beat tracker | **Ensemble**: 2 stem models + 2 beat trackers + 2 key detectors |
| Humanization seeds | 1 | **4–8 per render**, critic picks best feel |
| Iteration | one-shot | **Render → preference label → DPO → re-render** loop |
| Output length | 90–150 s | **Up to 30 min** pre-baked sets |

### 13.3 Six expansion tracks

Ranked by impact-per-hour. All run **on top of** Phase A polish (§3 fixes still apply — they aren't replaced, they're foundation).

#### Track 1 — DPO/ORPO LoRA Director, then iterate

- **Base**: Qwen2-Audio-7B (current). Or upgrade to Qwen2-Audio-72B if VRAM headroom permits at training time.
- **Data**: render K=8 candidates per prompt across 30 prompts → 240 candidates. Pairwise label via human listen + critic v2 score → ~120 preference pairs.
- **Train**: LoRA DPO ~4 hr on MI300X.
- **Iterate**: re-render → re-label top vs bottom → second DPO pass.
- **Compounds**: every iteration improves Director taste. Best long-term investment.
- **Effort**: 4 hr train × 2 iterations + ~3 hr labeling = ~11 hr.

#### Track 2 — Mix critic v2 (codec-invariant, big)

- **Replace v1's mel features with CLAP embeddings** (already a dependency, codec-invariant).
- **Window**: 30 sec, not 8 sec. Catches transition coherence.
- **Negatives**: include current v15 outputs labelled vs real DJ-set windows. Domain-aligned, not synthetic splices.
- **Scale**: train on 100 k samples (was 1.6 k). Real DJ corpus + self-play.
- **Train**: 4 hr on MI300X.
- **Output**: reliable rerank signal → unblocks best-of-K (Track 3) and self-play data (Track 1).

#### Track 3 — K=32 parallel render farm + critic rerank

- **Current**: 1 plan → 1 render. N-best is timeline-level score only.
- **New**: 1 plan → render 32 variants in parallel (different humanization seeds, different stem-mix ratios, different accent FX placements). Critic v2 ranks. Top 3 surfaced for human pick.
- **Hardware**: MI300X 192 GB easily holds 32 concurrent renders (~6 GB each).
- **Effort**: 3 hr to wire parallel renderer + critic batch-scoring.
- **Output**: every released MP3 is best-of-32, not best-of-1.

#### Track 4 — Multi-model bridge ensemble

- **Skip audiocraft** (ROCm install fails per STATUS.md bug #11). Use HF `diffusers` and `transformers` directly:
  - **Stable Audio Open 1.0** via `diffusers` — 47 sec audio, text-conditioned, ROCm-clean.
  - **MusicGen-Large** via `transformers` — 30 sec stereo, melody-conditional.
  - **AudioLDM2** via `diffusers` — high-quality short generations.
- **Generate 3 bridge candidates per incompat-key/BPM junction**, critic picks.
- **Use case**: covers gaps where current `cut` / `echo_out` sound rough.
- **Effort**: ~5 hr install + integrate + critic-rank wire.

#### Track 5 — Per-junction Director calls

- **Current**: Director called once per mix, emits whole-plan JSON.
- **New**: Director called per junction with full context — last 16 sec of rendered audio + clip pool + arc state + remaining budget. Picks tier + technique + accent_hints just for THIS junction.
- **Why**: matches research §4 streaming protocol from [ARCHITECTURE.md](../ARCHITECTURE.md). Decisions adapt to actual rendered output, not plan-time guesses.
- **Cost**: more LLM calls per mix (~10–15 vs 1). Fine — offline mode.
- **Effort**: ~4 hr to refactor planner into per-junction loop + Director state tracking.
- **Bonus**: if upgraded to 72B, taste jump is dramatic on per-junction reasoning.

#### Track 6 — Deep multi-pass analysis

- **Stems**: run **htdemucs + htdemucs_ft** (or htdemucs + mdx-extra), ensemble by averaging or pick-best-per-frame. Better stem separation = better stem-swap quality (§9b.2).
- **Beat track**: run **librosa + ensemble of beat-this / BeatNet** (newer 2024+ models). Median across detectors. Catches edge cases librosa misses.
- **Key detection**: **Essentia + custom CLAP-key classifier**. Two opinions per clip.
- **Phrase length**: existing autocorr per [src/phrase.py:22](../src/phrase.py#L22) + ensemble vote.
- **Camelot scorer**: wire existing [src/camelot.py](../src/camelot.py) into planner — incompat key pairs route to Track 4 bridges instead of direct blend.
- **Effort**: ~5 hr install + ensemble glue.
- **Output**: planner has reliable per-clip metadata for *every* decision.

#### Track 7 — Style-RAG activation (light, big lift)

- **Already exists**: [src/style_rag.py](../src/style_rag.py).
- **Embed**: all transitions from 25-set DJ corpus → CLAP-embedding vector index.
- **Use**: per junction, retrieve nearest 3 real-DJ transitions as Director's in-context few-shot examples.
- **Effect**: Director's choices conditioned on what real DJs actually did at similar pre/post audio context. Massive grounding.
- **Effort**: 2 hr — corpus segment + embed + retrieval wire.

### 13.4 Re-budgeted plan with expansion

| Block | Track | Hr | $ at $1.99/hr | Parallel? |
|---|---|---|---|---|
| A polish | quality fixes (§3) | 5.5 | 11 | sequential dev |
| B scope | sample whitelist, scope flag, Director prompt | 2 | 4 | sequential |
| D doc sync | AGENTS / HACKATHON / STATUS updates | 1 | 2 | sequential |
| **E** | DPO Director ×2 iterations + labeling | 11 | 22 | E2 train runs in BG |
| **F** | Critic v2 retrain (CLAP-feat, 100k samples) | 4 | 8 | runs in BG |
| **G** | K=32 parallel render + critic rerank | 3 | 6 | needs F |
| **H** | Multi-model bridge ensemble (Stable Audio + MusicGen + AudioLDM2) | 5 | 10 | parallel install |
| **I** | Per-junction Director loop | 4 | 8 | needs E1 done |
| **J** | Deep multi-pass analysis (stems + beats + key + Camelot wire) | 5 | 10 | parallel re-analyze |
| **K** | Style-RAG activation | 2 | 4 | parallel |
| **L** | Final batch render of 10–20 demo mixes (offline, 30-min sets) | 3 | 6 | sequential |
| | **TOTAL** | **45.5** | **~$91** | — |

**$91 vs $80 remaining** — tight. Three options to fit:

1. **Drop Track 1 second iteration** (saves 4 hr / $8 → $83). DPO pass once instead of twice.
2. **Drop AudioLDM2 from bridge ensemble** (saves 1.5 hr / $3 → $88). Keep Stable Audio + MusicGen only.
3. **Skip Track 5 per-junction Director if 7B can't reason that well** — defer to post-DPO.

**My pick**: take option 1 first (single DPO pass; iterate later if time permits). Net ~$83. Within budget if accepting small overrun.

### 13.5 Sequencing — saturate the GPU

Goal: never have GPU idle while there's queued work.

```
T0 ─────────────────────────────────────────────────────────►  T+45h

Foreground (sequential dev, ~10 hr):
  A polish (5.5h) ──► B scope (2h) ──► I per-junction Director (4h) ──►
                                       │
                                       L final batch render (3h, end)

Background GPU jobs (start ASAP, finish in parallel):
  J deep analysis re-run on full clip pool (5h) ───────────►
  F critic v2 train (4h)         ────────►
  E1 DPO iter-1 train (4h)              ─────────►
  K style-RAG embed (2h) ──►
  H bridge ensemble install + smoke (5h) ─────────►

Dependencies:
  G K=32 render farm  ──── needs F done ────► (3h)
  E2 DPO iter-2       ──── needs E1+G output for labels ───► (after L)
```

**Result**: foreground dev never blocked on training. GPU runs 60-80 % sustained instead of 10 % bursts.

### 13.6 New risks (offline-best-output mode)

| Risk | Likelihood | Mitigation |
|---|---|---|
| 72B Director slower per call → per-junction loop too slow even offline | medium | Stick with 7B for per-junction; reserve 72B for whole-plan reasoning + DPO data labels |
| Stable Audio Open / MusicGen ROCm install fails | medium | Try `diffusers` first (cleanest); fall back to `transformers`; if both fail, ship without bridges |
| K=32 parallel renders OOM if pool clips are long | low | Cap concurrent K by VRAM probe; 192 GB / 6 GB ≈ 32 fits comfortably |
| Critic v2 still unreliable after retrain | medium | Validate on held-out real-DJ pairs before promoting to rerank gate |
| DPO overfits 30-prompt training set | medium | Use 60+ prompts; augment with paraphrase; LoRA rank ≤ 16 |
| Multi-pass analysis disagrees on key/beat → planner stalls | low | Median voting; tie-break by CLAP confidence |
| Total work overshoots $80 budget | high | Track 1 iter-2 is the cut line; drop it if budget tightens |

### 13.7 Quality estimate (this version vs before)

| Output dimension | v15 current | After §3 polish only | After §3 + §13 expansion |
|---|---|---|---|
| Mechanical-feel | high | medium | **low** (Track 3 best-of-32 + Track 1 DPO) |
| Kick clash | occasional | rare | rare |
| Genre-aware tier choice | weak | medium | **strong** (Track 7 style-RAG + Track 5 per-junction) |
| Bridge quality on incompat pairs | bad | bad | **good** (Track 4 generative ensemble) |
| Director taste | "ok" | "ok" | **DPO-tuned to user prefs** (Track 1) |
| Per-clip metadata reliability | medium | medium | **high** (Track 6 ensemble analysis) |
| Mix duration | 90–150 s | 90–150 s | **up to 30 min** (Track L offline batch) |
| Output selection | best-of-1 | best-of-1 | **best-of-32, critic-ranked** |

### 13.8 Output deliverable shift

Original deliverable: HF Space serving live `/generate` requests.

New deliverable: **pre-rendered batch of 10–20 best-effort mixes**, each one best-of-32 of a tuned-Director / bridge-augmented / multi-pass-analyzed pipeline. Hosted as static MP3s. HF Space optional ("Try It" tab can stay password-gated, low-priority).

Demo story shifts from "live AI DJ" to: **"Each track in this set was selected from 32 critic-ranked candidates rendered by a DPO-tuned Director that hears every clip, retrieves real DJ transitions as in-context examples, and uses a 3-model generative ensemble to bridge incompatible-key gaps. Output is what comes out at the top of the rank."**

Stronger story. Stronger artifact. Same budget.

---

## 14. State-of-the-art alignment — patterns from big AI labs

This section answers: **how do well-funded labs solve adjacent problems, and what can we steal?** Constraints: open-weight / permissive license only, free or scraped-legally data, runs on MI300X.

### 14.1 What big labs do for music/audio gen

| System | Pattern | Useful to AiJockey? |
|---|---|---|
| **Suno / Udio** | End-to-end AR over discrete audio tokens, ~5-min songs | ✗ closed weights, ✗ doesn't use user pool |
| **MusicGen** (Meta) | AR transformer over EnCodec tokens, text-conditioned | ✓ MIT code, weights NC — usable for *research* fine-tune |
| **Stable Audio Open 1.0** (Stability) | Latent diffusion, text + duration conditioned, 47 s | ✓ permissive (Stability Community License) |
| **AudioLDM2** (Microsoft) | Latent diffusion, dual cond (text + retrieval) | ✓ MIT |
| **MusicLM** (Google) | Hierarchical: semantic w2v-BERT → coarse → fine SoundStream | ✗ no public weights; pattern reusable |
| **Jukebox** (OpenAI, 2020) | VQ-VAE + AR prior, multi-scale | architecturally instructive only |
| **Spotify auto-DJ** | Transition pattern library + sequence model | pattern: programmatic transitions, not generative |
| **Mubert / Endel** | Procedural synth + sample bank + agent | pattern matches our hybrid approach |
| **Anthropic / OpenAI alignment** | RLHF, Constitutional AI, preference modeling | ✓ apply same loop to mix taste |
| **DeepMind AlphaGo / AlphaZero** | Self-play, critic-as-reward, MCTS | ✓ apply: self-play renders + critic ranking |
| **OpenAI o1 / DeepSeek-R1** | Test-time compute scaling — more inference per query | ✓ apply: K=64 best-of, per-junction reasoning |
| **Mixtral / GPT-4** | Mixture-of-Experts routing | ✓ apply: per-genre LoRA Director adapters |
| **Cursor / Devin / Claude Code** | Tool-using agents with critic loops | ✓ matches ARCHITECTURE.md Director-as-agent target |

### 14.2 Patterns we steal — ranked by impact

#### Pattern P1 — Self-play data generation (AlphaGo-pattern)

**Problem**: only 25 real DJ sets. DPO needs 100s of preference pairs. User listening to 100s of mixes = bottleneck.

**Pattern**: system generates K mixes per prompt × N prompts. Critic v2 ranks. Top-decile vs bottom-decile = automatic preference pair.

**Concrete**:
- 60 prompts × K=8 renders = **480 candidates** in ~10 hr GPU.
- Critic v2 ranks → 60 × (best, worst) pairs = **60 strong DPO pairs**.
- Augment with human-in-the-loop on 30 borderline pairs.
- Combined: ~90 pairs = enough for LoRA DPO.

**Cost**: ~10 hr render + ~2 hr critic-rank + ~2 hr human spot-check = 14 hr.

#### Pattern P2 — Hierarchical generation (MusicLM-pattern)

**Problem**: current pipeline = Director (text plan) + DSP (audio mix). Gap in middle: no model that *synthesizes* audio for incompat-pair bridges.

**Pattern**: cascade three levels.
- **L1 semantic** = Director JSON (already have).
- **L2 generative bridge** = MusicGen-Small or Stable Audio Open, conditioned on (last 4 sec of out-going + first 4 sec of in-coming) → 4-sec bridge.
- **L3 DSP** = current execute.py.

**Concrete**:
- MusicGen-Small (300M params) **fine-tuned** on (pre, transition, post) triplets from real DJ corpus.
- ~6 hr training on MI300X with self-supervised reconstruction loss.
- Inference: invoke only on `cut` / `echo_out` junctions where direct blend fails.

**Cost**: ~6 hr fine-tune + ~3 hr integration = 9 hr.

#### Pattern P3 — Constitutional AI (Anthropic-pattern)

**Problem**: DPO is soft preference. System still violates hard musical rules (kick clash, off-phrase, key clash).

**Pattern**: hard constraints layer **above** soft preference. Director outputs candidate plans; constitutional validator rejects any that violate rules; only valid plans go to render.

**Rules to encode**:
1. Junctions on phrase grid (8/16/32 bar boundaries).
2. No two kicks playing simultaneously > 1 bar.
3. Camelot key compat for blends >16 bars (force bridge if violated).
4. No `drop` tier when either side is breakdown/intro.
5. ≤2 accent FX per junction.
6. BPM ratio ≤1.06 within mix.

**Concrete**: small Python rule engine in planner. Reject + retry up to 3 times. Already partially scaffolded as §5 diagnosis hooks — extend.

**Cost**: ~3 hr to formalize rules + wire validator.

#### Pattern P4 — Test-time compute scaling (o1-pattern)

**Problem**: more inference = more chances at quality. We're using 1× compute per output.

**Pattern**: scale K (best-of-K) until quality plateaus. o1/R1 spend 100× more compute per query.

**Concrete**:
- K=8 → K=32 → K=64. Critic ranks, human picks final.
- Per-junction Director with **majority-vote across 3 calls** (different seeds, same prompt).
- For final batch: K=64 per mix × 20 mixes × ~2 min/render at K=64 parallel ≈ ~40 hr GPU. **Schedule overnight.**

**Cost**: GPU-only, no dev. Already covered in Track 3 budget; just scale K.

#### Pattern P5 — Retrieval-augmented generation (RAG)

**Problem**: Director's choices are zero-shot per pool. Has no access to "what real DJ did in similar context."

**Pattern**: index real DJ transitions (audio + metadata + technique label). Per junction, retrieve top-3 nearest as Director's in-context examples. Already planned as Track 7.

**Concrete extensions**:
- Embed via CLAP (audio) **and** text-caption (Qwen2-Audio-72B captions each transition).
- Hybrid retrieval: audio-similarity + caption-similarity.
- Director gets up to 5 examples per call.

**Cost**: ~3 hr (1 hr embed + 1 hr captioning + 1 hr retrieval wire).

#### Pattern P6 — Mixture of Experts via LoRA adapters

**Problem**: one Director taste is generic. House mix needs different taste than techno mix.

**Pattern**: train **per-genre LoRA adapters** on Qwen2-Audio-7B. Pool genre detected → load matching adapter.

**Concrete**: 4 genre buckets (house, techno, trance, hybrid). 4 LoRA trains × ~2 hr each = 8 hr.

**Cost**: 8 hr. **Defer to Phase B+** unless time permits — single DPO Director (Track 1) is the cheaper first move.

#### Pattern P7 — Curriculum DPO (Phi-pattern, textbook quality)

**Problem**: random preference pairs train slowly.

**Pattern**: stage learning. Stage 1 = obviously-bad-vs-good (real-DJ vs random-splice). Stage 2 = subtle (best-of-32 top-1 vs top-16). Stage 3 = nuanced (human-tagged surprise wins).

**Concrete**: split DPO data into 3 difficulty buckets, train sequentially with decreasing LoRA learning rate.

**Cost**: bundled into Track 1 DPO, no extra time.

### 14.3 Mass legal data ingest

Constraint: no big local disk. Direct-to-MI300X scratch (5 TB NVMe per STATUS.md).

| Source | Audio? | License | Size | Use |
|---|---|---|---|---|
| **Free Music Archive (FMA)** | ✓ direct mp3 | CC variants | ~100 k tracks (FMA-large) | Pre-train / pool augmentation |
| **MTG-Jamendo** | ✓ direct | CC | 55 k tracks, 195 GB | Pre-train, autotagging |
| **Mixotic.net** | ✓ direct mp3 | CC DJ mixes | ~700 sets | **Real DJ transition corpus** |
| **archive.org Live Music** | ✓ direct | varied (Creative Commons subset) | huge | Long-form mix data |
| **NSynth** (Magenta) | ✓ direct | CC | 300 k single notes | Sample bank augmentation |
| **MusicCaps** (Google) | metadata only | CC-BY | 5.5 k captions | Evaluation only |
| **AudioSet** | ✗ no audio (YouTube IDs only) | — | — | Skip |
| **HF datasets `music`** | varies | mixed | varies | Spot-check per dataset |
| **1001 Tracklists** | metadata only | T&C | huge | Tracklist labels for Mixotic alignment |

**No YouTube** per user constraint.

**Ingest plan**:
1. **FMA-medium** (25k tracks, ~22 GB) first — fits scratch disk easily, fast download. ~2 hr.
2. **Mixotic** ~700 sets — direct mp3, CC. **Highest priority — real DJ transitions.** ~3 hr.
3. **MTG-Jamendo** subset (filter by genre tags matching Phase 1 scope) — ~10 k tracks, ~40 GB. ~3 hr.
4. **NSynth** if accent FX bank needs expansion — defer.

**Total ingest**: ~8 hr download. Runs in background while dev happens.

### 14.4 AI-assisted data labeling

User constraint: no big local data → label on GPU.

**Patterns**:

1. **Auto-caption via Qwen2-Audio-72B**: every clip → "uplifting techno, 128 BPM, female vox 0:30-1:00, breakdown 1:30." Direct retrieval feature.
2. **Auto-tag via CLAP zero-shot**: classify genre/mood/energy from text labels. Cheap.
3. **Auto-detect transitions in DJ sets**: novelty + onset jumps → segment Mixotic sets into (pre, transition, post) triplets. Heuristic, then refined by critic.
4. **Auto-rank via critic v2**: every self-play mix gets a score. Top + bottom = DPO data.
5. **Auto-detect technique label**: heuristic on transition window (filter sweep? echo tail? cut?). Already partial in `transitions.py`.

**Cost**: ~5 hr to wire labeling pipelines. Then re-runnable on any new ingest.

### 14.5 Custom training opportunities (within $80 budget)

| Train job | Time | Cost | Output |
|---|---|---|---|
| **DPO LoRA Qwen2-Audio-7B Director** | 4 hr | $8 | Tuned taste |
| **MusicGen-Small fine-tune on transition triplets** (Pattern P2) | 6 hr | $12 | Generative bridges |
| **Mix critic v2** (CLAP feats, 100 k samples) | 4 hr | $8 | Reliable rerank |
| **Per-genre LoRA adapters** ×4 (Pattern P6) | 8 hr | $16 | Genre-aware Director |
| **Auto-caption pass over all corpus with Qwen2-Audio-72B** | 6 hr | $12 | Searchable metadata |
| **Style-RAG corpus embed** (Pattern P5) | 3 hr | $6 | In-context retrieval |
| **Constitutional rules engine** (Pattern P3, dev) | 3 hr | $6 | Hard-constraint validator |
| **TOTAL training jobs** | **34 hr** | **~$68** | |

Plus existing Phase A polish (Block A–D ~9 hr ~$18). Plus self-play render (Pattern P1, ~14 hr ~$28). Plus final K=64 batch render (~3 hr ~$6).

**Grand total: ~60 hr ~ $120.** Over budget.

### 14.6 Recommended cuts to fit $80

Pick what stays, what cuts. **My honest pick:**

#### Tier 0 — must-do (foundation, $26)
- Phase A polish §3 fixes (5.5 hr / $11)
- Sample lib gating §9c (0.5 hr / $1)
- Mass legal data ingest §14.3 (8 hr / $16, mostly bandwidth)

#### Tier 1 — biggest impact-per-dollar ($28)
- Mix critic v2 retrain (4 hr / $8) — unlocks everything downstream
- DPO LoRA Director, single iteration (4 hr / $8) — taste compounds
- Self-play data gen via critic-rank (10 hr / $20 — labor cheap if critic reliable)

Wait — that math doesn't work, self-play uses GPU not just labor. Recompute. Self-play = 480 renders × ~1.5 min each / K=8 parallel = ~90 min wall = 1.5 hr GPU = $3. Plus critic-scoring pass = 0.5 hr = $1. **Self-play is cheap once critic is ready: ~$4.**

#### Tier 1 (revised) — $20
- Mix critic v2 retrain (4 hr / $8)
- Self-play data generation (2 hr / $4)
- DPO LoRA Director iter-1 (4 hr / $8)

#### Tier 2 — quality multiplier ($24)
- Style-RAG with auto-captioning (3 hr / $6)
- MusicGen-Small bridge fine-tune (6 hr / $12)
- Constitutional rules engine (3 hr / $6)

#### Tier 3 — final output ($10)
- K=64 best-of batch render of 20 mixes overnight (5 hr GPU / $10)

**Adjusted total: $26 + $20 + $24 + $10 = $80 exactly.** Fits.

#### Cuts (not doing this round)
- Per-genre LoRA adapters (Pattern P6) — Phase B+
- DPO iter-2 (refinement) — Phase B+
- Auto-caption full corpus with 72B — too expensive at GPU rate; trim to top 10 k clips only
- AudioLDM2 in bridge ensemble — MusicGen-Small + Stable Audio Open is enough

### 14.7 Architectural alignment with ARCHITECTURE.md

This work bridges Phase A → Phase B in the [ARCHITECTURE.md](../ARCHITECTURE.md) roadmap. Component-level mapping:

| ARCHITECTURE.md component | Current | After §14 work |
|---|---|---|
| Director | rule + Qwen2-Audio-7B (zero-shot) | **DPO-tuned + RAG-augmented + constitutional-validated** |
| DJ LLM core | none | **MusicGen-Small fine-tuned for bridges** (partial Phase B) |
| Selector | rule | unchanged |
| Mixer | rule (15 techniques) | + bridge insertion + best-of-64 |
| Critic | rule (eval.py) | **CLAP-feat critic v2, used as reward signal** |
| Pool index | CLAP npz | + Qwen-audio captions + style-RAG index |

**Tier roadmap** (ARCHITECTURE.md §"Training roadmap") progress:
- Tier 1 ✓ classifier (already done)
- Tier 2 ⊳ in-progress → completes with critic v2
- Tier 3 ⊳ in-progress → completes with MusicGen-Small fine-tune
- Tier 4 (DJ LLM core) — out of scope, weeks/$$$ of work
- Tier 5 (RLHF) — partial via DPO

So §14 advances us 2.5 architectural tiers in $80.

### 14.8 New deliverable specification

**Final output**: 20 pre-rendered MP3s, hosted statically on HF Space.

For each mix:
- Length 90–600 s (slider per mix; some short demos, some 10-min sets)
- Best-of-64 critic-ranked
- Director DPO-tuned + RAG-augmented + constitutionally-validated
- Bridges generated via MusicGen-Small for incompat junctions
- 60+ % MI300X utilization sustained during production phase
- Released with **per-mix metadata card**: arc, prompt, tier sequence, retrieved RAG examples, constitutional violations rejected, critic score, K-rank winner.

**Pitch story** (final form):
> "Hybrid agent + multi-model audio system. Tuned via DPO on self-play preference data ranked by a CLAP-feature critic trained on 700 real CC DJ sets. Director retrieves real DJ transitions as in-context examples. Hard musical rules enforced as constitutional validators. Generative bridges via fine-tuned MusicGen-Small fill incompat-key/BPM gaps. Each released mix is best-of-64. AGPL-3.0, $80 of MI300X compute, no proprietary data, no YouTube — only Free Music Archive + Mixotic CC corpus + MTG-Jamendo."

This is roughly the technical depth Stability/Mubert ship. Done in $80.

---

## 15. Pipeline-parallel execution — keep GPU saturated

User constraints:
- **No local downloads** — all data lands on MI300X (no laptop disk).
- **Never wait** — GPU must process whatever's currently available, not block on full data.

Solution: producer-consumer pipeline with **file-based queues**. Every stage reads from "what's available now," produces output to a staging dir, signals downstream via manifest. Multiple stages run concurrently in tmux. GPU saturated 60–80 % wall-clock instead of bursty 10 %.

### 15.1 Data flow architecture

```
Internet                MI300X scratch (5 TB)              GPU compute
─────────             ─────────────────────             ──────────────────
Mixotic CC ──┐                                          ┌─► analyze stage
FMA       ──┼─► /scratch/raw/{src}/   (queue 0)          │   (Demucs + CLAP
MTG       ──┘        │                                   │    + beat + key)
                     │ watch                             │
                     ▼                                   │
              /scratch/cache/{id}.json   (queue 1)──────►│─► embed stage
                                                         │   (CLAP, captions)
                     │                                   │
                     ▼                                   │
              /scratch/labels/{id}.json  (queue 2)──────►│─► train stage
                                                         │   (critic, DPO)
                     │                                   │
                     ▼                                   │
              /scratch/models/{epoch}.pt (queue 3)──────►│─► self-play
                                                         │   (render + rank)
                     │                                   │
                     ▼                                   │
              /scratch/preferences/*.jsonl (queue 4)────►│─► DPO retrain
                                                         │
                                                         └─► final batch
                                                             (K=64 render)
```

Every queue is a directory + JSON manifest. Stage N downstream watches queue N upstream via file mtime / inotify / 30-sec polling.

### 15.2 Stage definitions

Each stage is a **long-running tmux process** that reads its input queue, processes new items, writes to output queue, repeats forever (until killed when batch is final).

| Stage | Reads | Writes | Concurrency | Idempotent? |
|---|---|---|---|---|
| **S0 download** | URL list | `/scratch/raw/{src}/{id}.mp3` | 4 parallel curl/aria2 | yes (skip if file exists) |
| **S1 analyze** | `/scratch/raw/` | `/scratch/cache/{id}.json` (stems + beats + key + CLAP) | 4 GPU procs | yes (skip if cache exists + checksum match) |
| **S2 segment** | `/scratch/cache/` (DJ-set type only) | `/scratch/transitions/{set_id}/t{n}.json` (pre/transition/post triplets) | 1 CPU proc | yes |
| **S3 embed** | `/scratch/cache/` | `/scratch/embed/clap.npz`, `/scratch/embed/cap.json` (auto-captions via Qwen2-Audio-72B) | 1 GPU proc (72B is big) | append-only |
| **S4 critic-train** | `/scratch/transitions/` + `/scratch/cache/` | `/scratch/models/critic_v2_e{N}.pt` | 1 GPU proc | resume from latest checkpoint |
| **S5 self-play** | `/scratch/cache/` + latest critic | `/scratch/renders/{run}/k{i}.wav` + score | K=8 GPU procs | resume by skipping done runs |
| **S6 pref-build** | `/scratch/renders/` + critic | `/scratch/preferences/iter{n}.jsonl` | 1 CPU proc | append |
| **S7 DPO-train** | `/scratch/preferences/` | `/scratch/models/director_dpo_e{N}.pt` | 1 GPU proc | resume |
| **S8 bridge-train** | `/scratch/transitions/` | `/scratch/models/musicgen_small_ft_e{N}.pt` | 1 GPU proc | resume |
| **S9 final-render** | latest models + curated prompts | `/scratch/output/mix{n}.mp3` | K=64 GPU procs (overnight) | yes |

**No stage waits for "all" data.** S1 starts as soon as S0 has 1 file. S4 starts as soon as S2 has 100 transitions. S7 starts as soon as S6 has 30 pairs. Etc.

### 15.3 Concurrent tmux session layout

8 tmux windows, each owning one stage:

```
session: aijockey
├─ window 0: dev shell (foreground edits)
├─ window 1: S0 download   (aria2 + curl loop, 4-conn parallel)
├─ window 2: S1 analyze    (worker pool, multiprocessing pickup)
├─ window 3: S2 segment    (CPU-bound, slot for low priority)
├─ window 4: S3 embed      (GPU, runs after S1 produces enough)
├─ window 5: S4 critic     (GPU, retrains every hour as data grows)
├─ window 6: S5 self-play  (GPU, K=8 renders)
├─ window 7: S7 DPO        (GPU, retrains as preferences accumulate)
├─ window 8: S8 bridge     (GPU, MusicGen-Small FT)
└─ window 9: monitor       (rocm-smi + queue depth dashboard)
```

Each window runs a Python script in `while True: process_new(); sleep(30)` loop. Crash = restart that window only, not whole pipeline.

### 15.4 GPU sharing strategy

MI300X is one GPU but 192 GB VRAM. Multiple processes can share via:

#### Option A — single-process with intra-process scheduler
One Python process, asyncio event loop, dispatches jobs from queue. Cleanest but couples crashes.

#### Option B — multi-process with VRAM partitioning *(recommended)*
- Each tmux window = separate `python` process.
- Set `HIP_VISIBLE_DEVICES=0` for all (one physical GPU).
- Each process pre-allocates VRAM budget at startup via `torch.cuda.set_per_process_memory_fraction(frac)`.
- Reserved budgets:

| Process | VRAM cap |
|---|---|
| S1 analyze (Demucs + CLAP) | 30 GB |
| S3 embed (Qwen2-Audio-72B for captions) | 80 GB |
| S4 critic train | 20 GB |
| S5 self-play render (per worker × K=8) | 6 GB × 8 = 48 GB |
| S7 DPO train (Qwen2-Audio-7B + LoRA) | 30 GB |
| S8 bridge train (MusicGen-Small) | 15 GB |
| Overhead | 10 GB |

Total max simultaneous: ~140 GB if S5+S7+S1 run together. Stagger S3 (72B is biggest) — only run when others paused.

**Scheduler rule**: S3 (72B captioning) and S5 (K=8 self-play) are mutually exclusive. Everything else can co-run.

#### Option C — checkpoint-and-swap
Save model state to NVMe, swap models per task. Slower but unbounded VRAM. Use only if Option B saturates.

### 15.5 Streaming training pattern

Critical to "never wait for all data."

#### Pattern: incremental dataset with periodic refresh

```python
# inside S4 critic-train loop
while True:
    dataset = load_all_transitions("/scratch/transitions/")  # whatever exists now
    if len(dataset) < MIN_SAMPLES:
        sleep(300); continue
    train_one_epoch(model, dataset)
    save_checkpoint()
    if len(dataset) > last_seen * 1.5:
        # data grew 50% since last full retrain -> reset LR scheduler
        reset_lr_scheduler()
    last_seen = len(dataset)
```

- Start training on 500 samples within hour 1.
- By hour 8 (data fully ingested) dataset is 100k.
- Model has seen 8 epochs of growing data, last 2 on full set.
- Better than waiting 8 hr then 1 epoch.

#### Pattern: rolling buffer for self-play

S5 emits `(prompt, render, critic_score)` tuples. S6 keeps **rolling top-100 + bottom-100** across all renders, drops middle. S7 always trains on freshest top/bottom. Old renders can be deleted.

### 15.6 Backpressure + queue depth limits

Prevent disk fill:

| Queue | Max size | Action when full |
|---|---|---|
| `/scratch/raw/` | 500 GB | S0 pauses download |
| `/scratch/cache/` | 200 GB | S1 evicts stems for already-embedded clips (keep CLAP only) |
| `/scratch/renders/` | 100 GB | S5 deletes critic-bottom-50 % renders |
| `/scratch/models/` | 50 GB | S4/S7/S8 keep only last 3 checkpoints + best |

Monitor in window 9 with `df` + queue counts logged every 30 sec.

### 15.7 Restart resilience

Any stage can be killed + restarted without losing progress. Mechanisms:

- **Idempotent processing**: S1 skips clips with existing cache. S5 skips run-IDs already on disk.
- **Atomic writes**: write to `*.tmp` then rename. Partial writes ignored.
- **Resume from checkpoint**: S4/S7/S8 always load latest `.pt` if exists.
- **Manifest as source of truth**: each stage updates `manifest.json` with status `pending/running/done/failed`. Restart picks up `pending` only.

Lost work on crash: bounded by the in-flight item ≤ ~5 min.

### 15.8 GPU utilization targets

| Phase | Target avg utilization | Concurrent stages |
|---|---|---|
| Hour 0–2 (cold) | 30 % (only S0+S1 ramping) | download + analyze |
| Hour 2–8 (data filling) | 60 % | S1+S3+S4+S8 simultaneously |
| Hour 8–16 (training-heavy) | 80 % | S4+S7+S8+S5 |
| Hour 16–24 (final render) | 90 % | S9 K=64 saturates |
| Avg over 24 hr | **65 %** vs current ~10 % | |

**Effective compute multiplier**: 6.5×. $80 / 6.5 = "equivalent quality of $13 in old plan." Massive efficiency win.

### 15.9 Concrete kickoff sequence

When the MI300X comes back up, **5 minutes of setup** unlocks everything:

```bash
# tmux scaffolding
tmux new-session -d -s aijockey
for i in {0..9}; do tmux new-window -t aijockey:$i; done

# scratch dirs
mkdir -p /scratch/{raw,cache,transitions,embed,renders,models,preferences,output}
echo '{}' > /scratch/manifest.json

# launch stage workers (each in its tmux window)
tmux send-keys -t aijockey:1 'python scripts/stage0_download.py --src mixotic,fma_medium,mtg_jamendo' C-m
tmux send-keys -t aijockey:2 'python scripts/stage1_analyze.py --watch /scratch/raw --workers 4' C-m
tmux send-keys -t aijockey:3 'python scripts/stage2_segment.py --watch /scratch/cache' C-m
tmux send-keys -t aijockey:4 'python scripts/stage3_embed.py --watch /scratch/cache' C-m
tmux send-keys -t aijockey:5 'python scripts/stage4_critic.py --watch /scratch/transitions --min-samples 500' C-m
tmux send-keys -t aijockey:8 'python scripts/stage8_bridge.py --watch /scratch/transitions --min-samples 200' C-m
tmux send-keys -t aijockey:9 'python scripts/monitor.py' C-m
```

S5 self-play, S7 DPO, S9 final render fire later when their dependencies (critic checkpoint, preference data, all models) are ready — those scripts auto-start when prerequisites land.

**Foreground dev** (window 0) continues coding §3 fixes while all of the above runs.

### 15.10 Scripts to write (Block S, ~5 hr dev)

New scripts for the pipeline:

| Script | Hr | Purpose |
|---|---|---|
| `scripts/stage0_download.py` | 0.5 | aria2 wrapper, manifest of source URLs, retry, dedup |
| `scripts/stage1_analyze.py` | 0.5 | watch loop wrapping existing `analyze.py` |
| `scripts/stage2_segment.py` | 1 | novelty + onset → (pre, transition, post) on DJ sets |
| `scripts/stage3_embed.py` | 0.5 | CLAP batch + Qwen2-Audio-72B caption batch |
| `scripts/stage4_critic.py` | 0.5 | streaming retrain wrapper |
| `scripts/stage5_selfplay.py` | 0.5 | K=8 render + critic-rank |
| `scripts/stage7_dpo.py` | 0.5 | LoRA DPO retrain |
| `scripts/stage8_bridge.py` | 0.5 | MusicGen-Small fine-tune loop |
| `scripts/stage9_finalbatch.py` | 0.5 | K=64 render of curated prompts |
| `scripts/monitor.py` | 0.5 | rocm-smi + queue depth + uptime cost ticker |

**Dev cost**: 5 hr / $10. Already inside Tier 0 budget (folded into Phase A polish block).

### 15.11 Critical mindset shift

**Old plan** (sequential): "Wait for ingest → wait for analyze → train → render."

**New plan** (pipeline): every stage running, **no barrier syncs**. Quality of trained models grows monotonically as more data lands. Final render uses whatever-is-best-at-T=24h, which is N hours better than waiting for "complete" data.

Real-world example trajectory:
- T=1h: 50 clips analyzed, critic v1 still loaded, S4 starts training on 500 synthetic transitions.
- T=4h: 5k clips analyzed, 200 real transitions extracted, S4 retraining on real data, S5 self-play producing first batch.
- T=8h: 25k clips analyzed, 1k transitions, critic v2 reasonable, DPO iter-1 starts.
- T=16h: full data, critic v3 well-trained, DPO iter-1 done, MusicGen-Small bridge model ready.
- T=20h: S9 launches K=64 final render with everything tuned.
- T=24h: 20 best-of-64 mixes ready.

Total wall: 24 hr. GPU compute hours: ~20. Cost: ~$40 (well under $80 because efficiency is high).

Remaining ~$40 budget = **second iteration**: re-run DPO on better preferences from first batch's human-listen feedback, re-render top mixes. Compounds quality further.

---

## 16. Efficiency techniques — data, training, inference

This section enumerates **modern ML/AI tricks** that save data, save GPU-hours, or compound model quality. Each is gated by impact-per-hour for AiJockey context. Selected items folded into final budget.

### 16.1 Data efficiency (do more with less)

#### A. Audio data augmentation (5–10× effective dataset size, free)

Apply in dataloader, not on disk:

| Technique | What | Use for | Effective multiplier |
|---|---|---|---|
| **SpecAugment** | Random time/freq masks on mel-spectrogram | Critic v2 training | ~3× |
| **Pitch shift ±2 semitones** | rubberband / SoX | Critic, MusicGen FT | ~5× |
| **Time stretch ±10 %** | rubberband | Critic | ~3× |
| **Mixup** | Linear-interpolate two transitions, blend labels | Critic | ~2× |
| **Speed perturbation 0.9–1.1×** | resample | All audio models | ~3× |
| **Gain jitter ±6 dB** | scalar multiply | Critic | free |
| **Codec roundtrip** (mp3↔wav at random bitrate) | ffmpeg | Critic — kills codec bias from STATUS bug #3 | ~2× |

Stack these → 1 real transition becomes ~30 effective samples. **2.5k real transitions → 75k training samples.**

**Cost**: ~1 hr to wire augmentation pipeline. Save: skip ingesting more raw data.

#### B. Active learning (label only what matters)

Instead of random preference labeling, query the **most uncertain** samples.

- Critic v2 outputs probability `p(real_DJ | sample)`.
- Pick samples with `p ≈ 0.5` (highest uncertainty) for human listen.
- Boundary cases inform model 10× more than confident ones.

**Effort**: 1 hr. **Save**: 30 human-labels instead of 100 for same model gain.

#### C. Pseudo-labeling (self-training)

- Train critic v2 on small real-DJ corpus.
- Use it to label ~10k unlabeled mixes (self-play output).
- Filter to high-confidence labels only (p > 0.9 or p < 0.1).
- Retrain critic on real + pseudo-labeled.
- Iterate.

**Effort**: built into S5/S6 pipeline (§15). **Save**: 10× effective critic training data without new human labels.

#### D. Curriculum learning

Train order matters. Stage by difficulty:

1. **Stage 1**: real-DJ-vs-random-splice (obvious differences) — critic learns "what is a real transition."
2. **Stage 2**: real-DJ-vs-best-of-current-model (subtle) — critic learns "what makes ours fall short."
3. **Stage 3**: top-of-K vs middle-of-K (very subtle) — critic learns "what makes the best best."

Each stage uses lower LR. Final model sees all 3 ranges.

**Effort**: data ordering, ~30 min. **Gain**: 20–30 % faster convergence.

#### E. Multi-task auxiliary heads (free signal)

Critic v2 backbone (CLAP encoder) predicts multiple things from same audio:

- Main head: `is_real_DJ_transition`
- Aux head 1: BPM regression
- Aux head 2: key classification (24 Camelot codes)
- Aux head 3: technique classification (eq_swap / filter_fade / ...)

Loss = main + 0.3 × aux. **Auxiliary supervision is free** (BPM/key/technique already extracted by analyzer).

**Effort**: 1 hr to add aux heads. **Gain**: representations richer, main task improves.

### 16.2 Training speed (less GPU-time)

#### A. Mixed precision (bf16) — **2× speedup, free**

MI300X native bf16 path. Wrap training with `torch.amp.autocast(dtype=bfloat16)`. Already standard but verify enabled.

#### B. `torch.compile` — **30–50 % speedup, free**

PyTorch 2.x JIT. Wrap models:

```python
model = torch.compile(model, mode='reduce-overhead')
```

Works on MI300X via Inductor + ROCm backend (PyTorch 2.6+). Free win.

#### C. Flash Attention 2 — **2–4× attention speedup, lower memory**

Enable via `attn_implementation='flash_attention_2'` in HF transformers. Required for big context (per-junction Director with 16 sec audio context).

#### D. QLoRA — **4× memory reduction during fine-tune**

4-bit quantize base model, train LoRA adapters in fp16. Lets us fit Qwen2-Audio-**72B** for DPO training on single MI300X.

```python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb)
```

**Unlocks**: 72B Director DPO instead of 7B. Way better taste teacher.

#### E. ORPO instead of DPO — **half VRAM, simpler**

[ORPO](https://arxiv.org/abs/2403.07691) (2024) drops the reference model that DPO needs. Single forward pass per sample. Half the memory, ~1.5× faster training, comparable results.

**Switch**: `from trl import ORPOTrainer` instead of `DPOTrainer`. Drop-in.

#### F. KTO (Kahneman-Tversky Optimization) — **point-wise feedback, no pairs**

[KTO](https://arxiv.org/abs/2402.01306) (2024) trains on `(sample, good/bad)` flags instead of pairs. User can thumbs-up/down individual mixes — much faster than picking winner from pairs.

**Effort**: switch to `KTOTrainer`. **Save**: ~50 % human labeling time.

#### G. Lion / Sophia optimizer — **30–50 % faster than AdamW**

[Lion](https://arxiv.org/abs/2302.06675) and [Sophia](https://arxiv.org/abs/2305.14342) are newer optimizers. Both available in `torch-optimizer`. Smaller state (Lion = 1× param mem vs AdamW 2×).

**Drop-in replacement**. ~20 % wall-clock savings on training.

#### H. GaLore (gradient low-rank projection) — **65 % optimizer memory reduction**

[GaLore](https://arxiv.org/abs/2403.03507) (2024) projects gradients to low-rank subspace before optimizer step. Train Qwen-7B full fine-tune (not just LoRA) on a single MI300X.

**Use**: full fine-tune Director if LoRA results plateau.

#### I. Gradient checkpointing — **bigger batches, slower-per-step**

Trade compute for memory. Effective batch size 4× via re-computing activations. Useful when batch size is bottleneck for small models like critic v2.

#### J. 8-bit AdamW — **half optimizer memory**

`bitsandbytes.optim.AdamW8bit`. Drop-in. Combines well with QLoRA.

#### K. Knowledge distillation (teacher → student)

- **Hard distillation**: 72B labels training data, 7B trains on labels. Already covered.
- **Soft distillation**: 7B trains to match 72B's full output distribution (KL divergence). Smoother signal, faster convergence.
- **Cost**: only at training time. Inference uses 7B alone.

### 16.3 Inference acceleration

#### A. Batch processing for analysis stage

Demucs/CLAP currently per-clip. Batch 16 clips → ~10× throughput. ROCm batch primitives are stable.

**Effort**: rewrite stage S1 to process batches. ~1 hr. **Gain**: ingest 100k clips in ~3 hr instead of 30 hr.

#### B. int8 / int4 quantization for inference

`bitsandbytes.LLM.int8()` or `gptq` for Director. Same model, 1/4 VRAM, ~2× faster inference.

**Use**: per-junction Director inference path. Final-render path keeps full precision.

#### C. Consistency model distillation for diffusion bridges

[LCM / Consistency Model](https://arxiv.org/abs/2310.04378) distills 50-step diffusion → 1–4 step. **10–50× faster** Stable Audio Open inference.

**Effort**: ~2 hr to download or train consistency adapter for SAO. **Gain**: K=64 final render with bridges becomes feasible in 1 hr instead of overnight.

#### D. Continuous batching for LLM inference

vLLM or TGI batches requests dynamically. Per-junction Director makes ~15 calls/mix × 20 mixes × 64 K-renders = ~19k calls. Continuous batching → 5× throughput.

**Effort**: 2 hr to set up vLLM endpoint. **Gain**: K=64 renders complete in hours not days.

#### E. KV cache + speculative decoding

For per-junction calls with shared prefix (system prompt + clip metadata), cache the prefix once. Speculative decoding with smaller draft model gives another 2× on top.

#### F. FAISS for RAG retrieval

`faiss-gpu` or `faiss-cpu` over CLAP embeddings. 100k transitions retrieved in <50 ms. Negligible cost vs Director inference.

### 16.4 Algorithmic / architectural

#### A. Two-tower retrieval

Pool encoder + query encoder, dot-product retrieval. Decouples training from inference. Retrieval is exact + fast.

#### B. Sliding-window incremental embedding

When clip changes (re-analyze on edit), only re-embed affected segments. Saves CLAP recompute cost.

#### C. Hard negative mining

For critic training: after epoch 1, find samples critic gets wrong → upsample those in epoch 2. Up to 3× faster convergence to high accuracy.

#### D. Replay buffer

Keep "hard" examples across training runs. Like RL replay. Critic doesn't forget edge cases when new data dominates.

#### E. LoRA ensemble at inference

Train 3 LoRA adapters with different seeds. Average their outputs. ~5 % quality bump for free.

#### F. Adapter mixing — soft MoE

Per-junction, weight LoRA adapters by genre confidence. Smooth blend of "house Director" + "techno Director" instead of hard switch.

### 16.5 Concrete picks for AiJockey budget

Filter the list to high-impact-per-hour wins:

| Technique | Effort | GPU hr saved | Quality gain | **Priority** |
|---|---|---|---|---|
| bf16 + torch.compile + FA2 (16.2 A/B/C) | 1 hr | ~10 hr saved on critic+DPO | 0 | **P0 must-do** |
| QLoRA Director (16.2 D) | 1 hr | enables 72B → cheaper teacher | huge | **P0** |
| ORPO swap (16.2 E) | 0.5 hr | ~2 hr per DPO run | 0 | **P0** |
| Audio augmentation pipeline (16.1 A) | 1 hr | data acquisition cut 30 % | medium | **P0** |
| Multi-task aux heads on critic (16.1 E) | 1 hr | 0 | medium | **P0** |
| Active-learning labeling (16.1 B) | 1 hr | 0 | high (data quality) | **P0** |
| Batch-mode analyze stage (16.3 A) | 1 hr | ~20 hr saved on 100k ingest | 0 | **P0** |
| Curriculum DPO data ordering (16.1 D) | 0.5 hr | 0 | medium | **P0** |
| Consistency distillation for SAO (16.3 C) | 2 hr | ~5 hr on K=64 final | 0 | **P1** |
| vLLM continuous batching (16.3 D) | 2 hr | ~10 hr on per-junction renders | 0 | **P1** |
| Hard negative mining (16.4 C) | 0.5 hr | 0 | small | **P1** |
| KTO instead of DPO (16.2 F) | 0.5 hr | 0 (different labeling cost) | medium | **P1 if labeling becomes bottleneck** |
| Pseudo-labeling loop (16.1 C) | 1 hr (logic) | 0 (compute already in S5) | medium | **P1** |
| Lion / Sophia optimizer (16.2 G) | 0.25 hr | ~3 hr training | 0 | **P1 cheap** |
| GaLore full fine-tune (16.2 H) | 2 hr | enables full FT | maybe | **P2 only if LoRA stalls** |
| LoRA ensemble (16.4 E) | 1 hr | costs 3× train | small | **P2** |
| Per-genre adapter mixing (16.4 F) | 4 hr | 0 | medium | **P2** |
| Knowledge distill 72B → 7B soft labels (16.2 K) | 4 hr | 0 (extra train) | medium | **P2** |
| Speculative decoding (16.3 E) | 2 hr | ~5 hr | 0 | **P2** |

**P0 set** (10 items × ~7 hr total dev): saves ~30+ GPU-hours, doubles effective data. **Net: $14 dev time recovers $60 of GPU time + huge data leverage.**

**P1 set** (5 items × ~7 hr dev): another ~20 GPU-hours saved. Quality gains modest but real.

**P2 set**: only if Tier 0/1/2 finished early.

### 16.6 Re-budgeted total with efficiency tricks

Original plan (§14.6): $80 / 60 wall-hr.

With **P0 efficiency** applied:
- Analysis stage 10× faster (batch + compile + bf16) → S1 budget 10 hr → 1 hr.
- Critic/DPO 2× faster (bf16 + compile + ORPO) → S4/S7 budgets 8 hr → 4 hr.
- Bridge fine-tune 30 % faster (bf16 + compile + Lion) → S8 budget 6 hr → 4 hr.
- 72B teacher unlocked via QLoRA → can run captioner on S3 in 50 % time.
- Augmentation 5× effective data → can downsize MTG-Jamendo ingest by half.

**New compute budget**: ~30 GPU-hr ≈ **$60** for full pipeline.

**Saved**: ~$20 → reinvest in **DPO iter-2** (was deferred). Or extend final batch to 30 mixes K=64 instead of 20 mixes.

### 16.7 What's NOT worth it (ruled out)

| Technique | Why not for AiJockey |
|---|---|
| Pure end-to-end audio LM (Suno-style) | Hallucinates non-pool audio. Defeats purpose. |
| Reinforcement learning with PPO | Critic-as-reward is unstable; DPO/ORPO is direct. |
| MAML / meta-learning | Too few task instances; standard fine-tune fits. |
| Bayesian active learning | Marginal gain over uncertainty sampling, way more code. |
| Custom DAC tokenizer | EnCodec/DAC pretrained suffices for short bridges. |
| Multi-GPU distributed training | We have 1 MI300X. Skip. |
| TensorRT / ONNX export | ROCm path is fine; no inference SLA. |
| Quantization-aware training | Post-training quant is enough for our quality bar. |
| Diffusion noise schedule tuning | SAO defaults are good. |
| Architecture search | Way out of budget scope. |

### 16.8 Updated execution mode

§15 pipeline stays. But each stage gets P0 efficiency upgrades:

- **S1 analyze**: `BatchedAnalyzer` class, `torch.compile`, bf16, 4-clip batches.
- **S3 embed**: Qwen2-Audio-72B via QLoRA-load + Flash Attention 2 + int8 inference.
- **S4 critic**: ORPO loss head as auxiliary, multi-task heads (BPM/key/tech), augmentation pipeline, Lion optimizer.
- **S5 self-play**: vLLM Director endpoint for continuous batching.
- **S7 DPO**: ORPO + QLoRA + Lion + Flash Attn 2 + active-learning labeled data.
- **S8 bridge**: torch.compile MusicGen-Small + Lion + augmented triplets.
- **S9 final render**: vLLM Director + consistency-distilled SAO + int8 quantized.

All P0 techniques already drop into existing stages without architectural change.

### 16.9 Bottom line for §16

**Efficiency wins compound.** P0 set alone:
- ~30 GPU-hr saved → $60 saved → recoverable for second DPO iteration.
- 5–10× data multiplication via augmentation.
- 72B Director becomes affordable via QLoRA.
- ORPO/KTO simplify training loop, halve labeling burden.

**Without these tricks, $80 budget is tight.** With them, $80 covers full pipeline + iter-2.

---

## 10. Bottom line

**Two-track plan running in parallel:**

### Foundation: Phase A polish (§1–§9c)
`electronic + 4-on-floor + 110–130 BPM + 3–10 instrumental-leaning clips` → **3-tier vocab** (`minor/major/drop`) → **6 techniques** → **beatmix form** → **3 arcs** → **phrase-quantized junctions, stem-swap overlaps, humanized accents, whitelisted sample lib**.

These fixes close the mechanical-feel gap and remove failure classes. ~8.5 hr foundation work.

### GPU saturation expansion (§13)
Cold-start no longer matters — demo is **pre-rendered MP3s**, not live service. Spend the unused 30+ hr of MI300X capacity on quality:

- **Track 1**: DPO LoRA Director on user preference labels.
- **Track 2**: Mix critic v2 — CLAP features, 30-sec windows, 100k samples.
- **Track 3**: K=32 parallel renders per mix, critic-ranked.
- **Track 4**: Multi-model generative bridges (Stable Audio Open + MusicGen-Large + AudioLDM2) for incompat-key gaps.
- **Track 5**: Per-junction Director calls with full audio context (matches ARCHITECTURE.md streaming pattern).
- **Track 6**: Deep ensemble analysis (2× stem models, 2× beat trackers, 2× key detectors, Camelot wire-up).
- **Track 7**: Style-RAG activation — 25-set DJ corpus embedded, retrieved as Director's in-context examples.

### Total budget
~45 hr ≈ **$83** of remaining $80 credits (1 DPO iteration, drop AudioLDM2 if tight). GPU sustained at 60–80 % utilization vs current 10 %.

### Output deliverable
**10–20 best-effort offline mixes**, each best-of-32 from a DPO-tuned Director with bridge-augmented transitions and ensemble-analyzed clip metadata. Hosted as static MP3s. HF Space optional.

### New pitch story
> "Each track in this set was selected from 32 critic-ranked candidates rendered by a DPO-tuned Director that hears every clip, retrieves real DJ transitions as in-context examples, and uses a 3-model generative ensemble to bridge incompatible-key gaps. Phrase-quantized junctions, stem-aware role swaps, and humanized accents give it a producer's pocket. AGPL-3.0, MI300X-trained on 25 real DJ sets."

### Sequencing
Foreground: dev work on §3 fixes + per-junction Director + final batch render.
Background (parallel GPU): data ingest, DPO train, critic retrain, MusicGen-Small fine-tune, deep analysis re-run, auto-captioning — all firing while foreground codes.

**Aim**: best output, not best demo speed. GPU saturated, not idle. Phase 2 backlog (§9) becomes Phase B+ (training-heavy) per [ARCHITECTURE.md](../ARCHITECTURE.md) roadmap.

### Final budget plan (§14.6 picks)

| Tier | Items | Cost |
|---|---|---|
| 0 — foundation | Phase A polish + sample gating + data ingest (FMA + Mixotic + MTG-Jamendo subset) | $26 |
| 1 — train core | Critic v2 + self-play data gen + DPO LoRA iter-1 | $20 |
| 2 — quality multipliers | Style-RAG + auto-caption + MusicGen-Small bridge FT + constitutional rules | $24 |
| 3 — final batch | K=64 best-of × 20 mixes overnight | $10 |
| **TOTAL** | | **$80** |

Hits budget exactly. Phase B+ items deferred: per-genre MoE LoRAs, DPO iter-2, AudioLDM2, full-72B auto-caption.

### Architectural progress

Per [ARCHITECTURE.md](../ARCHITECTURE.md) tier roadmap, this plan advances:
- Tier 2 (real dataset infra) → **completed** via critic v2 + self-play
- Tier 3 (generative bridge) → **partial completion** via MusicGen-Small fine-tune
- Tier 5 (RLHF) → **partial** via DPO LoRA

Net: **+2.5 architectural tiers** in $80. Tier 4 (custom DJ-LLM core) remains out of scope.

### Final pitch story
> "Hybrid agent + multi-model audio system. Tuned via DPO on self-play preference data ranked by a CLAP-feature critic trained on 700 real CC DJ sets. Director retrieves real DJ transitions as in-context examples. Hard musical rules enforced as constitutional validators. Generative bridges via fine-tuned MusicGen-Small fill incompat-key/BPM gaps. Each released mix is best-of-64. AGPL-3.0, $80 of MI300X compute, no proprietary data, no YouTube — only Free Music Archive + Mixotic CC corpus + MTG-Jamendo."

### Execution mode (per §15)

**Pipeline-parallel, never blocking on data.** All downloads land directly on MI300X scratch (no laptop disk). 9 tmux windows, each a stage in the producer-consumer chain. GPU utilization 60–80 % sustained vs current 10 %.

Effective compute multiplier ≈ **6.5×** vs sequential plan. Same budget, far more output. After 24 hr wall-clock: 20 best-of-64 mixes ready, with DPO-tuned Director, fine-tuned bridge model, and CLAP-feature critic — all trained on data that arrived in parallel with the training itself.

Remaining ~$40 of $80 budget → **second DPO iteration** on first-batch human-listened preferences. Compounding quality.
