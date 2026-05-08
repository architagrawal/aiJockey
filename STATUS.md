# AiJockey — Status, Problems, Next Steps

Snapshot of what's built, what works, what's broken, what's next.
Last updated: end of hackathon prep day.

---

## What's done

### Core pipeline (Phase A — works)

- **Analyze** (`src/analyze.py`): Demucs htdemucs stems + madmom beats + librosa key + librosa structure segmentation + CLAP embedding via transformers (laion-clap fallback). Per-clip JSON + NPZ cache. Stem-aware section labeling using drums+bass RMS.
- **Planner** (`src/planner.py`): beam search over clip pool. Picks subset, free order, per-clip section, per-transition technique. Configurable arc shape (build/peak/rollercoaster/descend/flat_high/flat_low/custom). Min 30s segment, cooldown 5, min 5 unique clips.
- **Execute** (`src/execute.py`): renders timeline. Time-stretch + pitch-shift via rubberband. 15 transition techniques implemented.
- **Master** (`src/master.py`): HP30 → multiband compression → glue compression → LUFS norm → true-peak limiter.
- **Eval** (`src/eval.py`): beat continuity, beat-alignment-error ms, energy-arc correlation, LUFS, true peak, FAD vs reference.

### Transitions library (`src/transitions.py`)

15 techniques:
cut · crossfade · eq_swap · filter_fade · silence_drop · drum_break · mashup · stem_swap · echo_out · spinback · pitch_bend · loop_tighten · scratch_fill · loop_callback · riser_bridge

Plus procedural FX synthesis (`src/synth_fx.py`): impacts, sub_drops, risers, sweeps, snare_rolls, hihat_rolls, airhorns, vinyl. Auto-fallback if no real samples.

### ML control axes (NEW)

- **Text-prompt control** (`--prompt "..."`): natural-language input → CLAP text embedding → biases clip selection in CLAP space. Same pool produces different mixes.
- **Arc shape control** (`--arc <preset>`): planner's strategic intent. Controls opener energy + flow shape. 6 presets.

### Trained model scaffolds (NOT yet trained on data)

- `src/training/synthetic_dataset.py` — synthetic transition labels via smoothness scoring
- `src/training/classifier.py` — Tier 1 MLP technique classifier
- `src/training/clap_pairs.py` — heuristic triplet generator
- `src/training/clap_finetune.py` — Tier 1.5 CLAP compat head (InfoNCE)
- `src/training/dataset_builder.py` — Tier 3 real DJ mix dataset builder (yt-dlp + manual tracklists)
- `src/training/finetune_musicgen.py` — Tier 3 MusicGen fine-tune skeleton (incomplete)
- `src/training/genre_classify.py` — zero-shot genre classification via CLAP

Notebooks:
- `scripts/aijockey_env_check.ipynb` — env validation
- `scripts/aijockey_colab.ipynb` — full pipeline on Colab T4
- `scripts/train_classifier.ipynb` — Tier 1 training
- `scripts/train_clap_compat.ipynb` — Tier 1.5 training

### Demo infrastructure

- `app.py` — Gradio web UI (upload clips, configure, generate, audition)
- `HACKATHON.md` — MI300X hosting walkthrough
- `restricted_mode.py` — demo-safe technique whitelist

### Data downloaded

- 80 clips from yt-dlp (NCS, Argofox, Liquicity, Monstercat, plus iconic tracks: Daft Punk, Avicii, Skrillex, Calvin Harris, Martin Garrix, ODESZA, Pendulum, Kanye, Travis Scott, etc.)
- Curated 15-clip subset in `clips_demo/` for hackathon demo

### License + repo

- AGPL-3.0
- Pushed to https://github.com/architagrawal/aiJockey
- gitignore covers clips/, cache/, output/, downloaded media

---

## What works end-to-end (verified on Colab T4)

1. ✅ Analyze 13 of 15 demo clips
2. ✅ Plan with text prompt produces different timelines per prompt
3. ✅ Plan with arc shape produces different opener + flow per shape
4. ✅ Execute renders all 15 transition techniques (rubberband stretch, mixing)
5. ✅ Master chain produces club-LUFS audio
6. ✅ Eval prints metrics

---

## Known problems

### Audio quality
- 30-min mix v2 generated noise/repetitive output (before fixes). After fixes (min 30s segment, cooldown=5, arc shape control), 5-min mix sounds coherent.
- `pitch_bend` + `scratch_fill` + `spinback` produce artifacts on some clips. Currently filtered via `restricted_mode`.
- Time-stretch beyond ±5% audibly degrades. Planner respects this in compatibility filter.

### Pipeline issues
- `play_at` in timeline = pre-stretch sum, not actual mix sample position. Eval's beat_continuity / BAE compute against wrong reference times → metrics misleading. Subjective listen is the truth.
- madmom beat detection deprecation warning (`method` param). Cosmetic.
- deadmau5 Strobe (10 min) too long for fast Demucs analyze; manually skipped.
- ODESZA Severance soundtrack remixes ended up in pool from channel scraping; replaced via curated top-tracks search.

### ML/model issues
- **Genre classifier** confidence 0.3-0.6 — most clips misclassified (Bonobo→techno, Skrillex→techno, Kanye→house). CLAP prompt phrasing weak. Cosmetic feature, not used by planner.
- **Tier 1 classifier** trained on synthetic smoothness labels — weak signal, not validated to be better than rule tree.
- **Tier 1.5 CLAP compat head** — same issue, heuristic triplets.
- **Tier 3 MusicGen fine-tune** — code skeleton only. Needs audiocraft ConditionProvider integration + real DJ mix transition dataset.
- **No actual generative bridge** in pipeline yet. Phase B work.

### Infrastructure issues
- Colab T4 free tier hit GPU quota mid-day. Switched to CPU runtime for execute/master.
- MI300X access not yet granted as of cut-off.
- yt-dlp downloads = personal/research only. Pool contains copyrighted material.

### Data issues
- Pool of 13-15 clips too small for diverse mixes. Many clips share energy profile (all EDM at ~120-130 BPM).
- No labeled DJ-mix transition data collected yet (Tier 2). Bottleneck for real training.

---

## Demoable claims (honest)

- ✅ Open-source AGPL pipeline integrating SOTA models (Demucs, madmom, CLAP, librosa)
- ✅ 15 transition techniques in single OSS codebase (most auto-DJs do 1-3)
- ✅ Stem-aware mixing (drum_break, mashup, stem_swap)
- ✅ Phrase-aware planner (16/32-bar boundaries enforced)
- ✅ Text-prompt clip selection via CLAP cross-modal alignment
- ✅ Arc-shape strategic control (planner picks opener + flow per intent)
- ✅ Procedural FX synthesis fallback (impacts/risers/vinyl FX, no licensing)
- ✅ Club-LUFS mastering chain

## NOT claims (would be lying)

- ❌ "Multi-agent system" — sequential pipeline, no agent comms
- ❌ "Streaming AI DJ" — batch render, not real-time
- ❌ "Custom DJ LLM" — uses HF pretrained models, no custom AR audio model
- ❌ "Trained on DJ mixes" — only synthetic + heuristic labels so far
- ❌ "Better audio quality than Suno" — Suno >> us on audio fidelity, we win only on control + open-source

---

## Next steps (priority order)

### Today / hackathon prep

1. **Save Drive backup** — cache.zip + output/ to Google Drive (5 min)
2. **Render 4 arc mixes on CPU runtime** — pre-baked demo files (10 min)
3. **Download backup mixes to laptop** — safety net (2 min)
4. **Build slides** — use HACKATHON.md outline (2 hr)
5. **Pitch dry-run** — practice 5-min demo (1 hr)
6. **Provision MI300X if access granted** — deploy Gradio app

### Post-hackathon (week 1-2)

7. **Annotate 25 DJ mix tracklists** — manual JSON entry from YouTube/1001tracklists. ~30 min/mix. Unblocks Tier 2/3 training data.
8. **Run Tier 1 classifier training** on Colab T4 — validate it beats rule tree A/B.
9. **Run Tier 1.5 CLAP compat training** — same.
10. **Improve genre_classify prompts** — current ones too generic, needs domain wording.

### Mid-term (month 1)

11. **Finish `finetune_musicgen.py`** — wire DJContextEncoder into audiocraft ConditionProvider, real LM loss against EnCodec tokens. ~1 week dev.
12. **MI300X session** — fine-tune MusicGen-medium on transition dataset. ~$200-400 credits, 3-5 days compute.
13. **Add `generate_bridge` technique** — invoke fine-tuned MusicGen in execute.apply_transition for hard transitions.
14. **Fix `play_at` calculation** — track actual sample positions in execute, write back to timeline. Real BAE eval.

### Long-term (3-6 months)

15. **Phase C agent layer** — refactor into Director / Selector / Mixer / Critic classes. Online critic + revision (the "Option B" we deferred).
16. **Phase D custom DJ LLM** — pretrain audio token AR transformer on real DJ mix corpus. MI300X 2-4 weeks. ~$1500-3000 credits.
17. **Phase E RLHF** — pairwise preference learning, mix-A vs mix-B. Tune planner weights via learned reward.
18. **Streaming runtime** — real-time mix generation with 4-sec chunks + critic gating.

### Open questions / deferred

- License: MusicGen weights CC-BY-NC means generative output can't be commercial. For commercial path, must train Stable Audio Open or own model from scratch.
- Real DJ mix data legality: research/personal only. For redistribution, need licensed dataset or original recordings.
- Evaluation metric reliability: subjective listen is gold standard for now. FAD/BAE imperfect.

---

## Decision log

- **Architecture**: hybrid agent + ML over rule-based DSP. Right tool per layer.
- **Base models**: Demucs (stems), madmom (beats), CLAP via HF transformers (embeddings).
- **License**: AGPL-3.0 — protects forks from commercial appropriation.
- **Restricted scope**: 3-genre EDM/house/techno-leaning at 115-135 BPM for demo stability. Larger scope = larger data + tuning.
- **Skipped Tier 2 RLHF**: too research-heavy for hackathon. Replaced with simpler Tier 1 + 1.5 + arc shape control.
- **Generated synthetic data first** vs collecting real labels. Pragmatic — real labels are bottleneck.
- **No multi-agent buzzwords without functionality**: kept pipeline sequential. Refactor when streaming needed.

---

## Files of note

| File | Purpose |
|------|---------|
| `README.md` | onboarding |
| `AiJockey.md` | full architecture + code reference |
| `ARCHITECTURE.md` | streaming agent + LLM hybrid target design |
| `HACKATHON.md` | MI300X hosting + pitch outline |
| `STATUS.md` | this file |
| `LICENSE` | AGPL-3.0 |
| `app.py` | Gradio demo UI |
| `src/main.py` | CLI orchestrator |

---

## Bottom line

Pipeline works. 4 distinct mixes generated from same 13-clip pool via prompt + arc control. AGPL + open source = real contribution. Real ML novelty (custom DJ LLM, online critic, RLHF) deferred to post-hackathon — needs MI300X + months. For hackathon: pitch as best-in-class open-source DJ pipeline with 2-axis natural control. Honest. Demoable. Defensible.
