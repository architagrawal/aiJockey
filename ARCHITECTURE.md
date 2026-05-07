# AiJockey — Streaming Agent Architecture (Phase B+)

Real architecture for owning a streaming DJ AI system. Hybrid: agent layer
around a learned audio LM core. AGPL-3.0 keeps forks open.

This document = target. Current code (Phase A) is groundwork. Phase B/C work
toward this.

---

## The pattern: streaming next-audio prediction

Just like an LLM predicts next text token, our system predicts **next audio
chunk**, conditioned on:
- Last K seconds of mix output
- Pool of available clips + samples (with metadata)
- High-level goal (energy arc, duration remaining, key/BPM constraints)
- Past control choices (which clip used, last technique applied)

Output:
- Next 2-4 seconds of audio
- Control tokens (which pool item to use next, what technique)

Streaming = chunk-by-chunk, can play in real-time with lookahead buffer.

---

## Agent decomposition

```
┌─────────────────────────────────────────────────────────┐
│                   DIRECTOR AGENT                         │
│  - holds long-term plan (target duration, energy arc,    │
│    surprise budget, key/genre constraints)               │
│  - re-evaluates every N seconds based on actual output   │
│  - emits goal vectors: "next 8 bars: target_energy=X,    │
│    key=Y, prefer technique=Z..."                         │
└────────────────────┬─────────────────────────────────────┘
                     │ goal vector (high-level constraints)
                     ▼
┌─────────────────────────────────────────────────────────┐
│              DJ LLM — CORE MODEL                         │
│  Audio-token autoregressive transformer.                 │
│  Tokenizer: DAC or EnCodec (audio → discrete tokens).    │
│  Cross-attends to:                                       │
│    * Pool index: CLAP+metadata of all clips/samples      │
│    * Sample bank index: synth + real one-shots           │
│    * Director goal vector                                │
│    * Past N seconds of mix audio (rolling context)       │
│  Outputs:                                                │
│    * Next audio tokens (2-4 sec chunk)                   │
│    * Control tokens — "use clip 7 segment 3, eq_swap"    │
└──────────┬───────────────────────────┬───────────────────┘
   audio   │                  control  │
           ▼                            ▼
   ┌──────────────┐           ┌──────────────────┐
   │ DECODER      │           │  SELECTOR AGENT  │
   │ DAC/EnCodec  │           │  - parses control│
   │ tokens →     │           │  - picks best    │
   │ waveform     │           │    clip+segment  │
   └──────┬───────┘           │    from pool     │
          ▼                   │  - rule-based    │
     stream out               │    matching      │
                              └────────┬─────────┘
                                       │ selected clip+stems+technique
                                       ▼
                              ┌──────────────────┐
                              │  MIXER AGENT     │
                              │  - stretch/pitch │
                              │  - EQ-swap       │
                              │  - apply 15      │
                              │    techniques    │
                              │  - sample bank   │
                              └────────┬─────────┘
                                       │ audio chunk fed
                                       ▼ back to LLM context
                                 (next iteration)

┌─────────────────────────────────────────────────────────┐
│                  CRITIC AGENT                            │
│  Continuously scores recent output:                      │
│    - beat continuity                                     │
│    - key clash detection (chroma vs target)              │
│    - LUFS in target range                                │
│    - FAD vs reference distribution                       │
│  If score drops: signal Director → revise plan or rewind │
└─────────────────────────────────────────────────────────┘
```

---

## Component map: now vs target

| Component | Phase A (now, rule-based) | Phase B target (learned/agentic) |
|-----------|---------------------------|----------------------------------|
| **Director** | `planner.py` beam search | RL agent on top of beam, learns plan revision |
| **DJ LLM core** | ❌ none — direct execute | Custom AR transformer over DAC tokens, fine-tuned MusicGen as starting point |
| **Decoder** | rubberband + numpy mix | DAC/EnCodec decoder |
| **Selector** | `pick_segment` heuristic | Learned pool-retrieval (Tier 1 classifier expanded) |
| **Mixer** | `transitions.py` 15 techniques | Same techniques but LLM picks WHICH at each step |
| **Critic** | `eval.py` post-hoc only | Online streaming critic — gates output |
| **Pool index** | `cache/*.json + CLAP npz` | Same, plus learned pool-attention layer |
| **Sample bank** | `samples.py` + `synth_fx.py` | Same — LLM emits "play sample X" control tokens |

Phase A code is the **scaffolding**. Phase B replaces the rule-based parts with
a trained core, keeps the agent layer and DSP infra.

---

## Why hybrid (not pure end-to-end LLM)

Pure end-to-end audio LM that generates 30 minutes:
- ❌ Hallucinates audio that doesn't match input clips → useless for DJ task
- ❌ Slow inference (autoregressive over 50 Hz tokens × 1800 sec × ...)
- ❌ Hard to constrain to specific clips you own
- ❌ Quality ceiling = base model, can't exceed input fidelity

Hybrid (agent + LLM):
- ✅ LLM only decides next CHOICE; clips/samples are real audio from your pool
- ✅ Fast — LLM generates short transition tokens, decoder runs at audio rate only on bridges
- ✅ Constrained to your pool — model can't hallucinate non-existent tracks
- ✅ Audio quality bounded by inputs, not by LLM quality

Same trade-off other systems made:
- **Mubert** — agent picks from sample library + light generation
- **Endel** — agent + procedural synthesis + curated samples
- **Boomy/AIVA** — symbolic AI + audio synthesis
- **Suno** — pure end-to-end (3-min songs, chooses simplicity over constraint)

For 30-min DJ sets with YOUR clips, hybrid wins.

---

## What's "trained" vs "rule-based" in target arch

### Trained components

1. **DJ LLM core** — predicts next control + audio tokens
   - Pre-train: autoregressive next-token loss on real DJ mix audio
   - Fine-tune: retrieval-augmented — given pool, predict which item to use next
   - RL fine-tune: Critic score as reward

2. **Selector retrieval embeddings** — learned pool ranking
   - Currently: rule-based pick_segment + Tier 1 classifier
   - Target: cross-attention from LLM hidden state to pool item embeddings

3. **Critic scorer** — learned mix quality model
   - Tier 2 work: pairwise preference learning, mix-A vs mix-B
   - Used as RL reward signal during LLM fine-tune

### Rule-based (kept)

- Audio decoder (DAC/EnCodec — pretrained, frozen)
- Stem separation (Demucs — frozen)
- Time-stretch + pitch shift (rubberband — frozen)
- 15 transition technique implementations (DSP code)
- Sample bank + synth FX

Rationale: trainable where novelty / quality matters; rule-based where
existing tools are already optimal and stable.

---

## Streaming protocol

```
DIRECTOR.init(pool, target_duration=1800, energy_arc=[...])
buffer = []
clock = 0

while clock < target_duration:
    goal = DIRECTOR.next_goal(clock, buffer[-K:])    # 8-bar lookahead
    audio_tokens, control = LLM.predict_chunk(
        context=buffer[-context_len:],
        pool=POOL_INDEX,
        goal=goal,
    )

    # Materialize audio
    if control.type == 'play_clip':
        clip = SELECTOR.resolve(control.clip_query, pool)
        audio = MIXER.apply_technique(
            prev_audio=buffer[-K:],
            new_clip=clip,
            technique=control.technique,
        )
    elif control.type == 'play_sample':
        audio = SAMPLE_BANK.get_fx(control.sample_type, ...)
    elif control.type == 'generate_bridge':
        audio = LLM.decode(audio_tokens)  # only when LLM generates fresh

    # Critic gate
    score = CRITIC.score(buffer[-K:], audio)
    if score < threshold:
        DIRECTOR.revise()
        continue  # retry with new goal

    buffer.extend(audio)
    yield audio  # streaming output
    clock += len(audio) / SR
```

Real-time-capable if chunk_size > LLM inference time. With 4-second chunks +
50ms LLM step, lots of headroom.

---

## Training roadmap (concrete, tied to current code)

### Tier 1 — DONE (scaffolded) — Technique classifier
- `src/training/classifier.py`
- Trains MLP on synthetic + real transitions
- Replaces rule tree in `transition_score`

### Tier 2 — IN PROGRESS — Real dataset infra
- `src/training/scrape/*` + `dataset_builder.py`
- Builds (pre, transition, post) triplets from real DJ mixes
- Auto-labels technique heuristically

### Tier 3 — Next — Generative bridge model
- `src/training/finetune_musicgen.py` (skeleton)
- Fine-tune MusicGen on transition_audio targets
- Gives `generate_bridge` technique to MIXER

### Tier 4 — Future — DJ LLM core
- Custom autoregressive transformer over DAC tokens
- Pre-train on aggregated DJ mix corpus
- Tokenize all real DJ mixes + add control tokens for clip-selection events
- Train: next-token loss with retrieval over pool index

### Tier 5 — Far future — Critic + RLHF loop
- Pairwise preference dataset (your-mix vs Spotify-auto-DJ-mix)
- Train preference model
- RL fine-tune LLM with preference reward

### Tier 6 — Streaming runtime
- Real-time inference loop (4-sec chunks)
- WebSocket API for streaming clients
- Browser DJ booth UI

---

## Compute estimates (MI300X)

| Tier | Train time | Credit cost | Output |
|------|------------|-------------|--------|
| 1 — classifier | hours on T4 (free) | $0 | small MLP |
| 2 — dataset | CPU-bound | $0 | ~1000 transitions |
| 3 — MusicGen FT | 3-5 days | $200-400 | bridge generator |
| 4 — DJ LLM | 2-4 weeks | $1500-3000 | full streaming core |
| 5 — RLHF | 1 week | $500-1000 | preference-tuned model |

Total to full system: ~$2500-4500 of MI300X credits.

For research / personal: Tiers 1-3 cover ~80% of useful capability at ~$400.

---

## Existing similar products (study, don't copy)

| Product | What | Architecture | Open? |
|---------|------|--------------|-------|
| **Mubert** | endless AI music | rule + samples + ML | ❌ closed |
| **Endel** | adaptive ambient | procedural synth + agent | ❌ closed |
| **Suno V4** | text-to-song | pure AR audio LM, ~5min | ❌ closed |
| **Udio** | text-to-song | similar to Suno | ❌ closed |
| **AIVA** | composition | symbolic AI + sampler | ❌ closed |
| **Boomy** | song gen w/ user steering | hybrid | ❌ closed |
| **MusicGen** | text-to-music | AR over EnCodec, 30s clips | ✅ MIT code, NC weights |
| **Stable Audio Open** | text-to-audio | diffusion, 47s clips | ✅ Stability Community |
| **MAGNeT** | text-to-audio | masked LM | ✅ MIT |
| **Riffusion** | spectrogram diffusion | image-based | ✅ MIT |
| **AiJockey (you)** | DJ set gen with YOUR pool | hybrid agent + LLM | ✅ AGPL-3.0 |

Your differentiator: **DJ-specific + uses YOUR pool + open-source AGPL + streaming + 30min+ output**.

No one else hits all five.

---

## Phase ordering (do in this order)

1. **Now** (Phase A): Tier 1 + Tier 2 dataset infra. Use existing rule-based pipeline. Iterate.
2. **Next** (Phase B): Tier 3 MusicGen fine-tune. Adds `generate_bridge` technique.
3. **Then** (Phase C): Build agent layer (Director, Critic, Selector classes wrapping current code).
4. **Then** (Phase D): Tier 4 custom DJ LLM. The big one.
5. **Then** (Phase E): Tier 5 RLHF + streaming runtime.

Each phase ships standalone. Phase A is already useful (auto-DJ tool). Phase D
is the "Suno for DJ sets" novel contribution.

---

## Code organization for the agentic future

```
src/
├── agents/
│   ├── director.py          # high-level plan + revision
│   ├── selector.py          # pool retrieval / clip ranking
│   ├── mixer.py              # wraps execute.py + transitions.py
│   ├── critic.py             # online quality scoring
│   └── orchestrator.py       # streaming loop, ties agents together
├── llm/
│   ├── tokenizer.py          # DAC or EnCodec wrapper
│   ├── core.py               # AR transformer, custom or MusicGen-derived
│   ├── pool_attention.py     # cross-attn to pool index
│   └── decode.py             # streaming decoder
├── training/                 # already exists
└── (existing modules)        # analyze, transitions, master, etc.
```

Migration: incrementally move existing code into agent classes. Director ←
planner. Mixer ← execute + transitions. Selector ← pick_segment + classifier.
Critic ← eval.

---

## Bottom line

Yes — this is the right architecture. Streaming hybrid agent + LLM. No one
does it for DJ sets specifically. Your AGPL stance protects the open-source
contribution.

Phase A (current) = foundation. Phase B-D (months of work) = the actual
"DJ LLM" novel system. Total realistic timeline: 6-12 months part-time, with
~$3-5k MI300X credits for full system.

Starting points:
- **Today**: annotate tracklists for Tier 2 dataset
- **This week**: train Tier 1 classifier on Colab
- **This month**: scaffold agent classes (move existing code into agent shapes)
- **Next 1-3 months**: Tier 3 MusicGen fine-tune
- **3-12 months**: Tier 4 DJ LLM, agent runtime, streaming
