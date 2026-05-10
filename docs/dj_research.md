# DJ / Music Research for AiJockey

Reference doc grounding the Director model in real DJ practice. Three parts:
1. How songs are built (so we know what to mix *into* what).
2. How DJs actually use sound (EQ, frequency, stems, phrasing).
3. Mix-type taxonomy mapped to AiJockey's 5-tier vocab (`minor / major / drop / cut / loop`).

---

## 1. Song Structure (Electronic / Dance)

A track is not a flat loop. It has named sections with predictable energy and texture. Mixing well means knowing which section you are leaving and which you are entering.

### Canonical sections

| Section | Role | Texture | Mix in? | Mix out? |
|---|---|---|---|---|
| **Intro** | DJ tool. Stripped beat, no melody, often 16–32 bars of just kick + hat. | Sparse | Bad target (boring) | Excellent — purpose-built for it |
| **Verse / groove** | Main rhythmic body. Bass + drums + light melody. | Medium density | OK | OK |
| **Breakdown** | Energy valley. Drums removed or thinned, atmosphere/pads/vocal exposed. | Sparse, melodic | Good — frees low end of incoming track | Bad — removing energy further sounds dead |
| **Build-up** | Tension ramp. Risers, snare rolls, filter sweeps, kick removed last few bars. | Rising | Risky — incoming kick may clash with riser | Excellent — energy goes *somewhere* |
| **Drop** | Peak. Kick + bass + lead at full intensity. The hook. | Dense | Bad alone (clash) — only via *double drop* or *cut* | Bad — energy cliff |
| **Outro** | Mirror of intro. Drums fade out. | Sparse | OK | Excellent |

### Phrase grid (the invisible scaffold)

- **1 bar = 4 beats**
- **1 phrase = 8 bars = 32 beats** (standard in house/techno/EDM/pop)
- Sections almost always change on phrase boundaries: 16, 32, 64 bars
- A "drop" lands on the **1** of a new phrase
- DJs mix on the 1 — start the incoming track so its phrase aligns with the outgoing phrase. Off-phrase mixing = "noise" even if BPM-matched

> **Implication for AiJockey noise problem**: if junctions are placed mid-phrase, the mix sounds chaotic regardless of DSP polish. Cuepoints must snap to phrase grid.

### Genre variants

- **House / Tech-house**: 32-bar intro/outro, 16-bar breakdown, single drop. Forgiving. Long blends (32–64 bars) are normal.
- **Techno**: groove-driven, no big drops. Energy via *layering* and *swap*, not buildup. Loop-heavy.
- **Drum & Bass**: 32-bar intro, breakdown at 32, drop at 64. Drops are *the* moment. Double-drops are signature.
- **Trance / Big-room EDM**: long build (16–32 bars of riser+snare-roll), single explosive drop. Cuts work because builds prepare the ear.
- **Hip-hop / Pop**: verse-chorus-verse — no DJ-friendly intro. Use cuts, echo-outs, or stem-isolated acapellas.

---

## 2. How DJs Use Sound

### Frequency / EQ

A 3-band EQ is the most-used tool, more than crossfader. Two kicks playing simultaneously = mud. Two basslines = clash. Two leads = harmonic dissonance. Solution: at any moment, only **one track owns each frequency band**.

| Band | Range | Owner during transition |
|---|---|---|
| **Low** (kick + sub-bass) | <250 Hz | Always exactly **one track**. Swap on the 1 of a new phrase. |
| **Mid** (vocals, leads, snare body) | 250 Hz – 4 kHz | Often blended briefly; cut one when vocals collide |
| **High** (hats, cymbals, air) | >4 kHz | Most forgiving — can sum freely |

The classic "bass swap" technique: outgoing low cut to -∞, incoming low at 0, on the 1 of phrase. Three knob turns, no audible glitch.

### Harmonic mixing (Camelot)

Tracks have a key. Wrong-key mashups sound dissonant — that *is* noise.

- Each key gets a code: number 1–12 + letter A (minor) or B (major). E.g. C major = 8B, A minor = 8A.
- Compatible neighbors: same number (8A↔8B, relative major/minor), ±1 number same letter (8A↔7A↔9A).
- Energy boost trick: **+7 semitones = +1 number** on the wheel feels like a lift without dissonance.
- Acceptable for short blends only — full overlay needs same code.

### Stems / layering

Modern DJs split tracks live into 4 stems: vocals, melody, bass, drums. Lets you:
- Drop a vocal acapella over an incoming instrumental's drums
- Kill the outgoing kick early so the incoming sub can come up clean (no kick clash)
- Loop one stem (e.g. vocal) while everything else changes underneath

This is highly relevant to AiJockey: stem-aware DSP means we can **swap kicks cleanly even if the tracks are not perfectly beatmatched**, because we're not subtracting wave from wave — we're hard-cutting one stem.

### Phrasing rules of thumb

1. Cue the incoming track's first downbeat (the **1**) to land on the outgoing track's **1** of a phrase boundary.
2. Long blends (32+ bars) start on phrase, end on phrase — both ends matter.
3. Section roles must be compatible: outgoing breakdown ↔ incoming intro is golden. Outgoing drop ↔ incoming drop is *only* legal if executing a deliberate double-drop.

---

## 3. Mix-Type Taxonomy → AiJockey Tier Mapping

Current code has 5 tiers: `minor / major / drop / cut / loop`. Each should pick a *distinct* DSP technique. Below: every common DJ transition, classified by tier.

### MINOR — invisible blend (low drama, ~4–8 bars)

The "you didn't notice the change" mix. Used between similar tracks in similar sections.

| Technique | What | When |
|---|---|---|
| **Beat blend** | Both tracks play; volume crossfade over 8–16 bars | Default |
| **EQ swap (high)** | Cross-fade only the highs first, then mids, then bass last on the 1 | Tight same-genre blends |
| **Long volume ride** | 32-bar volume crossfade, no EQ moves | Ambient, downtempo, lo-fi |

### MAJOR — structurally significant (clearly audible, ~8–16 bars)

You hear the change. Adds shape but not climax.

| Technique | What | When |
|---|---|---|
| **Filter fade** | Low-pass sweep on outgoing (treble disappears), high-pass on incoming (bass disappears), they meet in the mids, then resolve | Energy lift between sections |
| **Echo-out tail** | Outgoing track gets heavy tape/dub delay; dry signal fades while delays continue, incoming starts clean | Genre/BPM jumps |
| **Drum-only break** | Strip outgoing to drums for 4 bars, layer incoming melody, then swap | Showcase incoming melody |
| **Held silence** | All bands cut for 1 beat → incoming on the 1 | Tension reset |

### DROP — climax engineered (8 bars build → 1 hit)

Energy peak. The crowd's hands go up.

| Technique | What | When |
|---|---|---|
| **Build + drop** | Riser/snare-roll on outgoing under last 8 bars, kick removed in last 2, incoming drops on the 1 | Trance/EDM/big-room |
| **Double drop** | Both tracks' drops aligned to hit on the same downbeat | DnB, dubstep, festival peaks |
| **Reverse-cymbal launch** | Reverse cymbal swell as transition glue, drop on resolution | Anywhere |

### CUT — instant, no blend (1 beat or less)

Highest contrast. Brutalist. Resets the room.

| Technique | What | When |
|---|---|---|
| **Hard cut on 1** | Outgoing stops dead, incoming starts, both on the 1 | After a build, after silence, BPM jump |
| **Cut + impact** | Hard cut + impact/whoosh sample to mask the seam | Genre swap |
| **Vocal stab cut** | Outgoing cuts mid-phrase, vocal stab marks the seam, incoming on next 1 | Hip-hop, mashups |

### LOOP — capture and stretch (variable length)

Buy time. Build tension. Convert one bar into a destination.

| Technique | What | When |
|---|---|---|
| **Beat juggle / loop roll** | 1-beat or 1/2-beat loop on outgoing, halve repeatedly (1 → 1/2 → 1/4 → 1/8) into incoming drop | Tension peak before drop |
| **Acapella loop** | Loop outgoing's vocal phrase over incoming's drums | Mashup feel |
| **8-bar stutter loop** | Lock outgoing's last 8 bars, ride filter, incoming on phrase end | Stalling for phrase alignment |

---

## 4. DJ Set Energy Curves (set-level structure)

Three macro shapes (each track placement should serve the curve):

- **Ramp** — monotonic energy increase. Warmup → main set → peak. Most common.
- **Wave** — peaks and valleys. 4–6 bar peaks separated by valleys so peaks hit harder. Festival headliner standard.
- **Flat** — sustained energy band. After-hours, lo-fi, ambient.

**Programming rules**:
- Valleys make peaks hit harder. A 90-min set at peak energy = numbing.
- Warmup ≠ peak time. Playing peak-time energy in a warmup slot ruins the headliner.
- Tempo can stay flat while energy rises (via instrumentation density, sub-bass, vocal hooks).

> **AiJockey arc presets** (`peak`, `flat_low`, `rollercoaster`, `build`) already model this. Director should pick tier *distributions* that match the arc — e.g. `flat_low` should be ~all `minor`, `rollercoaster` should alternate `drop`/`minor`/`cut`.

---

## 5. Diagnosis Hooks for the "Noise" Problem

Common causes of perceived noise in DJ mixes (and AiJockey-specific):

1. **Off-phrase junction** — cuepoints not snapped to 8/16-bar grid. Fix: phrase-quantize all junctions before DSP.
2. **Kick clash** — both tracks' lows playing simultaneously. Fix: enforce "low band has one owner" — bass-swap on the 1.
3. **Key clash** — incompatible Camelot codes during overlap. Fix: gate long blends on key compatibility, force `cut` or `echo_out` for incompatible pairs.
4. **Section role mismatch** — drop-into-drop without a planned double-drop, or breakdown-into-breakdown (energy crater). Fix: validate (out_section, in_section) pair before approving tier.
5. **Tier collapse** — every tier picking the same DSP (the `eq_swap` collapse already fixed). Fix: enforce distinct DSP per tier.
6. **Over-effecting** — risers + impacts + filter + echo all on same junction. Fix: budget at most 2 accent FX per junction.

---

## 6. Beyond Transitions — Other Mix/Production Forms

So far covered: how DJs *transition* between finished tracks. But "mixing" means more in DJ/producer culture. Below: forms where the source material itself is altered or combined, not just sequenced.

Two axes classify everything:
- **Authorization** — official (label-approved) vs unofficial (bootleg/edit underground)
- **Source granularity** — finished mixdown only, or full multitrack stems

| Form | Source | Authorized? | What changes | Typical use |
|---|---|---|---|---|
| **Remix** | Original stems | Yes | Full rebuild — new drums, arrangement, often new BPM/key | New version released by label |
| **Bootleg** | Finished track ± DIY acapella | No | Like remix but no stem access; uses isolation/AI-extracted parts | Underground, free downloads |
| **Edit / Re-edit** | Finished mixdown | Often unofficial | Rearrange existing audio: extend intro/outro, loop sections, remove vocals, restructure for DJ use | DJ tools |
| **Extended Mix** | Stems or mixdown | Yes | Standard radio version stretched: longer intro/outro for mixing in/out | Club/DJ-friendly release |
| **VIP** | Original stems | Yes (self) | Original artist's own rework — heavier, dancefloor-tuned | Live sets, IDs |
| **Dub mix** | Stems | Yes | Vocals removed/reduced, melody kept | Remixer source material; DJ blends |
| **Flip / Rework** | Anything | Varies | Vibe overhaul of already-remixed/edited track | Genre conversion (techno flip of pop, etc) |
| **Mashup** | 2+ finished tracks | No | Stack acapella of A over instrumental of B | Live blend / studio production |
| **Megamix** | Many finished tracks | Sometimes | Compressed medley — short snippets back-to-back | Showcase, party openers |

### Sampling (production primitive that fuels all the above)

Sampling = take a chunk of existing audio, reuse as building block. Five flavors:

| Sample type | Length | Treatment | Example |
|---|---|---|---|
| **One-shot** | Single hit | Triggered like instrument | Snare from old break, vocal stab |
| **Loop sample** | 1–4 bars | Repeated under new beat | Hip-hop drum break, jazz loop |
| **Vocal chop** | Sub-word fragments (vowels, syllables) | Pitch/time-shifted, sequenced melodically | EDM drop hook, future bass lead |
| **Chop / flip** | Multi-bar passage | Sliced, reordered, re-pitched into new pattern | Kanye/J Dilla style soul flips |
| **Found sound** | Anything | Heavy processing, often unrecognizable | Field recording → percussion |

Production tools per flavor: pitch shift, time-stretch, reverse, formant shift, granular freeze, slice-to-MIDI, layering.

### DJ performance styles (set-level identity, not single transitions)

| Style | Defining trait |
|---|---|
| **Beatmix / blend** | Long phrase-aligned blends. House/techno standard. |
| **Open format** | Genre-hops, BPM-jumps, no commitment to beatmatching. Cut/echo/scratch heavy. Weddings, top-40 clubs. |
| **Turntablism** | Turntable as instrument. Scratch patterns, beat juggling, no "songs" in normal sense. |
| **Scratch DJ / battle** | Subset of turntablism — competitive scratch routines over short loops. |
| **Live remix / live edit** | DJ rebuilds tracks on-the-fly with stems, loops, samplers, FX. Crosses into producer territory. |
| **Live mashup** | Open-format + stem isolation: acapella of A dropped over B's drums *during the set*. |
| **Megamix DJ** | High-density medley sets, ~30 sec per track, tight cuts. Radio/showcase format. |

### Implications for AiJockey

Right now the system is in **beatmix/blend** territory: take two finished tracks, transition between them. Adjacent forms unlocked by stem separation (which we already have via Demucs/similar):

1. **Live mashup mode** — at junction, drop incoming acapella stem over outgoing instrumental for N bars before full handoff. New tier candidate: `mashup`.
2. **Edit-on-the-fly** — re-loop a section (e.g. extend a too-short intro by looping its first 8 bars). Already partially modeled by `loop` tier; could be its own pre-junction operation.
3. **Megamix mode** — set-level macro: short clips, dense cuts. Maps to an arc/program option (e.g. `arc=megamix` → high `cut` density, short clip durations).
4. **Scratch / turntablism FX** — accent layer at junctions: scratch sample on impact (already supported via `accent_hints` fx_category). Add `scratch` category.
5. **Vocal chop accent** — chop incoming vocal stem into 1-beat chops over outgoing's last 4 bars. New accent category.
6. **VIP-style rework** — out-of-scope (requires full re-production), but the *aesthetic* (heavier kick, replaced drums) could be approximated by stem-swap accents.

These are not transitions — they are *forms the system can produce*. Worth modeling as orthogonal to tier (a `mashup` blend can still be `major` or `drop` tier).

---

## 7. Beyond DJing — Music Production Roles, Tools, and "What Feels Good"

DJ-thinking is one lens. But the system should aim higher: produce *sound that feels good* — which means borrowing from the people who *make* the source material, not just sequence it.

### 7.1 Roles in music creation (who does what)

| Role | Primary concern | What they decide |
|---|---|---|
| **Composer** | Notes — melody, harmony, chord progression, bassline | What is played |
| **Songwriter** | Song form + lyrics + hook | What story / emotional shape |
| **Arranger** | Instrumentation + section flow + counter-melodies | Who plays what, when |
| **Producer** | Vision + casting + creative direction | Sonic identity, what makes it the *song* |
| **Sound designer** | Timbre — synth patches, FX, atmospheres, one-shots | What it sounds like, beyond notes |
| **Recording engineer** | Capture — mic choice/placement, gain staging | How performance becomes audio |
| **Mixing engineer** | Balance — levels, EQ, compression, panning, FX, reverb | How elements sit together |
| **Mastering engineer** | Final polish — loudness, tonal balance, format | How it sounds on every system |
| **Performer / session musician** | Execution — feel, dynamics, micro-timing | How the part actually breathes |
| **DJ** | Sequencing + transitions of finished tracks | What plays after what |

> **AiJockey is currently the DJ.** Each row above is a missing capability the system *could* invoke. Roles relevant to "less noise, more good": **mixing engineer** (balance/EQ), **arranger** (section choice within a clip), **sound designer** (FX/accents), **producer** (taste/vision — the Director's job).

### 7.2 Instruments / tools (what sound comes from)

#### Sound generators

| Tool | How it makes sound |
|---|---|
| **Synthesizer** | Generates audio from oscillators + filters + envelopes |
| **Drum machine** | Pre-built drum sounds (synthesized or sampled) + step sequencer |
| **Sampler** | Plays back recorded audio, pitch/time-shifted, sliced |
| **Groovebox** | Synth + drum machine + sampler + sequencer in one box |
| **Acoustic instrument** | Strings, drums, horns, voice — captured via microphone |
| **MIDI controller** | No sound itself — triggers any of the above |

#### Workflow tools

| Tool | Purpose |
|---|---|
| **DAW** (Ableton, Logic, FL, Pro Tools) | Multi-track recording, MIDI sequencing, mixing — the studio |
| **Sequencer** | Arranges note/trigger events in time |
| **Plugin (VST/AU)** | Synth, sampler, or effect inside the DAW |
| **Effects unit** | Reverb, delay, distortion, modulation, EQ, compression |
| **Stem separator** (Demucs, Spleeter, etc) | Extract vocals/drums/bass/other from finished track |

#### Effects (the "shape" of sound, not the source)

| Effect | What it does | When |
|---|---|---|
| **EQ** | Boost/cut frequency bands | Carve space, fix muddiness |
| **Compression** | Reduce dynamic range | Glue, punch, control |
| **Reverb** | Simulate space | Depth, atmosphere |
| **Delay / echo** | Repeat with decay | Rhythm interest, transition glue |
| **Distortion / saturation** | Add harmonics | Warmth, aggression |
| **Chorus / flanger / phaser** | Modulation | Width, movement |
| **Filter** (HP/LP/BP) | Remove frequency range | Sweeps, transitions, focus |
| **Sidechain compression** | Duck one signal when another hits | "Pumping" kick-bass relationship |
| **Limiter** | Hard ceiling | Mastering, prevent clipping |

### 7.3 Sound design — making the timbre itself

Synthesis methods, in order of complexity:

| Method | How | Sound character | Example |
|---|---|---|---|
| **Subtractive** | Rich osc → filter carves | Classic warm, analog | Moog leads, 303 acid bass |
| **Additive** | Stack sine waves | Pure, controlled, clinical | Hammond organ |
| **FM** (frequency modulation) | One osc modulates another's frequency | Metallic, bell-like, bass | DX7 electric piano, sub bass |
| **Wavetable** | Scan through saved waveforms | Evolving, modern, animated | Serum, Vital — pop/EDM lead |
| **Granular** | Slice audio into 1–100ms grains, replay | Textural, ambient, glitchy | Pad/atmosphere from any input |
| **Physical modeling** | Math model of real instrument | Organic, expressive | Plucked string, blown reed |

Recording-based sources:

- **Foley** — recorded everyday sounds (footsteps, fabric, glass) used as percussion or texture.
- **Field recording** — captured outside studio (rain, traffic, crowd) for atmosphere.
- **One-shot library** — pre-curated kicks, snares, hits ready to drop in.

### 7.4 What makes music *feel good* (the actually important part)

Across psychoacoustics + groove research, recurring principles:

#### Timing — the "pocket"

- **Quantized perfectly** = mechanical, dead. **Random** = sloppy. **Pocket** = controlled micro-deviations.
- **Swing**: delay the offbeat slightly (e.g. 16th-note offbeats by 5–15%). Forward motion.
- **Push and pull**: certain elements ahead of grid (build energy), others behind (relax). Hats forward + kick centered + bass slightly back is a classic feel.
- **Humanization**: vary velocity ±5–15%, vary timing ±5–10ms, vary note length. Real performers cannot play identical hits.
- **Tempo modulation**: drop 3–5 BPM for one bar before chorus = "breath." Live bands do this unconsciously.

#### Frequency — the "fit"

- One element per band (already covered in §2). No two things fight for the same Hz range.
- **Sub** (<60 Hz) — kick + sub-bass only. Mono. Felt, not heard.
- **Low-mid** (250–500 Hz) — mud zone. Cut here aggressively on most elements except bass body.
- **Presence** (2–5 kHz) — vocals, leads, snare crack. Overcrowding = harshness.
- **Air** (>10 kHz) — cymbals, breath, sparkle. Lift here for "expensive" sound.

#### Dynamics — tension and release

- **Loud constantly = boring + tiring.** Brain adapts after ~30 seconds. Need contrast.
- **Quiet section makes loud section hit harder.** Same waveform, different perceived impact.
- **Sidechain pumping**: kick ducks bass/pads, creating breathing rhythm at kick rate.
- **Transient vs body**: punchy attack + soft body = "snap"; soft attack + sustained body = "wash."

#### Harmony — consonance vs dissonance

- **Consonance** (perfect intervals, major/minor triads) = stable, "resolved."
- **Dissonance** (tritone, m2, M7) = unstable, "wants to move." Used as tension *toward* resolution.
- **Modal color**: minor = sad/dark, Dorian = melancholy-cool, Phrygian = exotic/tense, Lydian = dreamy.
- Emotional weight comes from movement *between* tension and release, not from either alone.

#### Repetition + variation

- **Repetition** = familiarity, hypnosis, dancefloor lock-in.
- **Variation** = surprise, prevents boredom.
- 4-bar loop repeated 8 times feels different from 4-bar loop with one element changing per repetition.
- DJ rule: change *one* thing every 8 bars (or 4, or 16 — pick a phrase scale).

### 7.5 Implications for AiJockey

System currently operates at DJ layer (sequence + transition). Cheapest wins on "feels good":

1. **Mixing-engineer pass at junctions** — frequency-aware crossfade. EQ-carve outgoing track in incoming's dominant band before swap. Low-hanging fruit; partly done.
2. **Sidechain pumping** — when kicks must overlap briefly, sidechain incoming bass to outgoing kick. Eliminates clash without hard mute.
3. **Humanization on accents** — current accent FX (risers/impacts/snare-rolls) likely fire on grid. Add ±10ms jitter + ±10% velocity variance. Instantly less "robot."
4. **Reverb tail bridging** — at echo-out / cut transitions, add short reverb tail on outgoing's last hit. Masks seam, smooths perceived edit.
5. **Tension-and-release at set level** — Director already plans arcs; add explicit "breath" insertion (one quiet bar before each `drop` tier).
6. **Sound-designer accents** — beyond stock FX library, allow granular freeze of outgoing's last bar as transitional pad. Free atmosphere from material we already have.
7. **Producer taste filter** — Director's role. Penalize plans where same DSP fires twice in row, or accent density > 2 per junction (already noted). Variation = listenability.
8. **Stem-aware mix** (already raised in §6) — don't sum waves; swap roles. Outgoing keeps melody, incoming brings drums + bass. Produces a third thing that's neither track, more like a live mashup. This is the biggest upgrade per unit work.

The ladder: **DJ → live remix → producer**. Each step adds capability and reduces "noise" because the system has more handles than just volume + EQ + tier choice.

---

## Sources

- [Mastering EDM Song Structure — MixElite](https://mixelite.com/blog/edm-song-structure/)
- [EDM Song Structure — Cymatics](https://cymatics.fm/blogs/production/edm-song-structure)
- [16 Basic DJ Transition Techniques — DJ.Studio](https://dj.studio/blog/basic-transition-techniques)
- [12 High-Energy DJ Transitions — We Are Crossfader](https://wearecrossfader.co.uk/blog/12-high-energy-dj-transitions/)
- [Mixing techniques behind every major genre — Pioneer DJ](https://blog.pioneerdj.com/djtips/we-uncover-the-mixing-techniques-behind-every-major-genre/)
- [DJ Blending Techniques — PulseDJ](https://blog.pulsedj.com/dj-blending)
- [Camelot Wheel — Mixed In Key](https://mixedinkey.com/camelot-wheel/)
- [Harmonic Mixing Guide — Mixed In Key](https://mixedinkey.com/harmonic-mixing-guide/)
- [The DJ's Guide to the Camelot Wheel — DJ.Studio](https://dj.studio/blog/camelot-wheel)
- [DJ Set Energy Flow — HarmonySet](https://www.harmonyset.com/guides/dj-set-energy-flow)
- [How to Build Energy in a DJ Set — DJs On Demand](https://www.djsondemand.co.uk/how-to-build-energy-in-a-dj-set/)
- [Step-by-Step Peak Time Techno — Beatportal](https://www.beatportal.com/articles/783088-step-by-step-guide-to-producing-techno-peak-time-driving-in-the-style-of-layton-giordani-eli-brown-and-adam-beyer)
- [VirtualDJ Real-Time Stems](https://www.virtualdj.com/stems/)
- [How DJs Use Stem Separation (2026) — StemSplit](https://stemsplit.io/blog/dj-stem-separation-guide)
- [Extracting Acapellas — DJ.Studio](https://dj.studio/blog/acapellas-for-djs)
- [What is DJ Phrase Mixing — DJ.Studio](https://dj.studio/blog/phrasing-dj-mixing)
- [How to DJ 101: Phrasing — DJ TechTools](https://djtechtools.com/2014/11/16/how-to-dj-101-why-you-must-understand-phrasing/)
- [How to Match Phrases — DJing Tips](https://www.djingtips.com/how-to-dj/how-to-match-phrases/)
- [Bootlegs, Mashups, Re-edits & Remixes — Digital DJ Tips](https://www.digitaldjtips.com/bootlegs-mashups-re-edits-remixes/)
- [Bootleg vs Mashup vs Remix — EKM.CO](https://ekm.co/difference-between-bootlegmashup-and/)
- [Remix vs Edit vs Bootleg — SirenMix](https://sirenmix.com/blog/remix-vs-edit-vs-bootleg-explained)
- [Remix vs Edit vs VIP vs Bootleg vs Rework — Housewub](https://housewub.org/remix-vs-edit-vs-vip-vs-bootleg-vs-rework/)
- [Remix vs Re-Edit — DJ TechTools](https://djtechtools.com/2018/09/26/whats-the-difference-between-a-remix-and-a-re-edit/)
- [Sample Chopping — RouteNote](https://create.routenote.com/blog/sample-chopping-how-to-flip-a-sample-like-a-boss/)
- [Music Sampling Explained — LA Film School](https://www.lafilm.edu/blog/music-sampling-explained/)
- [Vocal Chops — Output](https://output.com/blog/vocal-chop-sounds)
- [23 Advanced DJ Mixing Techniques — DJ.Studio](https://dj.studio/blog/advanced-dj-mixing-techniques)
- [Open Format Scratch Mixing — Digital DJ Tips](https://1.digitaldjtips.com/ofsm-info)
- [DJ mix — Wikipedia](https://en.wikipedia.org/wiki/DJ_mix)
- [Record producer — Wikipedia](https://en.wikipedia.org/wiki/Record_producer)
- [Producer/Engineer/Arranger/Songwriter roles — Evergreen Records](https://evergreenrecords.com/blog/the-difference-between-producer-engineer-arranger-songwriter-mix-engineer-mastering-engineer-etc)
- [Composer/Arranger/Producer roles — OctaveUp](https://octaveup.com/index.php/2023/11/13/demystifying-the-roles-in-music-production-composer-arranger-and-producer/)
- [Drum Machine vs Sampler — Sweetwater](https://www.sweetwater.com/insync/drum-machine-vs-sampler-whats-the-difference/)
- [Types of Synthesis — LANDR](https://blog.landr.com/types-of-synthesis/)
- [Subtractive vs Additive vs FM vs Wavetable — Splice](https://splice.com/blog/difference-between-synthesis-types/)
- [Granular Synthesis — Unison](https://unison.audio/granular-synthesis/)
- [Sound design 101 — Native Instruments](https://blog.native-instruments.com/sound-design/)
- [Swing, Shuffle, Humanization — Sample Focus](https://blog.samplefocus.com/blog/swing-shuffle-and-humanization-how-to-program-grooves/)
- [Psychoacoustics 101 — Unison](https://unison.audio/psychoacoustics/)
- [Psychoacoustics of Drums — Attack Magazine](https://www.attackmagazine.com/technique/tutorials/the-psychoacoustics-of-drums/)
- [Humanize MIDI — Unison](https://unison.audio/how-to-humanize-midi/)
- [How to Make Your Music Sound More Human — Magnetic Mag](https://magneticmag.com/2026/01/how-to-make-your-music-sound-more-human/)
- [Groove sensation research — Frontiers in Psychology](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2014.00894/full)
