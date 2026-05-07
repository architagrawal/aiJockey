# Dataset Scraping — Legal + Practical Notes

## Goal

Build (pre, transition, post) audio triplets from real DJ mixes for fine-tuning
a generative bridge model (Tier 3).

## Pipeline

```
1. Find DJ mix + tracklist (manual)
       e.g. https://1001tracklists.com/...
       OR YouTube comment with timestamps
       OR Soundcloud description

2. Create JSON (datasets/tracklists/<name>.json)
       fill in URL + transition timestamps
       see scrape/tracklists.py:example_skeleton

3. Run dataset_builder.py
       downloads via yt-dlp
       extracts ±16s windows around each transition
       computes features (tempo, CLAP)
       auto-labels technique heuristically

4. Output: datasets/transitions_real/<mix_id>/*.wav + index.json
```

## Legal posture

- **Personal/research use** — generally tolerated under fair-use doctrines
  in many jurisdictions. NOT a guarantee.
- **Do NOT redistribute downloaded audio**. Keep `datasets/raw_mixes/` local.
- **Models trained on this data inherit AGPL-3.0** — anyone using them must
  open-source their fork. This protects against commercial appropriation
  but does not absolve copyright concerns.
- **Avoid commercial deployment** until copyright cleared. Possible paths:
  - License directly from DJs/labels
  - Use only CC-licensed mixes (Free Music Archive, some Bandcamp)
  - Use only your own mixes / mixes you have rights to

`.gitignore` excludes `datasets/` from version control to prevent accidental
public release of raw audio.

## Manual tracklist workflow (recommended)

1. Find a DJ set on YouTube with detailed timestamps (look in description or
   pinned comment, OR cross-reference with 1001tracklists).
2. Run `python src/training/scrape/tracklists.py skeleton` to get a JSON template.
3. Fill in URL + transitions list.
4. Save under `datasets/tracklists/`.
5. Run `python src/training/dataset_builder.py`.

Example sources for free/CC-permissive DJ mixes:
- NTS Radio archive (some shows licensed)
- BBC Essential Mix archive (UK, some episodes free)
- DJ-uploaded Soundcloud sets with tracklists
- Mixcloud free downloads

## Quality matters more than quantity

20 well-annotated mixes (~600 transitions) > 200 sloppy ones. Manual review of
auto-detected technique labels saves a lot of training pain.

## Future: 1001tracklists scraper

`tracklists.py` has a stub. 1001tracklists has anti-scraping; manual entry
from their site into your local JSON is the friendly path. If/when official
API exists, automate.

## Future: licensed datasets (paid path)

- **Beatport DJ Sets** — not aligned but could be paid for
- **Splice transition packs** — pro-engineered, labeled, $10-30/pack
- **Hire DJs to record labeled transitions** — $200-500 for 100 high-quality
  examples

These are out of scope for free-tier MVP but mentioned for future scaling.

## Compute footprint

- yt-dlp download: ~30-50 MB per hour of mix at 256kbps
- Extracted windows: ~1.5 MB per transition (8s + 16s + 16s = 40s @ 44.1kHz stereo)
- 20 mixes × 30 transitions = 600 samples = ~900 MB

Storage is cheap. CLAP embeddings + features add ~5KB per sample.

## How to share your dataset

Don't share raw audio. Share:
- The tracklists JSON files (legal — just URL + timestamps)
- CLAP embeddings (legal — derived features)
- Trained model weights (AGPL-3.0)

Someone else can rebuild the audio dataset themselves from your tracklists.
