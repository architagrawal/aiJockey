# Style-RAG Reference Mixes

Drop reference timelines here as JSON. Planner queries top-K most-similar
transition contexts and biases technique selection toward what pro DJs did
in similar situations.

## Format

`<name>.json`:

```json
{
  "name": "Solomun Sunset Set 2023",
  "transitions": [
    {
      "out_clap": [0.012, -0.034, ..., 0.001],
      "in_clap":  [0.005,  0.022, ..., -0.011],
      "out_energy": 0.7,
      "in_energy": 0.85,
      "technique": "eq_swap",
      "bars": 32
    }
  ]
}
```

## How to build references

1. Pick a curated DJ set you admire
2. Identify each transition (timestamp, technique used, energy levels)
3. Get CLAP embeddings of the outgoing + incoming clips around each transition
   (use `analyze.py` on the source tracks, then read `cache/*.npz['clap']`)
4. Save as JSON here

Helper: `src/style_rag.py:build_pattern_from_clips()` constructs the dict.

## Usage

```bash
python src/main.py plan --cache cache/ --style_rag references/
# or
python src/main.py all --clips clips/ --style_rag references/
```

Empty dir = no bias (planner falls back to rule-based selection).

## Build script (TODO)

A future `scripts/build_references.py` will:
1. Take a folder of reference DJ set audio + manual transition annotations
2. Auto-extract CLAP embeddings around each transition
3. Write reference JSON files

For now, build manually or skip Style-RAG.
