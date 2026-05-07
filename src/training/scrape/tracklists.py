"""
Tracklist parsing.

Two paths:
1. **Manual** (recommended, legal, easy): you provide a JSON file listing
   timestamps and track titles for a downloaded mix. Format below.
2. **Stub** for 1001tracklists scraping — left as TODO. Their site has
   anti-scraping; manual entry from their site is friendlier.

Manual JSON schema:
{
  "url": "https://youtube.com/watch?v=...",
  "title": "Solomun BBC Radio 1 Essential Mix",
  "duration_sec": 7200,
  "transitions": [
    {"at_sec":  120.5, "from_track": "Track A", "to_track": "Track B"},
    {"at_sec":  280.0, "from_track": "Track B", "to_track": "Track C"},
    ...
  ]
}

Each entry in `transitions` marks the moment a transition COMPLETES (i.e. the
incoming track is fully in). The dataset builder extracts a window centered on
this timestamp.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TransitionMark:
    at_sec: float
    from_track: str
    to_track: str


@dataclass
class MixTracklist:
    url: str
    title: str
    duration_sec: float
    transitions: list[TransitionMark]


def load_manual(json_path: str) -> MixTracklist:
    with open(json_path) as f:
        d = json.load(f)
    transitions = [TransitionMark(at_sec=float(t['at_sec']),
                                  from_track=t.get('from_track', ''),
                                  to_track=t.get('to_track', ''))
                   for t in d.get('transitions', [])]
    return MixTracklist(
        url=d.get('url', ''),
        title=d.get('title', ''),
        duration_sec=float(d.get('duration_sec', 0.0)),
        transitions=transitions,
    )


def load_all(tracklists_dir: str) -> list[MixTracklist]:
    out: list[MixTracklist] = []
    for jp in sorted(Path(tracklists_dir).glob('*.json')):
        try:
            out.append(load_manual(str(jp)))
        except Exception as e:
            print(f"warn: failed to load {jp}: {e}")
    return out


def example_skeleton(out_path: str = 'datasets/tracklists/example.json') -> None:
    """Write an empty skeleton tracklist file for user to fill in."""
    skeleton = {
        "url": "https://www.youtube.com/watch?v=REPLACE_ME",
        "title": "REPLACE_WITH_MIX_NAME",
        "duration_sec": 3600,
        "transitions": [
            {"at_sec": 120.5, "from_track": "Track 1 - Artist A", "to_track": "Track 2 - Artist B"},
            {"at_sec": 280.0, "from_track": "Track 2 - Artist B", "to_track": "Track 3 - Artist C"},
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(skeleton, f, indent=2)
    print(f"wrote skeleton: {out_path}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('skeleton')
    p_load = sub.add_parser('load')
    p_load.add_argument('--dir', default='datasets/tracklists')
    args = ap.parse_args()
    if args.cmd == 'skeleton':
        example_skeleton()
    else:
        for t in load_all(args.dir):
            print(f"{t.title}: {len(t.transitions)} transitions")
