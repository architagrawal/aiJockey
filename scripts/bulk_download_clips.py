"""
Bulk-download DJ-friendly tracks across multiple genres into clips/.

Two modes:
  1. From a list of playlist URLs (default — curated electronic playlists)
  2. From a custom YAML/JSON of (genre, url) pairs

Each track:
  - Downloaded as 44.1 kHz wav via yt-dlp
  - Filename: <genre>__<title>.wav (sanitized)
  - Skipped if already present (idempotent)

Default cap: 5 tracks per playlist. Tweak with --per_playlist.

LEGAL: personal/research use. Don't redistribute. AGPL doesn't override
upstream copyright.
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


# Curated electronic playlists. Mix of CC-BY (NCS, Argofox) and label channels.
# URLs may go stale — replace as needed.
DEFAULT_PLAYLISTS = [
    # Genre, playlist URL, recommended max tracks
    ('edm',      'https://www.youtube.com/@NoCopyrightSounds/playlists',  5),
    ('edm',      'https://www.youtube.com/@argofox/videos',               5),
    ('house',    'https://www.youtube.com/results?search_query=deep+house+free+download', 5),
    ('techno',   'https://www.youtube.com/results?search_query=melodic+techno+free+download', 5),
    ('trance',   'https://www.youtube.com/results?search_query=anjunabeats+free+download', 5),
    ('dnb',      'https://www.youtube.com/@LiquicityRecords/videos',      5),
    ('dubstep',  'https://www.youtube.com/@MonstercatUncaged/videos',     5),
    ('trap',     'https://www.youtube.com/@TrapNation/videos',            5),
]


SAFE = re.compile(r'[^A-Za-z0-9._-]')


def safe(s: str) -> str:
    return SAFE.sub('_', s)[:80]


def get_video_ids(playlist_url: str, n: int) -> list[tuple[str, str, int]]:
    """Returns list of (id, title, duration_sec) for first n videos in playlist."""
    cmd = ['yt-dlp', '--flat-playlist', '--playlist-end', str(n),
           '--print', '%(id)s\t%(title)s\t%(duration)s', playlist_url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  WARN: yt-dlp listing failed: {r.stderr[:200]}")
        return []
    out = []
    for line in r.stdout.strip().splitlines():
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        vid_id = parts[0]
        title = parts[1] if len(parts) > 1 else 'unknown'
        dur = 0
        if len(parts) > 2 and parts[2] not in ('', 'NA', 'None'):
            try:
                dur = int(float(parts[2]))
            except Exception:
                dur = 0
        out.append((vid_id, title, dur))
    return out


def download_one(vid_id: str, title: str, genre: str,
                 out_dir: Path, sample_rate: int = 44100) -> Path | None:
    """Download one video as wav into out_dir/<genre>__<title>.wav"""
    fname = f"{genre}__{safe(title)}__{vid_id}.wav"
    target = out_dir / fname
    if target.exists():
        print(f"  [skip] {fname} (exists)")
        return target
    url = f'https://www.youtube.com/watch?v={vid_id}'
    cmd = [
        'yt-dlp', '-x',
        '--audio-format', 'wav',
        '--audio-quality', '0',
        '-o', str(out_dir / f'{genre}__{safe(title)}__{vid_id}.%(ext)s'),
        '--postprocessor-args', f'-ar {sample_rate}',
        '--no-warnings',
        url,
    ]
    print(f"  [dl] {genre} :: {title[:60]}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"     FAILED: {r.stderr[:200]}")
        return None
    if target.exists():
        return target
    # yt-dlp may have created different sanitized name; find by id substring
    for cand in out_dir.glob(f'{genre}__*{vid_id}*'):
        return cand
    return None


def bulk_download(playlists: list[tuple[str, str, int]],
                  clips_dir: str,
                  per_playlist: int | None = None,
                  duration_min: int = 90,
                  duration_max: int = 360) -> None:
    """
    Download tracks. Filters by duration range to skip podcasts/long mixes
    (we want individual tracks for clip pool).

    duration_min: skip clips shorter than this (default 1.5 min — avoids previews)
    duration_max: skip clips longer than this (default 6 min — avoids full DJ mixes)
    """
    out = Path(clips_dir)
    out.mkdir(parents=True, exist_ok=True)
    total_dl = 0; total_skipped = 0; total_failed = 0
    for genre, url, default_n in playlists:
        n = per_playlist if per_playlist is not None else default_n
        print(f"\n=== {genre.upper()} :: {url} (max {n}) ===")
        videos = get_video_ids(url, n * 3)  # over-fetch for filter
        in_range = [(vid, title, dur) for vid, title, dur in videos
                    if duration_min <= dur <= duration_max] if videos else []
        if videos and not in_range:
            print(f"  WARN: no videos in {duration_min}-{duration_max}s range; trying first {n} anyway")
            in_range = videos[:n]
        in_range = in_range[:n]
        for vid_id, title, dur in in_range:
            try:
                p = download_one(vid_id, title, genre, out)
                if p is None:
                    total_failed += 1
                else:
                    total_dl += 1
            except KeyboardInterrupt:
                print("\nINTERRUPTED")
                return
            except Exception as e:
                print(f"     ERROR: {e}")
                total_failed += 1

    print(f"\n=== DONE: downloaded {total_dl}, failed {total_failed} ===")


def load_custom(path: str) -> list[tuple[str, str, int]]:
    p = Path(path)
    with open(p) as f:
        if p.suffix in ('.yaml', '.yml'):
            import yaml
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    out = []
    for entry in data:
        out.append((entry['genre'], entry['url'], int(entry.get('n', 5))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips_dir', default='clips',
                    help='destination for downloaded .wav files')
    ap.add_argument('--per_playlist', type=int, default=None,
                    help='override per-playlist count (default: per-entry default)')
    ap.add_argument('--duration_min', type=int, default=90,
                    help='min track duration in seconds (skip previews)')
    ap.add_argument('--duration_max', type=int, default=360,
                    help='max track duration in seconds (skip full mixes)')
    ap.add_argument('--custom', default=None,
                    help='custom playlists JSON/YAML, else use DEFAULT_PLAYLISTS')
    args = ap.parse_args()

    if args.custom:
        playlists = load_custom(args.custom)
    else:
        playlists = DEFAULT_PLAYLISTS
    print(f"using {len(playlists)} playlist(s)")

    # Verify yt-dlp installed
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("yt-dlp not installed. Run: pip install yt-dlp")
        sys.exit(1)

    bulk_download(playlists, args.clips_dir,
                  per_playlist=args.per_playlist,
                  duration_min=args.duration_min,
                  duration_max=args.duration_max)


if __name__ == '__main__':
    main()
