"""S0 — direct-to-MI300X downloader.

No laptop disk. Fetches CC-licensed music corpora to /scratch/raw/{src}/.
Sources allowed: mixotic, fma, mtg_jamendo, internet_archive.

aria2 used when available (parallel chunked download), curl fallback.

Usage:
    python scripts/stage0_download.py --src mixotic,fma_medium --max-per-src 1000
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.common import scratch_dir, write_json, atomic_write


# ---------------------------------------------------------------------------
# Source URL listers
# ---------------------------------------------------------------------------

def _mixotic_urls(max_n: int) -> list[tuple[str, str]]:
    """Mixotic.net CC-licensed DJ mixes. Public download URLs.

    Returns list of (url, suggested_filename). Real implementation should
    scrape the catalog index page; this stub uses a known prefix pattern.
    """
    # Known catalog uses sequential IDs at https://mixotic.net/mixes/####/
    # Real download URL is fetched via per-mix page parse; deferred to
    # an actual scraper. Stub returns empty list when offline.
    return []


def _fma_urls(max_n: int, subset: str = 'medium') -> list[tuple[str, str]]:
    """Free Music Archive bulk archive (zenodo or HF datasets).

    The archive on Zenodo is a single tar; we delegate to HF datasets
    streaming if available rather than mirror the tar locally.
    """
    return []


def _mtg_jamendo_urls(max_n: int, genre_filter: list[str] | None = None
                      ) -> list[tuple[str, str]]:
    """MTG-Jamendo via HF datasets — streamed in S1, no S0 needed."""
    return []


SOURCES = {
    'mixotic': _mixotic_urls,
    'fma_medium': lambda n: _fma_urls(n, 'medium'),
    'fma_small': lambda n: _fma_urls(n, 'small'),
    'mtg_jamendo': _mtg_jamendo_urls,
}


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _fetch_one(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    if shutil.which('aria2c'):
        cmd = ['aria2c', '-x', '4', '-s', '4', '-q', '--allow-overwrite=true',
               '-o', tmp.name, '-d', str(tmp.parent), url]
    elif shutil.which('curl'):
        cmd = ['curl', '-sSL', '--retry', '3', '--retry-delay', '5',
               '-o', str(tmp), url]
    else:
        print(f"warn: neither aria2c nor curl available; cannot fetch {url}")
        return False
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except Exception as e:
        print(f"warn: download failed for {url}: {e}")
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        return False
    if tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(dest)
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='mixotic',
                    help='comma-separated source names')
    ap.add_argument('--max-per-src', type=int, default=500)
    ap.add_argument('--manifest-out', default='downloaded.json')
    args = ap.parse_args()

    sources = [s.strip() for s in args.src.split(',') if s.strip()]
    raw_root = scratch_dir('raw')
    manifest = []

    for src in sources:
        lister = SOURCES.get(src)
        if lister is None:
            print(f"warn: unknown source {src}; skipping")
            continue
        urls = lister(args.max_per_src)
        if not urls:
            print(f"info: no urls returned for {src} (lister stub or offline)")
            continue
        src_dir = raw_root / src
        src_dir.mkdir(parents=True, exist_ok=True)
        ok_count = 0
        for url, fname in urls:
            dest = src_dir / fname
            if _fetch_one(url, dest):
                manifest.append({'src': src, 'url': url, 'path': str(dest)})
                ok_count += 1
        print(f"{src}: {ok_count}/{len(urls)} downloaded")

    write_json(raw_root / args.manifest_out, manifest)
    print(f"wrote manifest with {len(manifest)} entries to {raw_root / args.manifest_out}")


if __name__ == '__main__':
    main()
