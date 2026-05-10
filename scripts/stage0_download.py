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

def _http_get(url: str, timeout: int = 30) -> str | None:
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={'User-Agent': 'aijockey/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"warn: GET {url} failed: {e}")
        return None


def _mixotic_urls(max_n: int) -> list[tuple[str, str]]:
    """Mixotic.net catalog scraper.

    Catalog pattern observed: https://mixotic.net/mixes/?page=N (HTML index).
    Per-mix page contains a `<a href=".../download/...mp3">` direct link.

    We iterate index pages until max_n collected or pages exhausted. All
    Mixotic content is Creative Commons (CC-BY-NC-SA), suitable for
    research / fine-tune / non-commercial use. Production should still
    verify per-mix license tag.
    """
    import re
    out: list[tuple[str, str]] = []
    page = 1
    seen_mix_ids: set[str] = set()
    while len(out) < max_n and page <= 50:
        idx_url = f'https://mixotic.net/mixes/?page={page}'
        html = _http_get(idx_url)
        if not html:
            break
        # Find each mix link, e.g. /mixes/0712-FOO_BAR/
        mix_paths = set(re.findall(r'/mixes/(\d{4}-[A-Za-z0-9_\-]+)/?', html))
        if not mix_paths:
            break
        new_count = 0
        for mp in sorted(mix_paths):
            if mp in seen_mix_ids:
                continue
            seen_mix_ids.add(mp)
            mix_url = f'https://mixotic.net/mixes/{mp}/'
            mhtml = _http_get(mix_url)
            if not mhtml:
                continue
            dl = re.search(r'href="([^"]+\.mp3)"', mhtml)
            if not dl:
                continue
            url = dl.group(1)
            if url.startswith('/'):
                url = 'https://mixotic.net' + url
            fname = mp + '.mp3'
            out.append((url, fname))
            new_count += 1
            if len(out) >= max_n:
                break
        if new_count == 0:
            break
        page += 1
    return out


def _fma_urls(max_n: int, subset: str = 'medium') -> list[tuple[str, str]]:
    """Free Music Archive bulk archive on Zenodo.

    Single tar archive per subset. We download the tar and let stage1 walk
    contents. Returns one (url, fname) per subset request.
    """
    archives = {
        'small':  ('https://os.unil.cloud.switch.ch/fma/fma_small.zip',  'fma_small.zip'),
        'medium': ('https://os.unil.cloud.switch.ch/fma/fma_medium.zip', 'fma_medium.zip'),
        'large':  ('https://os.unil.cloud.switch.ch/fma/fma_large.zip',  'fma_large.zip'),
    }
    if subset not in archives:
        return []
    url, fname = archives[subset]
    # max_n unused here — single archive, expanded in stage1.
    return [(url, fname)]


def _mtg_jamendo_urls(max_n: int, genre_filter: list[str] | None = None
                      ) -> list[tuple[str, str]]:
    """MTG-Jamendo direct download.

    Audio is hosted on Jamendo's CDN as 96kbps mp3 by track id. The track-id
    list lives in the metadata TSV in the GitHub repo. We fetch that TSV,
    sample the requested count, and emit per-track URLs.
    """
    import re
    meta_url = ('https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/'
                'master/data/autotagging.tsv')
    txt = _http_get(meta_url)
    if not txt:
        return []
    rows = txt.strip().split('\n')[1:]  # skip header
    out: list[tuple[str, str]] = []
    for row in rows:
        cols = row.split('\t')
        if len(cols) < 4:
            continue
        track_id = cols[0]
        # Optional genre filter on tag column (cols[5+] are tag arrays in MTG)
        if genre_filter:
            tags = ' '.join(cols[5:]).lower()
            if not any(g.lower() in tags for g in genre_filter):
                continue
        # CDN URL pattern: jamendo.com/track/<id>/download/mp3 (96k preview)
        url = f'https://mp3l.jamendo.com/?trackid={track_id}&format=mp31'
        fname = f'jamendo_{track_id}.mp3'
        out.append((url, fname))
        if len(out) >= max_n:
            break
    return out


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
