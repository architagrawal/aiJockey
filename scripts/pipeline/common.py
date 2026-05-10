"""Shared primitives for pipeline-parallel stages.

File-based queues, atomic writes, idempotent processing, watch loops.
Reference: docs/phase1_plan.md §15.
"""
from __future__ import annotations
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Callable

SCRATCH = Path(os.environ.get('AIJOCKEY_SCRATCH', '/scratch'))


def scratch_dir(*parts: str) -> Path:
    p = SCRATCH.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def atomic_write(target: Path, mode: str = 'w'):
    """Write to *.tmp then rename — readers never see partial files."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.tmp')
    f = open(tmp, mode)
    try:
        yield f
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
        f.close()
        os.replace(tmp, target)
    except Exception:
        f.close()
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, obj) -> None:
    with atomic_write(path) as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def watch(directory: Path, glob_pat: str = '*',
          interval_sec: float = 30.0,
          done_marker: Callable[[Path], Path] | None = None,
          ) -> Iterator[Path]:
    """Yield new files matching glob in directory.

    `done_marker` is a function path -> path-of-done-marker. If marker
    exists, the file is considered already processed and skipped. The
    consumer is responsible for writing the marker after processing.
    """
    seen: set[str] = set()
    while True:
        directory.mkdir(parents=True, exist_ok=True)
        for fp in sorted(directory.glob(glob_pat)):
            key = str(fp)
            if key in seen:
                continue
            if done_marker is not None:
                marker = done_marker(fp)
                if marker.exists():
                    seen.add(key)
                    continue
            yield fp
            seen.add(key)
        time.sleep(interval_sec)


def queue_count(directory: Path, glob_pat: str = '*') -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob(glob_pat))


__all__ = [
    'SCRATCH', 'scratch_dir', 'atomic_write', 'write_json', 'read_json',
    'watch', 'queue_count',
]
