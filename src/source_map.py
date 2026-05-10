"""Sidecar source-tag store for clip provenance (USER vs library).

Replaces in-place mutation of `cache/<clip_id>.json` to add a `source`
field. In-place mutation is brittle: re-analyze rebuilds the JSON from
scratch and wipes the tag. The sidecar persists across re-analysis.

File layout:
    <cache_dir>/source_map.json     # canonical sidecar
    <cache_dir>/<clip_id>.json      # untouched analysis output

Schema:
    {
      "version": 1,
      "tags": {
        "<clip_id>": {
          "source": "user" | "library",
          "tagged_at": "2026-05-10T...",
          "session_id": "<job_id or 'manual'>"
        }
      }
    }

API:
    smap = SourceMap(cache_dir)
    smap.tag(clip_id, "user", session_id="job_abc")
    smap.tag_many({cid: "library" for cid in lib_ids})
    smap.get(clip_id) -> "user" | "library" | None
    smap.is_user(clip_id) -> bool
    smap.user_clip_ids() -> list[str]
    smap.library_clip_ids() -> list[str]
    smap.flush()                 # explicit save (auto-saved on tag())

Atomic save: write tmp + rename within same dir.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


class SourceMap:
    """In-memory tag store backed by a JSON sidecar.

    Thread-safe for tag()/get(); concurrent /generate calls can share one
    instance per cache_dir without losing writes.
    """

    SCHEMA_VERSION = 1
    SIDECAR_NAME = "source_map.json"

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / self.SIDECAR_NAME
        self._lock = threading.Lock()
        self._tags: dict[str, dict] = {}
        self._load()

    # ----- I/O -----

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except Exception as e:
            print(f"[source_map] warn: malformed {self.path} ({e}); starting fresh")
            return
        if not isinstance(data, dict):
            return
        ver = data.get("version", 0)
        if ver > self.SCHEMA_VERSION:
            print(f"[source_map] warn: sidecar version {ver} newer than {self.SCHEMA_VERSION}; "
                   f"will read but may not preserve unknown fields")
        tags = data.get("tags") or {}
        if isinstance(tags, dict):
            self._tags = {
                str(k): v for k, v in tags.items() if isinstance(v, dict)
            }

    def flush(self) -> None:
        """Atomic write. Holds lock during serialize + replace."""
        with self._lock:
            payload = {"version": self.SCHEMA_VERSION, "tags": self._tags}
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, default=str))
            os.replace(tmp, self.path)

    # ----- mutation -----

    @staticmethod
    def _validate_source(source: str) -> str:
        s = (source or "").strip().lower()
        if s not in ("user", "library"):
            raise ValueError(f"source must be 'user' or 'library', got {source!r}")
        return s

    def tag(self, clip_id: str, source: str,
             session_id: str | None = None,
             flush: bool = True) -> None:
        """Tag one clip. Auto-saves unless flush=False (batch mode)."""
        s = self._validate_source(source)
        with self._lock:
            self._tags[clip_id] = {
                "source": s,
                "tagged_at": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id or "manual",
            }
        if flush:
            self.flush()

    def tag_many(self, mapping: dict[str, str],
                  session_id: str | None = None) -> None:
        """Bulk-tag without per-call flush. One disk write at end."""
        for cid, src in mapping.items():
            self.tag(cid, src, session_id=session_id, flush=False)
        self.flush()

    def untag(self, clip_id: str, flush: bool = True) -> bool:
        """Remove a tag. Returns True if removed, False if not present."""
        with self._lock:
            removed = self._tags.pop(clip_id, None) is not None
        if removed and flush:
            self.flush()
        return removed

    # ----- queries -----

    def get(self, clip_id: str) -> str | None:
        with self._lock:
            entry = self._tags.get(clip_id)
        return entry["source"] if entry else None

    def is_user(self, clip_id: str) -> bool:
        return self.get(clip_id) == "user"

    def is_library(self, clip_id: str) -> bool:
        return self.get(clip_id) == "library"

    def user_clip_ids(self) -> list[str]:
        with self._lock:
            return [k for k, v in self._tags.items() if v.get("source") == "user"]

    def library_clip_ids(self) -> list[str]:
        with self._lock:
            return [k for k, v in self._tags.items() if v.get("source") == "library"]

    def all_tagged(self) -> list[str]:
        with self._lock:
            return list(self._tags.keys())

    # ----- migration helper -----

    def migrate_from_inplace(self, clip_ids: Iterable[str],
                              session_id: str | None = None) -> int:
        """One-shot migration: read `source` field from each cache JSON
        and copy into the sidecar. Returns count migrated.

        Safe to run repeatedly — already-tagged clips skipped unless
        the in-place value disagrees, in which case sidecar wins
        (since sidecar is the new source of truth).

        After migration, downstream code should use SourceMap.get() and
        ignore the in-place `source` field on cache JSONs.
        """
        migrated = 0
        for cid in clip_ids:
            cache_path = self.cache_dir / f"{cid}.json"
            if not cache_path.exists():
                continue
            try:
                meta = json.loads(cache_path.read_text())
            except Exception:
                continue
            inplace = meta.get("source")
            if not inplace:
                continue
            if cid in self._tags:
                continue  # already migrated
            try:
                self.tag(cid, inplace, session_id=session_id, flush=False)
                migrated += 1
            except ValueError:
                pass  # invalid source value, skip
        if migrated:
            self.flush()
        return migrated


__all__ = ["SourceMap"]
