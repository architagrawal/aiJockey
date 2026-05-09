"""Minimal structured job logging (JSON lines to stdout)."""

from __future__ import annotations

import json
import time
from typing import Any


def jlog(job_id: str, stage: str, ms: float | None = None, **fields: Any) -> None:
    rec: dict[str, Any] = {"ts": time.time(), "job_id": job_id, "stage": stage}
    if ms is not None:
        rec["ms"] = round(ms, 2)
    rec.update(fields)
    print(json.dumps(rec, default=str), flush=True)
