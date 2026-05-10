"""Transition catalog reader — pure data accessor, zero behavior change.

Reads `datasets/transitions/catalog.json` and exposes lookup helpers:

    list_techniques(tier=None, category=None, status='implemented')
    get(name)              -> dict | None
    techniques_for_section(section_label)  -> list of compatible techniques
    blocked_for_section(section_label)     -> list of incompatible techniques

Catalog is reference-only — Director / planner / execute may consume any
subset, none, or extend with custom techniques. Use it when you want a
broader vocabulary than what's currently hardcoded in transition_mapping.py.

No imports of audio libs — safe to import from any path. Cached on first
load.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_CATALOG_CACHE: dict[str, Any] | None = None


def _catalog_path() -> Path:
    explicit = os.environ.get("AIJOCKEY_TRANSITION_CATALOG")
    if explicit:
        return Path(explicit)
    return Path(__file__).resolve().parent.parent / "datasets" / "transitions" / "catalog.json"


def _load() -> dict[str, Any]:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE
    p = _catalog_path()
    if not p.exists():
        _CATALOG_CACHE = {"version": 0, "techniques": []}
        return _CATALOG_CACHE
    try:
        _CATALOG_CACHE = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[transition_catalog] failed to load {p}: {e}")
        _CATALOG_CACHE = {"version": 0, "techniques": []}
    return _CATALOG_CACHE


def all_techniques() -> list[dict]:
    """Full list, regardless of status / tier."""
    return list(_load().get("techniques") or [])


def list_techniques(tier: str | None = None,
                    category: str | None = None,
                    status: str | None = "implemented") -> list[dict]:
    """Filter techniques by tier / category / implementation_status.

    status default 'implemented' — only returns ready-to-use techniques.
    Pass status=None to include 'stub' and 'planned'.
    """
    out = []
    for t in all_techniques():
        if tier is not None and t.get("tier") != tier:
            continue
        if category is not None and t.get("category") != category:
            continue
        if status is not None and t.get("implementation_status") != status:
            continue
        out.append(t)
    return out


def get(name: str) -> dict | None:
    for t in all_techniques():
        if t.get("name") == name:
            return t
    return None


def techniques_for_section(section_label: str,
                           tier: str | None = None,
                           status: str | None = "implemented") -> list[dict]:
    """Techniques whose `best_for` includes this section label and whose
    `incompatible_with` does NOT include it."""
    label = (section_label or "").lower()
    out = []
    for t in list_techniques(tier=tier, status=status):
        if label in [s.lower() for s in (t.get("best_for") or [])]:
            if label not in [s.lower() for s in (t.get("incompatible_with") or [])]:
                out.append(t)
    return out


def blocked_for_section(section_label: str) -> list[str]:
    """Techniques explicitly incompatible with this section label.
    Returns just names. Use for planner-level rejection."""
    label = (section_label or "").lower()
    return [
        t["name"] for t in all_techniques()
        if label in [s.lower() for s in (t.get("incompatible_with") or [])]
    ]


def techniques_by_tier() -> dict[str, list[str]]:
    """Pre-aggregated tier → [name, ...] map from catalog summary."""
    return dict(_load().get("tier_summary") or {})


def categories() -> dict[str, str]:
    """Category → description map."""
    return dict(_load().get("category_summary") or {})


def reload() -> None:
    """Force re-read of the catalog file (e.g. after edits)."""
    global _CATALOG_CACHE
    _CATALOG_CACHE = None


__all__ = [
    "all_techniques",
    "list_techniques",
    "get",
    "techniques_for_section",
    "blocked_for_section",
    "techniques_by_tier",
    "categories",
    "reload",
]
