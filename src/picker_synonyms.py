"""Section-label synonym + transition prereq tables for picker layer.

Two related lookup tables:

1. SECTION_SYNONYMS — maps All-In-One canonical labels (verse/chorus/break/
   bridge/inst/solo/intro/outro) to legacy/alternate names that
   pick_segment may receive from older analyzers (drop/peak/hook/big/
   breakdown/etc). Bidirectional: query either direction returns the
   canonical bucket.

2. TRANSITION_REQUIRES — what a transition technique needs from the
   PICKED section: vocal-heavy / instrumental-only / drums-only / break-
   like / drop-like. Read by candidate_picker before scoring so we don't
   pick a vocal chorus for an `acapella_drop` (wants outgoing vocals
   alone) on the WRONG side, etc.

Pure data + tiny lookup helpers. No audio imports.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Section synonyms — canonical bucket → list of equivalent label strings.
# Lowercased input matched against any value; returns the canonical key.
# ---------------------------------------------------------------------------

SECTION_SYNONYMS: dict[str, list[str]] = {
    # Energetic peak — chorus / hook / drop / climax
    "chorus":   ["chorus", "drop", "peak", "hook", "big", "climax", "anthem"],
    # Verse — standard vocal-on-instrumental section
    "verse":    ["verse", "main", "body", "groove", "stanza"],
    # Break / breakdown — energy reduction, often instrumental-only
    "break":    ["break", "breakdown", "quiet", "drop_down", "stripped",
                  "fill", "interlude_short"],
    # Bridge — transitional section between verse/chorus
    "bridge":   ["bridge", "transition_section", "middle8", "interlude"],
    # Pure instrumental section
    "inst":     ["inst", "instrumental", "no_vocal", "music", "groove_only"],
    # Solo section (lead instrument carries)
    "solo":     ["solo", "lead", "feature"],
    # Intro / outro
    "intro":    ["intro", "build", "opener", "ramp_up", "start", "head"],
    "outro":    ["outro", "ending", "wind_down", "tail", "fade_out_section"],
    # Drop section (specifically electronic-music drop, not pop chorus)
    "drop":     ["drop", "drop_section", "energy_peak", "explosion"],
    # Pre-drop (build into drop)
    "pre_drop": ["pre_drop", "buildup", "tension", "ramp"],
}


# Reverse lookup table built once.
_LABEL_TO_CANONICAL: dict[str, str] = {}
for canonical, syns in SECTION_SYNONYMS.items():
    for s in syns:
        _LABEL_TO_CANONICAL[s.lower()] = canonical


def canonical_section(label: str | None) -> str:
    """Return canonical bucket for a label, or '?' if unknown.

    Case-insensitive. Strips whitespace.
    """
    if not label:
        return "?"
    return _LABEL_TO_CANONICAL.get(label.lower().strip(), "?")


def matches_canonical(label: str | None, canonical: str) -> bool:
    """True iff `label` is a synonym of canonical."""
    return canonical_section(label) == canonical


def expand(canonical: str) -> list[str]:
    """All synonyms for a canonical bucket. Empty list if unknown bucket."""
    return list(SECTION_SYNONYMS.get(canonical) or [])


# ---------------------------------------------------------------------------
# Transition prerequisites — what each technique needs from picked sections.
# Each entry has up to two keys: 'prev_requires' / 'cur_requires'. Each is a
# spec dict with optional fields:
#   sections: list[str]          — canonical labels acceptable
#   max_vocal_activity: float    — 0..1; reject if section.va > this
#   min_vocal_activity: float    — reject if section.va < this
#   needs_drums: bool            — section must have drum stem present
# When entry missing → no constraint (any section ok).
# ---------------------------------------------------------------------------

TRANSITION_REQUIRES: dict[str, dict] = {
    # Aggressive techniques want instrumental on BOTH sides
    "drum_break": {
        "prev_requires": {"sections": ["break", "bridge", "inst"],
                           "max_vocal_activity": 0.20, "needs_drums": True},
        "cur_requires":  {"max_vocal_activity": 0.30},
    },
    "drum_replace": {
        "prev_requires": {"max_vocal_activity": 0.25, "needs_drums": True},
        "cur_requires":  {"max_vocal_activity": 0.25, "needs_drums": True},
    },
    "kickless_swap": {
        "prev_requires": {"max_vocal_activity": 0.30, "needs_drums": True},
        "cur_requires":  {"max_vocal_activity": 0.30, "needs_drums": True},
    },
    "chop": {
        "prev_requires": {"max_vocal_activity": 0.20},
        "cur_requires":  {"max_vocal_activity": 0.20},
    },
    "loop_tighten": {
        "prev_requires": {"max_vocal_activity": 0.30},
    },
    "loop_roll": {
        "prev_requires": {"max_vocal_activity": 0.30},
    },
    "beat_juggle": {
        "prev_requires": {"max_vocal_activity": 0.20},
        "cur_requires":  {"max_vocal_activity": 0.20},
    },
    "scratch_fill": {
        "prev_requires": {"max_vocal_activity": 0.30},
    },
    "spinback": {
        "prev_requires": {"max_vocal_activity": 0.40, "sections": ["chorus", "drop", "outro"]},
    },
    "forward_spin": {
        "prev_requires": {"max_vocal_activity": 0.40, "sections": ["chorus", "drop"]},
    },
    "tape_stop": {
        "prev_requires": {"max_vocal_activity": 0.30},
    },
    "pitch_bend": {
        "prev_requires": {"sections": ["bridge", "inst", "break"],
                           "max_vocal_activity": 0.20},
    },
    "bpm_warp": {
        "prev_requires": {"sections": ["bridge", "verse", "inst"],
                           "max_vocal_activity": 0.30},
    },
    "spectral_hold": {
        "prev_requires": {"sections": ["bridge", "break"],
                           "max_vocal_activity": 0.30},
    },

    # Drop-tier needs drop-compatible sections on BOTH sides
    "build_riser_drop": {
        "prev_requires": {"sections": ["chorus", "drop", "pre_drop", "solo"]},
        "cur_requires":  {"sections": ["drop", "chorus", "pre_drop"]},
    },
    "silence_drop": {
        "cur_requires": {"sections": ["drop", "chorus", "pre_drop"]},
    },
    "snare_buildup": {
        "prev_requires": {"sections": ["chorus", "drop", "pre_drop", "verse"]},
        "cur_requires":  {"sections": ["drop", "chorus"]},
    },
    "breakdown_to_drop": {
        "prev_requires": {"sections": ["break", "bridge"]},
        "cur_requires":  {"sections": ["drop", "chorus"]},
    },

    # Stem-based — vocals required on the appropriate side
    "acapella_drop": {
        "prev_requires": {"min_vocal_activity": 0.30, "sections": ["chorus", "verse"]},
        "cur_requires":  {"sections": ["drop", "chorus"]},
    },
    "mashup": {
        "prev_requires": {"max_vocal_activity": 0.20},          # backing — instrumental side
        "cur_requires":  {"min_vocal_activity": 0.30},           # vocal source
    },
    "instrumental_swap": {
        "prev_requires": {"min_vocal_activity": 0.30},           # vocal carry
        "cur_requires":  {"max_vocal_activity": 0.30, "needs_drums": True},
    },
    "stem_swap": {
        # Both sides need stems. No vocal constraint — runs additively.
    },
    "echo_out": {
        "prev_requires": {"sections": ["outro", "bridge", "verse", "inst"]},
    },
    "reverb_wash": {
        "prev_requires": {"sections": ["outro", "bridge", "break"]},
    },
}


def requires_for(transition_name: str) -> dict:
    """Return requires dict for a technique, empty if no constraints."""
    return dict(TRANSITION_REQUIRES.get(transition_name) or {})


def section_satisfies(section: dict, spec: dict) -> bool:
    """Return True iff section meets all constraints in spec.

    section: dict with optional fields {label, type, vocal_activity,
              has_drums (bool inferred), energy}.
    spec: optional fields per TRANSITION_REQUIRES schema.
    """
    if not spec:
        return True

    label = canonical_section(section.get("label", section.get("type")))
    sec_filter = spec.get("sections")
    if sec_filter and label not in sec_filter:
        # Allow if the raw label string IS in the spec (for non-canonical labels)
        raw = (section.get("label") or section.get("type") or "").lower()
        if raw not in sec_filter:
            return False

    va = section.get("vocal_activity")
    if isinstance(va, (int, float)):
        max_va = spec.get("max_vocal_activity")
        if max_va is not None and va > float(max_va):
            return False
        min_va = spec.get("min_vocal_activity")
        if min_va is not None and va < float(min_va):
            return False

    if spec.get("needs_drums"):
        # Heuristic: section without explicit `has_drums=False` assumed to have drums
        if section.get("has_drums") is False:
            return False

    return True


__all__ = [
    "SECTION_SYNONYMS",
    "TRANSITION_REQUIRES",
    "canonical_section",
    "matches_canonical",
    "expand",
    "requires_for",
    "section_satisfies",
]
