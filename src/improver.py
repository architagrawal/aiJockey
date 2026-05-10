"""Probe-driven timeline improver.

Reactive quality loop: render → probe → diagnose → mutate timeline → re-render
the affected segments only.

Three artifact types map to three deterministic actions:

  rms_envelope_mismatch   →  inject fade bars OR mark segment for re-pick
                             at lower/higher target energy
  vocal_bleed_xcorr       →  swap transition to a vocal-suppressing variant
                             (drum_break / filter_fade / eq_swap with longer
                             stem-additive overlap)
  spectral_phasing        →  shorten overlap (halve bars) OR swap to cut
                             transition at the junction

Returns a `TimelineEdit` list — caller applies edits to the timeline JSON
then re-runs execute on affected segments. Cascade rule: changing entry N
invalidates transitions N-1→N AND N→N+1.

This is the foundation for the probe → DPO preference loop: every applied
edit becomes a (before, after, probe_delta) triple for S7 training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Tunable thresholds. Tighter than probe defaults so we only act on
# clearly-audible artifacts, not borderline cases.
_RMS_DB_THRESHOLD = 6.0          # dB step that triggers energy action
_VOCAL_BLEED_THRESHOLD = 0.45    # xcorr that triggers vocal-suppress action
_PHASE_THRESHOLD = 0.25          # phase delta that triggers overlap action
_SEVERITY_FLOOR = 0.55           # any single-axis severity below this = ignore


# Transitions that suppress vocals during overlap (use these when bleed flagged).
_VOCAL_SUPPRESS_TRANSITIONS = (
    'drum_break',     # incoming muted to drums-only during overlap
    'filter_fade',    # lowpass close on outgoing, mids only carry voice
    'eq_swap',        # bass + treble swap masks vocal mid
)


@dataclass
class TimelineEdit:
    """One mutation to apply to a timeline entry."""
    junction_index: int          # which transition (entry index)
    action: str                  # 'shorten_overlap' | 'swap_transition' |
                                 # 'energy_repick' | 'inject_bridge'
    rationale: str               # human-readable why
    field_changes: dict[str, Any] = field(default_factory=dict)
    requires_replan: bool = False  # True = caller should re-run planner pick


@dataclass
class ImproverReport:
    edits: list[TimelineEdit]
    severity_before: float
    issues_addressed: int
    issues_skipped: int

    def to_dict(self) -> dict:
        return {
            'severity_before': self.severity_before,
            'issues_addressed': self.issues_addressed,
            'issues_skipped': self.issues_skipped,
            'edits': [
                {'junction_index': e.junction_index,
                 'action': e.action,
                 'rationale': e.rationale,
                 'field_changes': e.field_changes,
                 'requires_replan': e.requires_replan}
                for e in self.edits
            ],
        }


def diagnose_junction(probe_row: dict) -> list[tuple[str, str]]:
    """Return [(action, rationale), ...] for a single probe junction.

    Empty list = no actionable issue. Multiple issues at one junction
    are returned in priority order (most-fixable first).
    """
    out: list[tuple[str, str]] = []
    rms = probe_row.get('rms') or {}
    bleed = probe_row.get('vocal_bleed') or {}
    phase = probe_row.get('phasing') or {}

    db_diff = float(rms.get('db_diff', 0.0))
    rms_sev = float(rms.get('severity', 0.0))
    bleed_xc = float(bleed.get('xcorr_max', 0.0))
    bleed_sev = float(bleed.get('severity', 0.0))
    phase_d = float(phase.get('phase_delta', 0.0))
    phase_sev = float(phase.get('severity', 0.0))

    # Vocal bleed first — easiest to fix (just swap transition technique).
    if bleed_sev >= _SEVERITY_FLOOR and abs(bleed_xc) >= _VOCAL_BLEED_THRESHOLD:
        out.append(('swap_transition',
                    f'vocal_bleed xcorr={bleed_xc:+.2f} → vocal-suppress technique'))

    # Phase cancellation — shorten overlap to reduce constructive collision window.
    if phase_sev >= _SEVERITY_FLOOR and phase_d >= _PHASE_THRESHOLD:
        out.append(('shorten_overlap',
                    f'phase_delta={phase_d:.2f} → halve overlap bars'))

    # Energy mismatch — biggest hammer, requires planner re-pick.
    if rms_sev >= _SEVERITY_FLOOR and abs(db_diff) >= _RMS_DB_THRESHOLD:
        direction = 'higher' if db_diff > 0 else 'lower'
        out.append(('energy_repick',
                    f'db_diff={db_diff:+.2f} → re-pick segment with {direction} starting energy'))

    return out


def improve_timeline(timeline: list[dict], probe_result: dict) -> ImproverReport:
    """Top-level entry: scan probes, emit edits.

    Mutates nothing — returns edits for caller to apply. Caller decides
    whether to apply (e.g. only on first re-render attempt to avoid
    infinite loops).
    """
    edits: list[TimelineEdit] = []
    skipped = 0
    for jrow in probe_result.get('junctions') or []:
        ji = int(jrow.get('junction_index', -1))
        if ji < 0 or ji >= len(timeline):
            continue
        # Junction `ji` corresponds to transition INTO timeline[ji+1] in some
        # schemas, INTO timeline[ji] in others. Probes count junction 0 as
        # between segment 0 and 1 → entry index ji+1 holds the transition.
        entry_idx = ji + 1
        if entry_idx >= len(timeline):
            continue
        entry = timeline[entry_idx]
        actions = diagnose_junction(jrow)
        if not actions:
            continue
        for action, rationale in actions:
            ed = _build_edit(action, rationale, entry_idx, entry, jrow)
            if ed is not None:
                edits.append(ed)
            else:
                skipped += 1

    return ImproverReport(
        edits=edits,
        severity_before=float(probe_result.get('overall_severity', 0.0)),
        issues_addressed=len(edits),
        issues_skipped=skipped,
    )


def _build_edit(action: str, rationale: str, entry_idx: int,
                entry: dict, jrow: dict) -> TimelineEdit | None:
    tin = entry.get('transition_in') or {}
    cur_name = tin.get('name', 'crossfade')
    cur_bars = int(tin.get('bars', 16))

    if action == 'shorten_overlap':
        new_bars = max(2, cur_bars // 2)
        if new_bars >= cur_bars:
            return None
        return TimelineEdit(
            junction_index=entry_idx, action=action, rationale=rationale,
            field_changes={'transition_in.bars': new_bars,
                           'transition_in._improver_orig_bars': cur_bars},
            requires_replan=False,
        )

    if action == 'swap_transition':
        # Pick first vocal-suppress technique that isn't current.
        new_name = next((n for n in _VOCAL_SUPPRESS_TRANSITIONS
                         if n != cur_name), 'eq_swap')
        if new_name == cur_name:
            return None
        return TimelineEdit(
            junction_index=entry_idx, action=action, rationale=rationale,
            field_changes={'transition_in.name': new_name,
                           'transition_in._improver_orig_name': cur_name},
            requires_replan=False,
        )

    if action == 'energy_repick':
        # Mark segment for re-pick. Real planner integration writes a constraint
        # (target_energy_lt / _gt) and re-runs pick_segment. For now flag for
        # caller to handle.
        return TimelineEdit(
            junction_index=entry_idx, action=action, rationale=rationale,
            field_changes={'_improver_repick': True,
                           '_improver_db_diff': float(
                               (jrow.get('rms') or {}).get('db_diff', 0.0))},
            requires_replan=True,
        )

    return None


def apply_edits(timeline: list[dict],
                edits: list[TimelineEdit]) -> list[dict]:
    """Apply non-replan edits in-place. Replan-required edits are left as
    metadata for caller to handle separately.

    Returns the list of segment indices that were touched (for cascade
    re-render: indices N-1, N, N+1 all need re-render when N changes).
    """
    touched: set[int] = set()
    for ed in edits:
        idx = ed.junction_index
        if idx >= len(timeline):
            continue
        for path, val in ed.field_changes.items():
            parts = path.split('.')
            target = timeline[idx]
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = val
        touched.add(idx)
        if idx > 0:
            touched.add(idx - 1)
        if idx + 1 < len(timeline):
            touched.add(idx + 1)
    return sorted(touched)


def main() -> None:
    """CLI: improver --probe probe.json --timeline tl.json [--apply --out edited.json]"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', required=True, help='probe.json from audio_probes')
    ap.add_argument('--timeline', required=True)
    ap.add_argument('--apply', action='store_true',
                    help='write mutated timeline to --out')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    probe = json.load(open(args.probe))
    tl_blob = json.load(open(args.timeline))
    tl = tl_blob.get('timeline') if isinstance(tl_blob, dict) else tl_blob
    report = improve_timeline(tl, probe)
    print(json.dumps(report.to_dict(), indent=2))
    if args.apply:
        touched = apply_edits(tl, report.edits)
        out_path = args.out or args.timeline.replace('.json', '.improved.json')
        if isinstance(tl_blob, dict):
            tl_blob['timeline'] = tl
            tl_blob.setdefault('meta', {})['improver_touched'] = touched
            json.dump(tl_blob, open(out_path, 'w'), indent=2)
        else:
            json.dump(tl, open(out_path, 'w'), indent=2)
        print(f'\n→ wrote {out_path} (touched segments: {touched})')


if __name__ == '__main__':
    main()
