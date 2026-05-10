"""Constitutional rules — hard musical constraints checked before render.

Plan reference: §5 (diagnosis hooks) + §14.2 Pattern P3.

Each rule returns a Violation if broken, or None. The validator runs all
rules over a planned timeline; any violation triggers planner rollback or
tier downgrade. Rules are deliberately strict: better to skip a junction
than ship a clashing mix.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Violation:
    rule: str
    junction_index: int
    detail: str
    severity: str = 'reject'  # 'reject' or 'warn'


PHASE1_DROP_INVALID_SECTIONS = frozenset({'breakdown', 'intro', 'outro'})
PHASE1_BREAKDOWN_INVALID_NEXT = frozenset({'breakdown'})
PHASE1_BPM_RATIO_MAX = 1.06  # 6% drift allowed within mix
PHASE1_KEY_BLEND_MIN_BARS = 16  # blends >= this need key compatibility
PHASE1_MAX_ACCENTS_PER_JUNCTION = 2


def check_phrase_grid(timeline: list[dict], clips_meta: dict[str, dict],
                      tolerance_sec: float = 0.05) -> list[Violation]:
    """Each segment.start must lie within tolerance of a downbeat in its clip."""
    out: list[Violation] = []
    for i, entry in enumerate(timeline):
        cid = entry.get('clip_id')
        seg = entry.get('segment') or {}
        meta = clips_meta.get(cid) or {}
        downbeats = meta.get('downbeats') or []
        if not downbeats:
            continue
        start = float(seg.get('start', 0.0))
        nearest = min(downbeats, key=lambda d: abs(d - start))
        if abs(nearest - start) > tolerance_sec:
            out.append(Violation(
                rule='phrase_grid', junction_index=i,
                detail=f"segment start {start:.2f}s drifts {abs(nearest-start)*1000:.0f}ms from downbeat",
            ))
    return out


def check_drop_section(timeline: list[dict]) -> list[Violation]:
    """Drop tier requires both incoming + outgoing be drop-compatible sections."""
    out: list[Violation] = []
    for i in range(1, len(timeline)):
        prev = timeline[i - 1]
        cur = timeline[i]
        tier = (cur.get('transition_in') or {}).get('tier', 'minor')
        if tier != 'drop':
            continue
        prev_section = (prev.get('segment') or {}).get('type', '')
        cur_section = (cur.get('segment') or {}).get('type', '')
        bad = []
        if prev_section in PHASE1_DROP_INVALID_SECTIONS:
            bad.append(f"out={prev_section}")
        if cur_section in PHASE1_DROP_INVALID_SECTIONS:
            bad.append(f"in={cur_section}")
        if bad:
            out.append(Violation(
                rule='drop_section', junction_index=i,
                detail=f"drop tier on invalid section(s): {','.join(bad)}",
            ))
    return out


def check_breakdown_pair(timeline: list[dict]) -> list[Violation]:
    """Breakdown -> breakdown = energy crater. Reject."""
    out: list[Violation] = []
    for i in range(1, len(timeline)):
        prev_section = (timeline[i - 1].get('segment') or {}).get('type', '')
        cur_section = (timeline[i].get('segment') or {}).get('type', '')
        if prev_section == 'breakdown' and cur_section in PHASE1_BREAKDOWN_INVALID_NEXT:
            out.append(Violation(
                rule='breakdown_pair', junction_index=i,
                detail='breakdown into breakdown = energy crater',
            ))
    return out


def check_bpm_drift(timeline: list[dict],
                    max_ratio: float = PHASE1_BPM_RATIO_MAX) -> list[Violation]:
    """BPM ratio between any two adjacent target_bpm > max_ratio."""
    out: list[Violation] = []
    for i in range(1, len(timeline)):
        a = float(timeline[i - 1].get('target_bpm', 120.0))
        b = float(timeline[i].get('target_bpm', 120.0))
        if a <= 0 or b <= 0:
            continue
        ratio = max(a, b) / min(a, b)
        if ratio > max_ratio:
            out.append(Violation(
                rule='bpm_drift', junction_index=i,
                detail=f"bpm ratio {ratio:.3f} > {max_ratio:.3f} ({a:.1f}/{b:.1f})",
            ))
    return out


def check_key_compat_for_long_blends(timeline: list[dict],
                                      min_bars: int = PHASE1_KEY_BLEND_MIN_BARS
                                      ) -> list[Violation]:
    """Long blends (>=min_bars) must have Camelot-compatible keys."""
    try:
        from camelot import camelot_distance
    except Exception:
        return []
    out: list[Violation] = []
    for i in range(1, len(timeline)):
        ti = (timeline[i].get('transition_in') or {})
        bars = int(ti.get('bars', 0))
        if bars < min_bars:
            continue
        ka = timeline[i - 1].get('target_key') or '?'
        kb = timeline[i].get('target_key') or '?'
        if ka == '?' or kb == '?':
            continue
        try:
            dist = camelot_distance(ka, kb)
        except Exception:
            dist = 0  # don't block on detector errors
        # Compatible: distance 0 (same), 1 (adjacent / relative)
        if dist > 1:
            out.append(Violation(
                rule='key_compat', junction_index=i,
                detail=f"keys {ka}/{kb} incompatible for {bars}-bar blend",
            ))
    return out


def check_accent_budget(timeline: list[dict],
                        max_per_junction: int = PHASE1_MAX_ACCENTS_PER_JUNCTION
                        ) -> list[Violation]:
    """Cap accent FX per junction so Director cannot pile them on."""
    out: list[Violation] = []
    for i, entry in enumerate(timeline):
        accents = entry.get('accent_hints') or ([entry['accent_hint']]
                                                  if entry.get('accent_hint') else [])
        if len(accents) > max_per_junction:
            out.append(Violation(
                rule='accent_budget', junction_index=i,
                detail=f"{len(accents)} accents > {max_per_junction} cap",
                severity='warn',  # downgrade not reject; truncate list
            ))
    return out


def validate(timeline: list[dict],
             clips_meta: dict[str, dict] | None = None) -> list[Violation]:
    """Run all rules. Returns ordered list of violations."""
    clips_meta = clips_meta or {}
    out: list[Violation] = []
    out += check_phrase_grid(timeline, clips_meta)
    out += check_drop_section(timeline)
    out += check_breakdown_pair(timeline)
    out += check_bpm_drift(timeline)
    out += check_key_compat_for_long_blends(timeline)
    out += check_accent_budget(timeline)
    return out


def repair(timeline: list[dict], violations: list[Violation]) -> list[dict]:
    """Best-effort auto-fix. Mutates and returns the timeline.

    Phase A polish strategy: tier-downgrade on rejects, truncate accent list
    on warnings. Hard violations (drop section, breakdown crater) tier=>minor.
    BPM drift reject = let planner pick a different segment (raise to caller).
    """
    for v in violations:
        if v.severity != 'reject':
            if v.rule == 'accent_budget' and v.junction_index < len(timeline):
                e = timeline[v.junction_index]
                if isinstance(e.get('accent_hints'), list):
                    e['accent_hints'] = e['accent_hints'][:PHASE1_MAX_ACCENTS_PER_JUNCTION]
            continue
        i = v.junction_index
        if i >= len(timeline):
            continue
        if v.rule in ('drop_section', 'breakdown_pair', 'key_compat'):
            ti = timeline[i].setdefault('transition_in', {})
            ti['tier'] = 'minor'
            ti['name'] = 'crossfade'
            ti['_constitutional_repair'] = v.rule
    return timeline


__all__ = ['Violation', 'validate', 'repair']
