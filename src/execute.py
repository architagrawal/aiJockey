"""
Render timeline.json -> raw_mix.wav.

For each entry: load stems from cache, slice segment, time-stretch + pitch-shift
to target BPM/key, then apply transition_in technique against previous entry.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import torch
import torchaudio
import pyrubberband as pyrb

import transitions as T
from camelot import semitones_between
from phrase import snap_to_phrase
from samples import SampleBank

SR = 44100


# ---------------------------------------------------------------------------
# Phrase quantization
# ---------------------------------------------------------------------------

def _phrase_bars_for_clip(meta: dict) -> int:
    """Use detected phrase length if stored, default 8 (Phase A polish).

    Phase A polish snaps to 8-bar boundaries even on 16-bar clips because
    8-bar resolution gives enough valid junctions while keeping segment
    selection flexible.
    """
    return int(meta.get('phrase_bars', 8))


def quantize_timeline_to_phrase(timeline: list[dict],
                                clips_meta: dict[str, dict],
                                max_drift_bars: int = 2) -> list[dict]:
    """Snap each segment's start/end to nearest phrase boundary in clip's
    downbeats grid. Mutates entries in place and returns same list.

    max_drift_bars: hard limit on how far snap may move a boundary. If the
    nearest phrase boundary is farther than this, the original time is kept
    (better unquantized than wrong-section).
    """
    for entry in timeline:
        cid = entry.get('clip_id')
        seg = entry.get('segment') or {}
        meta = clips_meta.get(cid) or {}
        downbeats = meta.get('downbeats') or []
        if not downbeats or 'start' not in seg or 'end' not in seg:
            continue
        bpp = _phrase_bars_for_clip(meta)
        bpm = float(meta.get('tempo', 120.0)) or 120.0
        bar_dur = 4.0 * 60.0 / bpm
        max_drift_sec = max_drift_bars * bar_dur

        orig_start = float(seg['start'])
        orig_end = float(seg['end'])
        snap_start = snap_to_phrase(orig_start, downbeats, bars_per_phrase=bpp)
        snap_end = snap_to_phrase(orig_end, downbeats, bars_per_phrase=bpp)

        if abs(snap_start - orig_start) <= max_drift_sec:
            seg['start'] = snap_start
        if abs(snap_end - orig_end) <= max_drift_sec and snap_end > seg['start']:
            seg['end'] = snap_end
        seg['phrase_quantized'] = (
            seg['start'] != orig_start or seg['end'] != orig_end
        )
    return timeline


# ---------------------------------------------------------------------------
# Stem loading
# ---------------------------------------------------------------------------

def load_stems(clip_id: str, cache_dir: str = 'cache') -> dict[str, np.ndarray]:
    base = Path(cache_dir) / 'stems' / clip_id
    out: dict[str, np.ndarray] = {}
    for name in ('drums', 'bass', 'other', 'vocals'):
        fp = base / f'{name}.wav'
        if not fp.exists():
            continue
        wav, sr = torchaudio.load(str(fp))
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.size(0) == 1:
            wav = wav.repeat(2, 1)
        out[name] = wav.numpy().astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Time-stretch + pitch shift
# ---------------------------------------------------------------------------

def stretch_and_pitch(wav: np.ndarray, sr: int, src_bpm: float,
                      dst_bpm: float, semitones: float,
                      max_bpm_ratio: float | None = None) -> np.ndarray:
    """rubberband expects (T, channels). wav is (channels, T)."""
    x = wav.T.astype(np.float32)
    dst = float(dst_bpm)
    if src_bpm > 0 and max_bpm_ratio and max_bpm_ratio > 1.0:
        lo = src_bpm / max_bpm_ratio
        hi = src_bpm * max_bpm_ratio
        dst = max(lo, min(hi, dst))
    if src_bpm > 0 and abs(src_bpm - dst) > 0.01:
        try:
            x = pyrb.time_stretch(x, sr, dst / src_bpm)
        except Exception as e:
            print(f"warn: time_stretch failed ({e}), using original")
    if abs(semitones) > 0.01:
        try:
            x = pyrb.pitch_shift(x, sr, semitones)
        except Exception as e:
            print(f"warn: pitch_shift failed ({e}), using original")
    return x.T.astype(np.float32)


# ---------------------------------------------------------------------------
# Render one segment (return mix + per-stem)
# ---------------------------------------------------------------------------

def render_segment(entry: dict, clips_meta: dict[str, dict],
                   cache_dir: str = 'cache',
                   max_bpm_ratio: float | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cid = entry['clip_id']
    seg = entry['segment']
    target_bpm = float(entry['target_bpm'])
    target_key = entry['target_key']
    meta = clips_meta[cid]
    src_bpm = float(meta.get('tempo', target_bpm))
    src_key = meta.get('key', '?')
    semitones = semitones_between(src_key, target_key)

    stems = load_stems(cid, cache_dir)
    if not stems:
        raise FileNotFoundError(f"no stems for clip {cid} in {cache_dir}/stems/{cid}/")
    s_start = max(0, int(seg['start'] * SR))
    s_end = int(seg['end'] * SR)
    sliced = {n: s[:, s_start:s_end] for n, s in stems.items()}
    processed = {n: stretch_and_pitch(s, SR, src_bpm, target_bpm, semitones,
                                        max_bpm_ratio=max_bpm_ratio)
                 for n, s in sliced.items()}
    # Equalize lengths (rubberband can produce slightly different sample counts)
    min_len = min(s.shape[1] for s in processed.values())
    processed = {n: s[:, :min_len] for n, s in processed.items()}
    # Instrumental-only mode (Phase 1 default): drop vocals stem from full mix
    # everywhere, not just overlap. Stems dict still holds vocals so accent
    # logic / future re-injection paths can reference them.
    import os
    inst_only = os.environ.get('AIJOCKEY_INSTRUMENTAL_ONLY', '0') == '1'
    if inst_only:
        full = sum(s for n, s in processed.items() if n != 'vocals').astype(np.float32)
    else:
        full = sum(processed.values()).astype(np.float32)
    return full, processed


# ---------------------------------------------------------------------------
# Apply transition between two rendered segments
# ---------------------------------------------------------------------------

def _stem_sum(stems: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    """Additively sum the named stems. Returns None if none present."""
    arrs = [stems[n] for n in names if n in stems]
    if not arrs:
        return None
    target_len = min(a.shape[1] for a in arrs)
    return sum(a[:, :target_len] for a in arrs).astype(np.float32)


def _suppress_intro_vocals_stemwise(cur: dict, n_samples: int,
                                     ramp_in_samples: int = 0) -> np.ndarray:
    """Stem-additive intro mute. Reconstructs cur's overlap region from
    drums+bass+other stems only — never computes mix-minus-vocals, so no
    subtraction phase residue. Vocals ramp in after overlap.
    """
    full = cur['full']
    stems = cur.get('stems') or {}
    vox = stems.get('vocals')
    if vox is None or n_samples <= 0:
        return full
    inst = _stem_sum(stems, ('drums', 'bass', 'other'))
    if inst is None:
        return full
    out = full.copy()
    n_samples = int(min(n_samples, full.shape[1], inst.shape[1]))
    out[:, :n_samples] = inst[:, :n_samples]
    if ramp_in_samples > 0:
        end = min(out.shape[1], n_samples + ramp_in_samples, inst.shape[1])
        ramp_n = end - n_samples
        if ramp_n > 0:
            ramp = np.linspace(0.0, 1.0, ramp_n, dtype=np.float32)
            inst_seg = inst[:, n_samples:end]
            vox_seg = vox[:, n_samples:end] if vox.shape[1] >= end else None
            if vox_seg is not None:
                out[:, n_samples:end] = inst_seg + vox_seg * ramp
            else:
                out[:, n_samples:end] = inst_seg
    return out


def _suppress_outro_vocals_stemwise(output: np.ndarray, prev: dict,
                                     n_samples: int,
                                     ramp_out_samples: int = 0) -> np.ndarray:
    """Stem-additive outro mute. Replaces the last n_samples of output with
    prev's instrumental stems (drums+bass+other), with a pre-overlap ramp-out
    of vocals via stem replacement.
    """
    stems = prev.get('stems') or {}
    vox = stems.get('vocals')
    inst = _stem_sum(stems, ('drums', 'bass', 'other'))
    if vox is None or inst is None or n_samples <= 0 or output.shape[1] == 0:
        return output
    out = output.copy()
    n_samples = int(min(n_samples, out.shape[1], inst.shape[1]))
    overlap_start = out.shape[1] - n_samples
    inst_overlap = inst[:, -n_samples:]
    out[:, overlap_start:] = inst_overlap
    if ramp_out_samples > 0:
        ramp_n = min(ramp_out_samples, overlap_start, inst.shape[1] - n_samples)
        if ramp_n > 0:
            ramp = np.linspace(1.0, 0.0, ramp_n, dtype=np.float32)
            ramp_start = overlap_start - ramp_n
            inst_pre = inst[:, -(n_samples + ramp_n):-n_samples]
            vox_pre = vox[:, -(n_samples + ramp_n):-n_samples]
            out[:, ramp_start:overlap_start] = inst_pre + vox_pre * ramp
    return out


# Legacy subtractive paths kept for A/B comparison via env flag.
def _suppress_intro_vocals_subtractive(cur: dict, n_samples: int,
                                        ramp_in_samples: int = 0) -> np.ndarray:
    full = cur['full']
    vox = cur.get('stems', {}).get('vocals')
    if vox is None or n_samples <= 0:
        return full
    n_samples = int(min(n_samples, full.shape[1]))
    out = full.copy()
    out[:, :n_samples] -= vox[:, :n_samples]
    if ramp_in_samples > 0:
        end = min(out.shape[1], n_samples + ramp_in_samples)
        ramp_n = end - n_samples
        if ramp_n > 0:
            ramp = np.linspace(0.0, 1.0, ramp_n, dtype=np.float32)
            out[:, n_samples:end] -= vox[:, n_samples:end] * (1.0 - ramp)
    return out


def _suppress_outro_vocals_subtractive(output: np.ndarray, prev: dict,
                                        n_samples: int,
                                        ramp_out_samples: int = 0) -> np.ndarray:
    vox = prev.get('stems', {}).get('vocals')
    if vox is None or n_samples <= 0 or output.shape[1] == 0:
        return output
    out = output.copy()
    n_samples = int(min(n_samples, out.shape[1]))
    overlap_start = out.shape[1] - n_samples
    prev_full_len = prev.get('full', vox).shape[1] if hasattr(prev.get('full', vox), 'shape') else vox.shape[1]
    vox_tail = vox[:, -prev_full_len:] if vox.shape[1] >= prev_full_len else vox
    if vox_tail.shape[1] >= n_samples:
        vox_overlap = vox_tail[:, -n_samples:]
    else:
        pad = n_samples - vox_tail.shape[1]
        vox_overlap = np.concatenate([np.zeros((vox_tail.shape[0], pad), dtype=vox_tail.dtype),
                                      vox_tail], axis=1)
    out[:, overlap_start:] -= vox_overlap[:, :n_samples]
    if ramp_out_samples > 0:
        ramp_n = min(ramp_out_samples, overlap_start)
        if ramp_n > 0:
            ramp = np.linspace(0.0, 1.0, ramp_n, dtype=np.float32)
            ramp_start = overlap_start - ramp_n
            if vox_tail.shape[1] >= n_samples + ramp_n:
                vox_ramp = vox_tail[:, -(n_samples + ramp_n):-n_samples]
                out[:, ramp_start:overlap_start] -= vox_ramp * ramp
    return out


def _suppress_intro_vocals(cur: dict, n_samples: int,
                           ramp_in_samples: int = 0) -> np.ndarray:
    """Dispatch to stem-additive (default) or subtractive (legacy A/B)."""
    import os
    if os.getenv('AIJOCKEY_STEM_SWAP', '1') == '0':
        return _suppress_intro_vocals_subtractive(cur, n_samples, ramp_in_samples)
    return _suppress_intro_vocals_stemwise(cur, n_samples, ramp_in_samples)


def _suppress_outro_vocals(output: np.ndarray, prev: dict,
                           n_samples: int,
                           ramp_out_samples: int = 0) -> np.ndarray:
    """Dispatch to stem-additive (default) or subtractive (legacy A/B)."""
    import os
    if os.getenv('AIJOCKEY_STEM_SWAP', '1') == '0':
        return _suppress_outro_vocals_subtractive(output, prev, n_samples, ramp_out_samples)
    return _suppress_outro_vocals_stemwise(output, prev, n_samples, ramp_out_samples)


# Transitions where output length is NOT prev_len + cur_len (concat/insert/loop):
# accent overlay offset based on cur length is unreliable, skip.
_ACCENT_INCOMPATIBLE = frozenset({
    'scratch_fill',     # concat: prev + scratch + cur
    'silence_drop',     # inserts silence + impacts at boundary
    'drum_break',       # body + drum_seg + cur
    'loop_callback',    # cuts to looped cur
    'spinback',         # has own vinyl FX at boundary
    'loop_tighten',     # has own riser/airhorn
})


# Categories that must remain grid-locked. Risers/sweeps need to RESOLVE on
# the phrase 1 of the incoming drop; jittering the start would mistime the
# climax. Other categories (impacts/snare_rolls/hihat_rolls) take micro-jitter
# happily — that's where the human "pocket" lives.
_LOCKED_ACCENT_CATEGORIES = frozenset({'risers', 'sweeps'})

# Humanization tuning. Tightened for techno/house feel; widen for hip-hop
# in Phase 2.
_JITTER_MS = 8.0
_VELOCITY_RANGE = (0.92, 1.08)


def _humanize_accent(category: str, anchor_seed) -> tuple[float, float]:
    """Return (timing_jitter_ms, velocity_scalar) for an accent overlay.

    Locked categories return (0.0, 1.0). All others jitter deterministically
    via a per-junction seed so re-renders are reproducible.
    """
    if category in _LOCKED_ACCENT_CATEGORIES:
        return 0.0, 1.0
    import random
    rng = random.Random(anchor_seed)
    jitter_ms = rng.uniform(-_JITTER_MS, _JITTER_MS)
    velocity = rng.uniform(*_VELOCITY_RANGE)
    return jitter_ms, velocity


def _overlay_accent_hint(out: np.ndarray, cur: dict, sample_bank: SampleBank,
                         target_bpm: float, beat_dur: float) -> np.ndarray:
    ah = cur['entry'].get('accent_hint')
    if not ah:
        return out
    tech_name = cur['entry'].get('transition_in', {}).get('name', 'crossfade')
    if tech_name in _ACCENT_INCOMPATIBLE:
        return out
    cat = str(ah.get('fx_category', 'hihat_rolls'))
    if not sample_bank.has(cat):
        # gated by allowed_types — silently skip
        return out
    beats = float(ah.get('beats', 2.0))
    try:
        sample = sample_bank.get_fx(cat, target_bpm, beats=beats)
    except Exception:
        return out
    base_at = max(0, out.shape[1] - cur['full'].shape[1] - int(beats * beat_dur * SR))
    seed = (cur['entry'].get('clip_id', '?'),
            cur['entry'].get('junction_index', 0),
            cat)
    jitter_ms, velocity = _humanize_accent(cat, seed)
    jitter_samples = int(jitter_ms * SR / 1000.0)
    at = max(0, base_at + jitter_samples)
    return T.overlay_sample(out, sample, at, gain=0.35 * velocity)


def apply_transition(output: np.ndarray, prev: dict, cur: dict,
                     sample_bank: SampleBank, target_bpm: float) -> np.ndarray:
    tech = cur['entry']['transition_in']
    name = tech.get('name', 'crossfade')
    bars = int(tech.get('bars', 16))
    beat_dur = 60.0 / max(target_bpm, 1.0)
    overlap_n = int(bars * 4 * beat_dur * SR)
    ramp = int(beat_dur * SR * 2)
    # Build vocal-suppressed cur for overlap-style transitions
    cur_no_intro_vox = dict(cur)
    cur_no_intro_vox['full'] = _suppress_intro_vocals(
        cur, n_samples=overlap_n, ramp_in_samples=ramp)
    # Also suppress outgoing vocals during the same overlap region
    output_no_outro_vox = _suppress_outro_vocals(
        output, prev, n_samples=overlap_n, ramp_out_samples=ramp)

    def _done(o: np.ndarray) -> np.ndarray:
        return _overlay_accent_hint(o, cur, sample_bank, target_bpm, beat_dur)

    if name in ('cut', 'fade_in'):
        return _done(T.cut_transition(output, cur['full']))
    if name == 'crossfade':
        return _done(T.crossfade_transition(output_no_outro_vox, cur_no_intro_vox['full'], SR, bars, beat_dur))
    if name == 'eq_swap':
        # Embellish: hi-hat roll lead-in to incoming on energy lift
        out = T.eq_swap_transition(output_no_outro_vox, cur_no_intro_vox['full'], SR, bars, beat_dur)
        if cur['entry']['segment'].get('energy', 0.5) > 0.7:
            roll = sample_bank.get_fx('hihat_rolls', target_bpm, beats=2.0)
            roll_at = max(0, out.shape[1] - cur['full'].shape[1] - int(2 * beat_dur * SR))
            out = T.overlay_sample(out, roll, roll_at, gain=0.35)
        return _done(out)
    if name == 'filter_fade':
        out = T.filter_fade_transition(output_no_outro_vox, cur_no_intro_vox['full'], SR, bars, beat_dur)
        # Down-sweep accentuates filter close
        sweep = sample_bank.get_fx('sweeps', target_bpm, beats=float(bars))
        sweep_at = max(0, out.shape[1] - cur['full'].shape[1] - int(bars * 4 * beat_dur * SR))
        return _done(T.overlay_sample(out, sweep, sweep_at, gain=0.25))
    if name == 'silence_drop':
        silence_beats = float(tech.get('silence_beats', 2))
        out = T.silence_drop_transition(output, cur['full'], SR, silence_beats, beat_dur)
        impact = sample_bank.get_fx('impacts', target_bpm, beats=1.5)
        impact_at = out.shape[1] - cur['full'].shape[1]
        out = T.overlay_sample(out, impact, impact_at, gain=0.6)
        # Sub-drop on the re-entry too
        sub = sample_bank.get_fx('sub_drops', target_bpm, beats=2.0)
        out = T.overlay_sample(out, sub, impact_at, gain=0.5)
        return _done(out)
    if name == 'drum_break':
        drums = prev['stems'].get('drums')
        if drums is None:
            return _done(T.crossfade_transition(output, cur['full'], SR, bars, beat_dur))
        drum_seg = drums[:, :prev['full'].shape[1]]
        body_minus_break_n = int(bars * 4 * beat_dur * SR)
        body = output[:, :-body_minus_break_n] if output.shape[1] > body_minus_break_n else np.zeros((2, 0), dtype=np.float32)
        out = T.drum_break_transition(drum_seg, cur['full'], SR, bars, beat_dur,
                                      out_full_remainder=body)
        # Snare roll lead-in to incoming
        roll = sample_bank.get_fx('snare_rolls', target_bpm, beats=4.0)
        roll_at = max(0, out.shape[1] - cur['full'].shape[1] - int(4 * beat_dur * SR))
        return _done(T.overlay_sample(out, roll, roll_at, gain=0.4))
    if name == 'mashup':
        inst = sum(v for k, v in prev['stems'].items() if k != 'vocals')
        in_vox = cur['stems'].get('vocals')
        if in_vox is None or not hasattr(inst, 'shape'):
            return _done(T.crossfade_transition(output, cur['full'], SR, bars, beat_dur))
        n_overlay = int(bars * 4 * beat_dur * SR)
        body = output[:, :-n_overlay] if output.shape[1] > n_overlay else np.zeros((2, 0), dtype=np.float32)
        inst_tail = inst[:, -n_overlay:] if inst.shape[1] >= n_overlay else inst
        return _done(T.mashup_transition(inst_tail, in_vox, cur['full'], SR, bars, beat_dur,
                                         out_full_remainder=body))
    if name == 'stem_swap':
        inst = sum(v for k, v in prev['stems'].items() if k != 'vocals')
        in_vox = cur['stems'].get('vocals')
        if in_vox is None:
            return _done(T.crossfade_transition(output, cur['full'], SR, bars, beat_dur))
        n_overlay = int(bars * 4 * beat_dur * SR)
        body = output[:, :-n_overlay] if output.shape[1] > n_overlay else np.zeros((2, 0), dtype=np.float32)
        inst_tail = inst[:, -n_overlay:] if inst.shape[1] >= n_overlay else inst
        return _done(T.stem_swap_transition(inst_tail, in_vox, cur['full'], SR, bars, beat_dur,
                                            out_full_remainder=body))
    if name == 'echo_out':
        return _done(T.echo_out_transition(
            output_no_outro_vox, cur_no_intro_vox['full'], SR, bars, beat_dur,
            delay_beats=float(tech.get('delay_beats', 0.5)),
            feedback=float(tech.get('feedback', 0.55)),
        ))
    if name == 'spinback':
        sb_beats = float(tech.get('spinback_beats', 4))
        out = T.spinback_transition(output, cur['full'], SR, sb_beats, beat_dur)
        vinyl = sample_bank.get_fx('vinyl', target_bpm, beats=sb_beats)
        fx_at = max(0, out.shape[1] - cur['full'].shape[1] - int(sb_beats * beat_dur * SR))
        return _done(T.overlay_sample(out, vinyl, fx_at, gain=0.5))
    if name == 'pitch_bend':
        return _done(T.pitch_bend_transition(
            output, cur['full'], SR, bars, beat_dur,
            semitones=float(tech.get('semitones', 1.0)),
        ))
    if name == 'loop_tighten':
        out = T.loop_tighten_transition(
            output, cur['full'], SR, beat_dur,
            start_bars=int(tech.get('start_bars', 4)),
        )
        # Riser building tension during tighten
        sb = int(tech.get('start_bars', 4))
        riser = sample_bank.get_fx('risers', target_bpm, beats=float(sb * 2))
        tighten_n = int(sb * 4 * beat_dur * SR * 2)
        riser_at = max(0, out.shape[1] - cur['full'].shape[1] - tighten_n)
        out = T.overlay_sample(out, riser, riser_at, gain=0.4)
        # Airhorn drop-in disabled in Phase A polish — pollutes mix on
        # well-mastered tracks. SampleBank will return silence for 'airhorns'
        # under PHASE1_ALLOWED_TYPES anyway; this avoids the overlay call entirely.
        return _done(out)
    if name == 'scratch_fill':
        hook_seg = prev['full'][:, -int(2 * beat_dur * SR):]
        scratch = T.scratch_fill(hook_seg, SR, beat_dur, n_jogs=int(tech.get('n_jogs', 4)))
        return _done(np.concatenate([output, scratch, cur['full']], axis=1))
    if name == 'loop_callback':
        reps = int(tech.get('repetitions', 2))
        looped = T.loop_callback(cur['full'], reps)
        return _done(T.cut_transition(output, looped))
    return _done(T.crossfade_transition(output, cur['full'], SR, bars, beat_dur))


# ---------------------------------------------------------------------------
# Top-level execute
# ---------------------------------------------------------------------------

def execute(timeline_path: str, cache_dir: str, out_path: str,
            samples_dir: str = 'samples') -> np.ndarray:
    with open(timeline_path) as f:
        blob = json.load(f)
    tl = blob['timeline']
    meta = blob.get('meta') or {}
    max_ratio = meta.get('max_stretch_ratio')
    max_bpm_ratio = float(max_ratio) if max_ratio is not None else None
    if not tl:
        raise SystemExit("empty timeline")

    # Collect needed clip metadata
    clips_meta: dict[str, dict] = {}
    for entry in tl:
        cid = entry['clip_id']
        if cid in clips_meta:
            continue
        with open(Path(cache_dir) / f'{cid}.json') as f:
            clips_meta[cid] = json.load(f)

    # Phrase-quantize segment boundaries so junctions land on real phrase 1.
    # Disable with AIJOCKEY_PHRASE_QUANTIZE=0 if A/B testing.
    import os
    if os.getenv('AIJOCKEY_PHRASE_QUANTIZE', '1') != '0':
        before = [(e['segment'].get('start'), e['segment'].get('end')) for e in tl]
        quantize_timeline_to_phrase(tl, clips_meta, max_drift_bars=2)
        moved = sum(1 for (a, e) in zip(before, tl)
                    if a != (e['segment'].get('start'), e['segment'].get('end')))
        print(f"phrase-quantize: snapped {moved}/{len(tl)} segment boundaries")

    # Constitutional validation. Hard musical-rule layer above LLM choices.
    # Set AIJOCKEY_CONSTITUTIONAL=0 to disable.
    if os.getenv('AIJOCKEY_CONSTITUTIONAL', '1') != '0':
        try:
            import constitutional as C
            violations = C.validate(tl, clips_meta)
            if violations:
                rejects = [v for v in violations if v.severity == 'reject']
                warns = [v for v in violations if v.severity != 'reject']
                print(f"constitutional: {len(rejects)} rejects, {len(warns)} warns")
                for v in rejects:
                    print(f"  REJECT j{v.junction_index} {v.rule}: {v.detail}")
                for v in warns:
                    print(f"  warn j{v.junction_index} {v.rule}: {v.detail}")
                C.repair(tl, violations)
        except ImportError:
            pass

    sample_bank = SampleBank(samples_dir)
    print(f"sample bank: real types={list(sample_bank.bank.keys())}, "
          f"synth types={list(sample_bank.list_available_types())}")

    # Render all segments first
    rendered = []
    for i, entry in enumerate(tl):
        print(f"rendering {i+1}/{len(tl)}: {entry['clip_id']} [{entry['segment']['type']}]")
        full, stems = render_segment(entry, clips_meta, cache_dir,
                                     max_bpm_ratio=max_bpm_ratio)
        rendered.append({'entry': entry, 'full': full, 'stems': stems})

    output = rendered[0]['full']
    for i in range(1, len(rendered)):
        prev = rendered[i - 1]
        cur = rendered[i]
        target_bpm = float(cur['entry']['target_bpm'])
        print(f"transition {i}: {cur['entry']['transition_in']['name']}")
        output = apply_transition(output, prev, cur, sample_bank, target_bpm)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(out_path, torch.from_numpy(output.astype(np.float32)), SR)
    print(f"wrote {out_path} ({output.shape[1] / SR:.1f}s)")
    return output


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeline', default='output/timeline.json')
    ap.add_argument('--cache', default='cache')
    ap.add_argument('--out', default='output/raw_mix.wav')
    ap.add_argument('--samples', default='samples')
    args = ap.parse_args()
    execute(args.timeline, args.cache, args.out, args.samples)
