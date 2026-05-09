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
from samples import SampleBank

SR = 44100


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
    full = sum(processed.values()).astype(np.float32)
    return full, processed


# ---------------------------------------------------------------------------
# Apply transition between two rendered segments
# ---------------------------------------------------------------------------

def _suppress_intro_vocals(cur: dict, n_samples: int,
                           ramp_in_samples: int = 0) -> np.ndarray:
    """Return a 'full' array where the incoming clip's vocals are silenced
    for the first n_samples (the overlap window with the outgoing clip),
    then ramp back to full vocals over ramp_in_samples.

    Prevents two vocal tracks from overlapping during crossfade-style
    transitions. Drums/bass/other unchanged throughout.
    """
    full = cur['full']
    vox = cur.get('stems', {}).get('vocals')
    if vox is None or n_samples <= 0:
        return full
    n_samples = int(min(n_samples, full.shape[1]))
    out = full.copy()
    # zero vocals for the overlap window
    out[:, :n_samples] -= vox[:, :n_samples]
    # ramp back: linearly add vocals over ramp_in_samples after overlap
    if ramp_in_samples > 0:
        end = min(out.shape[1], n_samples + ramp_in_samples)
        ramp_n = end - n_samples
        if ramp_n > 0:
            ramp = np.linspace(0.0, 1.0, ramp_n, dtype=np.float32)
            out[:, n_samples:end] -= vox[:, n_samples:end] * (1.0 - ramp)
    return out


def _suppress_outro_vocals(output: np.ndarray, prev: dict,
                           n_samples: int,
                           ramp_out_samples: int = 0) -> np.ndarray:
    """Mute the outgoing clip's vocals across the last n_samples of `output`,
    with a pre-overlap ramp-out over ramp_out_samples so the vocals fade
    rather than abruptly cut.

    Required because output already contains the rendered prev clip with
    full vocals. We retroactively subtract them in the overlap region.
    """
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


def _overlay_accent_hint(out: np.ndarray, cur: dict, sample_bank: SampleBank,
                         target_bpm: float, beat_dur: float) -> np.ndarray:
    ah = cur['entry'].get('accent_hint')
    if not ah:
        return out
    cat = ah.get('fx_category', 'hihat_rolls')
    beats = float(ah.get('beats', 2.0))
    try:
        sample = sample_bank.get_fx(str(cat), target_bpm, beats=beats)
    except Exception:
        return out
    at = max(0, out.shape[1] - cur['full'].shape[1] - int(beats * beat_dur * SR))
    return T.overlay_sample(out, sample, at, gain=0.35)


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
        # Airhorn at the drop, if it's a high-energy incoming
        if cur['entry']['segment'].get('energy', 0.5) > 0.85:
            horn = sample_bank.get_fx('airhorns', target_bpm, beats=1.0)
            horn_at = out.shape[1] - cur['full'].shape[1]
            out = T.overlay_sample(out, horn, horn_at, gain=0.3)
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
