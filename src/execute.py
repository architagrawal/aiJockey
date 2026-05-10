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
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchaudio
import pyrubberband as pyrb

import transitions as T
from camelot import semitones_between, camelot_distance
from phrase import snap_to_phrase
from samples import SampleBank

SR = 44100

# Perf: pre-load CPU thread budget for stem-parallel rubberband + I/O.
# pyrubberband shells out to the rubberband CLI which is single-threaded
# per invocation but releases the GIL — N independent stems can run
# concurrently. cap at 4 (= stem count) to avoid oversubscription.
_STEM_WORKERS = max(1, min(8, int(os.environ.get('AIJOCKEY_STEM_WORKERS', '8'))))
# Allow rendering multiple timeline segments concurrently. Each segment
# allocates ~200MB of PCM at 600s/2ch/float32; default 6 saturates the MI300X
# host CPU (which is what bottlenecks the rubberband subprocess) without
# blowing peak RSS on a 240GB-RAM box. Drop to 2 only on RAM-constrained envs.
_RENDER_WORKERS = max(1, min(16, int(os.environ.get('AIJOCKEY_RENDER_WORKERS', '6'))))

# CUDA/ROCm autotuner: lets cuDNN/MIOpen pick fastest conv algos for the
# fixed input shapes seen in this pipeline. Idempotent + harmless on CPU.
try:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


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


def enforce_min_segment_length(timeline: list[dict],
                                clips_meta: dict[str, dict],
                                min_bars: int = 8) -> list[dict]:
    """Extend any segment shorter than min_bars (at clip BPM) by walking
    forward in the clip via downbeats. Prevents tiny (<8 bar) segments
    that get fully consumed by overlap windows (STATUS bug #1 root cause).

    Only patches the truly tiny ones — segments already >= min_bars are
    left alone so planner duration targets stay honored.
    """
    for entry in timeline:
        cid = entry.get('clip_id')
        seg = entry.get('segment') or {}
        meta = clips_meta.get(cid) or {}
        if 'start' not in seg or 'end' not in seg:
            continue
        bpm = float(meta.get('tempo', 120.0)) or 120.0
        bar_dur = 4.0 * 60.0 / bpm
        target_min = min_bars * bar_dur
        cur_len = float(seg['end']) - float(seg['start'])
        if cur_len >= target_min:
            continue
        clip_dur = float(meta.get('duration', seg['end']))
        new_end = min(clip_dur, float(seg['start']) + target_min)
        downbeats = meta.get('downbeats') or []
        if downbeats:
            try:
                new_end = snap_to_phrase(new_end, downbeats, bars_per_phrase=4)
            except Exception:
                pass
        if new_end > float(seg['start']):
            seg['end'] = new_end
            seg['min_length_extended'] = True
    return timeline


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

def _load_one_stem(fp: Path) -> np.ndarray | None:
    if not fp.exists():
        return None
    wav, sr = torchaudio.load(str(fp))
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.size(0) == 1:
        wav = wav.repeat(2, 1)
    return wav.numpy().astype(np.float32)


def snap_segment_end_to_vocal_silence(timeline: list[dict],
                                       clips_meta: dict[str, dict],
                                       cache_dir: str = 'cache',
                                       look_ahead_sec: float = 5.0,
                                       look_back_sec: float = 1.5,
                                       quiet_threshold_db: float = -36.0,
                                       quiet_min_dur: float = 0.25,
                                       ) -> list[dict]:
    """Snap segment.end to nearest vocal-quiet point so transitions never
    cut a vocal phrase mid-flow. Operates only when vocals stem available.

    For each non-final entry: scan vocals stem RMS in
    [end - look_back_sec, end + look_ahead_sec]. Find earliest window of
    duration >= quiet_min_dur where RMS < threshold. Snap segment.end to
    start of that window. If no quiet window in the look-ahead, leave end
    alone (do NOT truncate vocals abruptly — extending a few seconds is
    musical, cutting is not).

    Skips entry when:
      - vocals stem missing
      - vocals are essentially silent throughout (instrumental clip)
      - quiet window already at/before original end
    """
    import os
    if os.environ.get('AIJOCKEY_VOCAL_END_SNAP', '1') == '0':
        return timeline
    if len(timeline) < 2:
        return timeline
    snapped = 0
    for entry in timeline[:-1]:
        cid = entry.get('clip_id')
        seg = entry.get('segment') or {}
        if 'end' not in seg or 'start' not in seg:
            continue
        meta = clips_meta.get(cid) or {}
        clip_dur = float(meta.get('duration', seg['end']))
        vfp = Path(cache_dir) / 'stems' / cid / 'vocals.wav'
        if not vfp.exists():
            continue
        try:
            wav, sr = torchaudio.load(str(vfp))
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=False)
            else:
                wav = wav.squeeze(0)
            v = wav.numpy().astype(np.float32)
        except Exception:
            continue
        # If clip is mostly instrumental (vocals near silent), no need to snap
        global_rms = float(np.sqrt(np.mean(v[: int(min(len(v), 30 * sr))] ** 2) + 1e-12))
        if global_rms < 10 ** (quiet_threshold_db / 20.0) * 1.5:
            continue
        end_s = float(seg['end'])
        scan_start = max(0.0, end_s - look_back_sec)
        scan_end = min(clip_dur, end_s + look_ahead_sec)
        i0, i1 = int(scan_start * sr), int(scan_end * sr)
        if i1 <= i0 + sr // 4:
            continue
        seg_audio = v[i0:i1]
        win = max(1, int(0.10 * sr))   # 100ms RMS window
        # Vectorized RMS via cumulative sum of squares
        sq = seg_audio ** 2
        cs = np.concatenate(([0.0], np.cumsum(sq)))
        rms = np.sqrt((cs[win:] - cs[:-win]) / win + 1e-12)
        thr_lin = 10 ** (quiet_threshold_db / 20.0)
        quiet_mask = rms < thr_lin
        # Need a run of >= quiet_min_dur seconds (in 100ms hops)
        need_hops = max(1, int(quiet_min_dur * sr / win))
        # Find earliest run of `need_hops` consecutive True
        run = 0
        snap_idx = -1
        for k, q in enumerate(quiet_mask):
            run = run + 1 if q else 0
            if run >= need_hops:
                snap_idx = k - need_hops + 1
                break
        if snap_idx < 0:
            continue
        # Convert idx in seg_audio back to absolute clip seconds
        new_end = scan_start + (snap_idx * win) / sr
        if new_end <= float(seg['start']) + 0.5:
            continue   # would create degenerate segment
        if abs(new_end - end_s) < 0.10:
            continue   # already aligned
        seg['end'] = float(new_end)
        seg['vocal_end_snapped'] = True
        seg['vocal_end_snap_delta'] = round(new_end - end_s, 3)
        snapped += 1
    if snapped:
        print(f"vocal-end-snap: adjusted {snapped}/{len(timeline)-1} junctions to vocal-quiet boundary")
    return timeline


def load_stems(clip_id: str, cache_dir: str = 'cache') -> dict[str, np.ndarray]:
    base = Path(cache_dir) / 'stems' / clip_id
    names = ('drums', 'bass', 'other', 'vocals')
    paths = [base / f'{n}.wav' for n in names]
    # Parallel I/O — torchaudio.load + resample releases the GIL.
    with ThreadPoolExecutor(max_workers=_STEM_WORKERS) as ex:
        arrs = list(ex.map(_load_one_stem, paths))
    return {n: a for n, a in zip(names, arrs) if a is not None}


# ---------------------------------------------------------------------------
# Time-stretch + pitch shift
# ---------------------------------------------------------------------------

_RB_BIN = os.environ.get('AIJOCKEY_RUBBERBAND_BIN', 'rubberband')


def _rubberband_combined(x: np.ndarray, sr: int, rate: float,
                          semitones: float) -> np.ndarray:
    """Run rubberband CLI ONCE with both --tempo and --pitch flags.

    pyrubberband's `time_stretch` + `pitch_shift` invokes the binary
    twice (two subprocess spawns + four tempfile WAV reads/writes).
    Calling rubberband directly with both flags cuts that overhead in
    half and avoids one resample-quality round-trip through the WAV
    container.

    Falls back to the two-pass pyrubberband path on any failure so the
    behavior degrades to current performance, never below.
    """
    import subprocess
    import tempfile
    import soundfile as sf

    in_fd, in_path = tempfile.mkstemp(prefix='rb_in_', suffix='.wav')
    out_fd, out_path = tempfile.mkstemp(prefix='rb_out_', suffix='.wav')
    os.close(in_fd)
    os.close(out_fd)
    try:
        sf.write(in_path, x, sr, subtype='FLOAT')
        cmd = [_RB_BIN, '-q']
        if abs(rate - 1.0) > 1e-4:
            # rubberband interprets --tempo as OUTPUT_TEMPO/INPUT_TEMPO,
            # which equals (dst_bpm/src_bpm) — same as the rate we already
            # computed. NB: pyrubberband.time_stretch calls this 'rate'.
            cmd += ['--tempo', f'{rate:.6f}']
        if abs(semitones) > 1e-4:
            cmd += ['--pitch', f'{semitones:.6f}']
        if len(cmd) == 2:
            # No-op → just return input (avoids a useless rubberband pass).
            return x
        cmd += [in_path, out_path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f'rubberband exit {r.returncode}: {r.stderr[:200]}')
        y, _ = sf.read(out_path, dtype='float32', always_2d=True)
        return y
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def stretch_and_pitch(wav: np.ndarray, sr: int, src_bpm: float,
                      dst_bpm: float, semitones: float,
                      max_bpm_ratio: float | None = None,
                      vocal_safe: bool = True) -> np.ndarray:
    """rubberband expects (T, channels). wav is (channels, T).

    vocal_safe (default True): hard-cap stretch ratio to [0.92, 1.09] —
    DJ industry 8% rule. Beyond that = audible vocal warp. When cap fires
    the effective dst BPM moves toward src; residual beat-grid drift
    handled by DTW alignment in overlap region. Set vocal_safe=False for
    legacy unlimited stretch (env AIJOCKEY_VOCAL_SAFE_STRETCH=0).
    """
    x = wav.T.astype(np.float32)
    # Tempo octave normalization: beat trackers sometimes detect half/double
    # tempo (trap 130 → 65). Canonicalize to [90, 180] before stretch.
    from tempo_octave import normalize_tempo
    src_bpm = normalize_tempo(float(src_bpm))
    dst = float(dst_bpm)
    # Vocal-safe stretch cap — overrides max_bpm_ratio when stricter.
    # 0.92/1.09 = ±8% (DJ rule). Default ON.
    if vocal_safe and os.environ.get('AIJOCKEY_VOCAL_SAFE_STRETCH', '1') != '0' and src_bpm > 0:
        ratio = dst / src_bpm
        if ratio < 0.92:
            dst = src_bpm * 0.92
        elif ratio > 1.09:
            dst = src_bpm * 1.09
    if src_bpm > 0 and max_bpm_ratio and max_bpm_ratio > 1.0:
        lo = src_bpm / max_bpm_ratio
        hi = src_bpm * max_bpm_ratio
        dst = max(lo, min(hi, dst))
    rate = (dst / src_bpm) if (src_bpm > 0 and abs(src_bpm - dst) > 0.01) else 1.0
    pitch = semitones if abs(semitones) > 0.01 else 0.0
    if rate == 1.0 and pitch == 0.0:
        return wav
    # Try single-call path first (2× faster than pyrubberband two-pass).
    if os.environ.get('AIJOCKEY_RB_COMBINED', '1') != '0':
        try:
            return _rubberband_combined(x, sr, rate, pitch).T.astype(np.float32)
        except Exception as e:
            print(f"warn: rubberband single-call failed ({e}), falling back to two-pass")
    # Fallback: original pyrubberband two-pass path.
    if rate != 1.0:
        try:
            x = pyrb.time_stretch(x, sr, rate)
        except Exception as e:
            print(f"warn: time_stretch failed ({e}), using original")
    if pitch != 0.0:
        try:
            x = pyrb.pitch_shift(x, sr, pitch)
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
    # Stretch + pitch each stem in parallel — pyrubberband is a subprocess
    # that releases the GIL, so 4 stems run truly concurrently.
    names = list(sliced.keys())

    def _work(n):
        return stretch_and_pitch(sliced[n], SR, src_bpm, target_bpm, semitones,
                                  max_bpm_ratio=max_bpm_ratio)
    with ThreadPoolExecutor(max_workers=_STEM_WORKERS) as ex:
        results = list(ex.map(_work, names))
    processed = dict(zip(names, results))
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
        ramp_n = min(ramp_out_samples, overlap_start,
                     max(0, inst.shape[1] - n_samples),
                     max(0, vox.shape[1] - n_samples))
        if ramp_n > 0:
            inst_pre = inst[:, -(n_samples + ramp_n):-n_samples]
            vox_pre = vox[:, -(n_samples + ramp_n):-n_samples]
            # Final length sanity — guard against any upstream stem-length skew.
            ramp_n = min(ramp_n, inst_pre.shape[1], vox_pre.shape[1])
            if ramp_n > 0:
                ramp = np.linspace(1.0, 0.0, ramp_n, dtype=np.float32)
                ramp_start = overlap_start - ramp_n
                out[:, ramp_start:overlap_start] = (
                    inst_pre[:, -ramp_n:] + vox_pre[:, -ramp_n:] * ramp
                )
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
    # Clamp overlap to ONE THIRD of the shorter side (incoming OR outgoing
    # tail). Each segment then contributes ~67% unique audio post-handoff
    # instead of being consumed in half. 1-bar floor keeps transitions
    # audible; absolute cap at 8 bars (~16s at 120 BPM) prevents long
    # overlaps from eating the whole mix even when both sides are long.
    cur_len = cur['full'].shape[1]
    out_len = output.shape[1]
    bar_samples = int(beat_dur * 4 * SR)
    abs_cap_samples = 8 * bar_samples
    max_overlap_samples = max(bar_samples,
                               min(cur_len, out_len) // 3)
    max_overlap_samples = min(max_overlap_samples, abs_cap_samples)
    # Harmonic-distance overlap cap. Long overlaps on dissonant key pairs
    # stack semitone clashes audibly; tighten window so clash exposure is
    # short. Inspired by kckDeepak/AI-DJ-Mixing-System (Camelot-driven
    # dynamic overlap). Disable with AIJOCKEY_HARMONIC_OVERLAP_CAP=0.
    if os.getenv('AIJOCKEY_HARMONIC_OVERLAP_CAP', '1') != '0':
        prev_key = prev['entry'].get('_clip_key', '?') if prev else '?'
        cur_key = cur['entry'].get('_clip_key', '?')
        key_dist = camelot_distance(prev_key, cur_key)
        if key_dist >= 4:
            harm_cap = 4 * bar_samples
        elif key_dist >= 2:
            harm_cap = 6 * bar_samples
        else:
            harm_cap = max_overlap_samples
        if harm_cap < max_overlap_samples:
            max_overlap_samples = harm_cap
            tech['harmonic_dist'] = key_dist
            tech['harmonic_cap_bars'] = harm_cap // bar_samples
    overlap_n = min(int(bars * 4 * beat_dur * SR), max_overlap_samples)
    # Critical: transition primitives (crossfade_transition, eq_swap_transition,
    # etc.) recompute overlap from `bars` internally. We must shadow `bars`
    # to the clamped value, otherwise they consume the full segment and
    # output stops growing past the longest segment (STATUS bug #1).
    bars = max(1, overlap_n // bar_samples)
    if int(tech.get('bars', 16)) != bars:
        tech['bars_effective'] = bars
    # Vocal ramp window (was 2 beats, now 4) — smoother fade-in of incoming
    # vocals after stem-additive overlap. Tunable via env.
    ramp_beats = float(os.environ.get('AIJOCKEY_VOCAL_RAMP_BEATS', '4'))
    ramp = int(beat_dur * SR * ramp_beats)
    # Vocal-aware transition guard (per docs/dj_research.md §6, §10):
    #
    # Vocals tolerate frequency-band swaps, filter sweeps, echo/reverb,
    # drum-only manipulation, and stem swaps. They DON'T tolerate
    # time/pitch warps and per-sample manipulation that shreds the vocal
    # waveform itself.
    #
    # Two-tier gating:
    #   - SHREDDERS (always reject when vocals present > 0.30):
    #       time/pitch warps + per-sample mangling
    #   - HEAVY (reject only when vocals DENSE > 0.55):
    #       drum manipulations that may briefly imbalance with vocal —
    #       fine for verse-level vocals, risky over chorus hooks
    #   - SAFE (always pass): everything else (eq_swap, bass_swap,
    #       echo_out, drum_break, silence_drop, filter_fade, stem_swap,
    #       acapella_drop, mashup, etc.)
    # AIJOCKEY_VOCAL_GUARD=0 to disable; AIJOCKEY_VOCAL_GUARD_THR overrides
    # the SHREDDERS threshold (default 0.30); AIJOCKEY_VOCAL_GUARD_THR_HEAVY
    # overrides HEAVY threshold (default 0.55).
    if os.environ.get('AIJOCKEY_VOCAL_GUARD', '1') != '0':
        prev_va = float(((prev.get('entry') or {}).get('segment') or {}).get('vocal_activity') or 0.0)
        cur_va = float(((cur.get('entry') or {}).get('segment') or {}).get('vocal_activity') or 0.0)
        thr_shred = float(os.environ.get('AIJOCKEY_VOCAL_GUARD_THR', '0.30'))
        thr_heavy = float(os.environ.get('AIJOCKEY_VOCAL_GUARD_THR_HEAVY', '0.55'))
        SHREDDERS = {'pitch_bend', 'bpm_warp', 'tape_stop', 'chop',
                     'scratch_fill', 'beat_juggle', 'loop_roll',
                     'loop_tighten', 'spectral_hold', 'spinback',
                     'forward_spin'}
        HEAVY = {'drum_replace', 'kickless_swap', 'snare_buildup',
                 'build_riser_drop', 'punch_in'}
        max_va = max(prev_va, cur_va)
        downgrade = False
        reason = ''
        if name in SHREDDERS and max_va > thr_shred:
            downgrade, reason = True, f'shredder>{thr_shred}'
        elif name in HEAVY and max_va > thr_heavy:
            downgrade, reason = True, f'heavy>{thr_heavy}'
        if downgrade:
            print(f"[vocal_guard] {name} → crossfade "
                  f"({reason}, prev_va={prev_va:.2f} cur_va={cur_va:.2f})")
            tech['_vocal_guard_downgraded_from'] = name
            name = 'crossfade'
            tech['name'] = 'crossfade'
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
        inst = _stem_sum(prev.get('stems') or {}, ('drums', 'bass', 'other'))
        in_vox = cur['stems'].get('vocals')
        if in_vox is None or inst is None:
            return _done(T.crossfade_transition(output, cur['full'], SR, bars, beat_dur))
        n_overlay = int(bars * 4 * beat_dur * SR)
        body = output[:, :-n_overlay] if output.shape[1] > n_overlay else np.zeros((2, 0), dtype=np.float32)
        inst_tail = inst[:, -n_overlay:] if inst.shape[1] >= n_overlay else inst
        return _done(T.mashup_transition(inst_tail, in_vox, cur['full'], SR, bars, beat_dur,
                                         out_full_remainder=body))
    if name == 'stem_swap':
        inst = _stem_sum(prev.get('stems') or {}, ('drums', 'bass', 'other'))
        in_vox = cur['stems'].get('vocals')
        if in_vox is None or inst is None:
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

    # Stash per-clip key on each entry for harmonic-distance overlap cap
    # (apply_transition reads entry['_clip_key']). Camelot codes only.
    for entry in tl:
        entry['_clip_key'] = clips_meta[entry['clip_id']].get('key', '?')

    # Phrase-quantize segment boundaries so junctions land on real phrase 1.
    # Disable with AIJOCKEY_PHRASE_QUANTIZE=0 if A/B testing.
    import os
    # Enforce minimum segment length BEFORE phrase quantize. Stops short
    # segments (0.5-2s) from being fully consumed by overlap windows —
    # the root of STATUS bug #1 'render duration shortfall'.
    min_bars = int(os.getenv('AIJOCKEY_MIN_SEGMENT_BARS', '8'))
    before_lens = [e['segment'].get('end', 0) - e['segment'].get('start', 0) for e in tl]
    enforce_min_segment_length(tl, clips_meta, min_bars=min_bars)
    extended = sum(1 for e in tl if e.get('segment', {}).get('min_length_extended'))
    if extended:
        print(f"min-length: extended {extended}/{len(tl)} short segments to >= {min_bars} bars")

    if os.getenv('AIJOCKEY_PHRASE_QUANTIZE', '1') != '0':
        before = [(e['segment'].get('start'), e['segment'].get('end')) for e in tl]
        quantize_timeline_to_phrase(tl, clips_meta, max_drift_bars=2)
        moved = sum(1 for (a, e) in zip(before, tl)
                    if a != (e['segment'].get('start'), e['segment'].get('end')))
        print(f"phrase-quantize: snapped {moved}/{len(tl)} segment boundaries")

    # Vocal-phrase end snap: never cut a vocal mid-flow at junction. Run
    # AFTER phrase-quantize so we may slightly drift past phrase boundary
    # to land in a vocal-quiet pocket. AIJOCKEY_VOCAL_END_SNAP=0 disables.
    snap_segment_end_to_vocal_silence(tl, clips_meta, cache_dir=cache_dir)

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

    # Render segments in parallel — each render is independent; the GIL
    # is released during pyrubberband subprocess + torchaudio file I/O.
    # Bounded by _RENDER_WORKERS so peak memory stays predictable.
    def _render_one(idx_entry):
        i, entry = idx_entry
        full, stems = render_segment(entry, clips_meta, cache_dir,
                                     max_bpm_ratio=max_bpm_ratio)
        return i, {'entry': entry, 'full': full, 'stems': stems}

    rendered: list[dict | None] = [None] * len(tl)
    with ThreadPoolExecutor(max_workers=_RENDER_WORKERS) as ex:
        for i, item in ex.map(_render_one, list(enumerate(tl))):
            print(f"rendered {i+1}/{len(tl)}: {item['entry']['clip_id']} "
                  f"[{item['entry']['segment']['type']}]")
            rendered[i] = item

    output = rendered[0]['full']
    for i in range(1, len(rendered)):
        prev = rendered[i - 1]
        cur = rendered[i]
        target_bpm = float(cur['entry']['target_bpm'])
        print(f"transition {i}: {cur['entry']['transition_in']['name']}")
        output = apply_transition(output, prev, cur, sample_bank, target_bpm)
        # Drop PCM of consumed segment — never referenced again.
        # On a 10-segment / 600s mix this reclaims ~200MB per step.
        rendered[i - 1] = None

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
