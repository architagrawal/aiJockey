"""
DJ transition library — 15 techniques.

API convention:
- All transitions return the FULL stitched audio (head_of_out + transition_region + tail_of_in).
- All accept stereo np.ndarray of shape (2, T).
- bars/beat_dur let caller specify exact musical length.

Sample bank: load_sample_bank() reads samples/manifest.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
from scipy.signal import butter, sosfilt

SR = 44100


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def lp_filter(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(4, cutoff, btype='low', fs=sr, output='sos')
    return np.stack([sosfilt(sos, ch) for ch in x])


def hp_filter(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(4, cutoff, btype='high', fs=sr, output='sos')
    return np.stack([sosfilt(sos, ch) for ch in x])


def equal_power_xfade(a: np.ndarray, b: np.ndarray, n: int,
                       sr: int = 44100,
                       use_spec_mask: bool | None = None,
                       use_beat_align: bool | None = None) -> np.ndarray:
    """
    Equal-power crossfade. a fades out, b fades in over n samples of overlap.
    Output length = a.shape[1] + b.shape[1] - n.

    Two opt-in upgrades on top of time-domain equal-power:
      - Spectrogram-mask blend in the overlap region (AIJOCKEY_SPEC_XFADE=1)
        — avoids 1-8 kHz comb filtering.
      - Beat-grid DTW + sub-sample phase fix on b's lead-in
        (AIJOCKEY_BEAT_ALIGN=1) — kills phase cancellation.
    Both default OFF until validated on droplet.
    """
    import os as _xos
    n = min(n, a.shape[1], b.shape[1])
    if n <= 0:
        return np.concatenate([a, b], axis=1)

    if use_spec_mask is None:
        use_spec_mask = _xos.environ.get("AIJOCKEY_SPEC_XFADE", "0") == "1"
    if use_beat_align is None:
        use_beat_align = _xos.environ.get("AIJOCKEY_BEAT_ALIGN", "0") == "1"

    a_tail = a[:, -n:]
    b_head = b[:, :n]

    if use_beat_align:
        try:
            from beat_align import align_for_overlap
            _, b_head = align_for_overlap(a_tail, b_head, sr=sr)
        except Exception:
            pass

    # FX orchestrator: mutex pairs + per-set budget. Replaces direct
    # env checks with junction-aware orchestration so we don't stack.
    try:
        from fx_orchestrator import is_fx_active, junction_gets_fx
        _jidx = int(_xos.environ.get("AIJOCKEY_CURRENT_JUNCTION_IDX", "0"))
        _budget_ok = junction_gets_fx(_jidx, total_junctions=8)
    except Exception:
        def is_fx_active(e, _j, _g=None):
            return _xos.environ.get(e, "0") == "1"
        _budget_ok = True

    if _budget_ok and is_fx_active("AIJOCKEY_SIDECHAIN_DUCK", _jidx,
                                      "overlap_processing"):
        try:
            from sidechain import sidechain_overlap
            a_tail = sidechain_overlap(a_tail, b_head, sr=sr)
        except Exception:
            pass
    if _budget_ok and is_fx_active("AIJOCKEY_FREQ_DUCK", _jidx,
                                      "overlap_processing"):
        try:
            from freq_mask_duck import freq_mask_duck
            a_tail, b_head = freq_mask_duck(a_tail, b_head, sr=sr)
        except Exception:
            pass
    if _budget_ok and is_fx_active("AIJOCKEY_DEESSER", _jidx,
                                      "vocal_clarity"):
        try:
            from deesser import deess
            a_tail = deess(a_tail, sr=sr)
            b_head = deess(b_head, sr=sr)
        except Exception:
            pass

    if use_spec_mask and n >= 4096:
        try:
            from spec_crossfade import spectral_crossfade
            blended = spectral_crossfade(a_tail, b_head, sr=sr)
        except Exception:
            blended = None
    else:
        blended = None

    if blended is None:
        t = np.linspace(0, np.pi / 2, n)
        fade_out = np.cos(t)
        fade_in = np.sin(t)
        blended = (a_tail * fade_out + b_head * fade_in).astype(np.float32)

    out_len = a.shape[1] + b.shape[1] - n
    out = np.zeros((a.shape[0], out_len), dtype=np.float32)
    pre = a.shape[1] - n
    if pre > 0:
        out[:, :pre] = a[:, :pre]
    out[:, pre:pre + n] = blended
    if b.shape[1] > n:
        out[:, pre + n:] = b[:, n:]
    return out


# ---------------------------------------------------------------------------
# Transitions — each returns full stitched audio
# ---------------------------------------------------------------------------

def cut_transition(out_full: np.ndarray, in_full: np.ndarray) -> np.ndarray:
    """Hard cut on downbeat. Caller responsible for phrase alignment."""
    return np.concatenate([out_full, in_full], axis=1)


def crossfade_transition(out_full: np.ndarray, in_full: np.ndarray,
                         sr: int, bars: int, beat_dur: float) -> np.ndarray:
    """The Fade — equal-power crossfade over N bars."""
    n = int(bars * 4 * beat_dur * sr)
    return equal_power_xfade(out_full, in_full, n)


def eq_swap_transition(out_full: np.ndarray, in_full: np.ndarray,
                       sr: int, bars: int, beat_dur: float,
                       bass_cutoff: float = 200.0) -> np.ndarray:
    """
    EQ Mixing / Blending — kill outgoing bass while raising incoming bass over N bars.
    Highs equal-power crossfade. Avoids bass mud.
    """
    n_region = int(bars * 4 * beat_dur * sr)
    n = min(n_region, out_full.shape[1], in_full.shape[1])
    if n <= 0:
        return np.concatenate([out_full, in_full], axis=1)
    out_region = out_full[:, -n:]
    in_region = in_full[:, :n]
    out_low = lp_filter(out_region, sr, bass_cutoff)
    out_high = hp_filter(out_region, sr, bass_cutoff)
    in_low = lp_filter(in_region, sr, bass_cutoff)
    in_high = hp_filter(in_region, sr, bass_cutoff)
    ramp = np.linspace(1.0, 0.0, n).astype(np.float32)
    inv = 1.0 - ramp
    # Low band: hard swap
    low_mix = out_low * ramp + in_low * inv
    # High band: equal-power
    t = np.linspace(0, np.pi / 2, n).astype(np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    high_mix = out_high * fade_out + in_high * fade_in
    transition = (low_mix + high_mix).astype(np.float32)
    pre = out_full[:, :-n] if out_full.shape[1] > n else np.zeros((2, 0), dtype=np.float32)
    post = in_full[:, n:] if in_full.shape[1] > n else np.zeros((2, 0), dtype=np.float32)
    return np.concatenate([pre, transition, post], axis=1)


def filter_fade_transition(out_full: np.ndarray, in_full: np.ndarray,
                           sr: int, bars: int, beat_dur: float) -> np.ndarray:
    """
    Filter Fade — sweep LP cutoff DOWN on outgoing (8kHz -> 200Hz) while
    fading volume out and incoming in.
    """
    n_region = int(bars * 4 * beat_dur * sr)
    n = min(n_region, out_full.shape[1], in_full.shape[1])
    if n <= 0:
        return np.concatenate([out_full, in_full], axis=1)
    out_region = out_full[:, -n:].astype(np.float32)
    in_region = in_full[:, :n].astype(np.float32)
    chunk = max(1, sr // 50)  # ~20ms
    out_filtered = np.zeros_like(out_region)
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        progress = i / max(1, n)
        cutoff = 8000 - (8000 - 200) * progress
        out_filtered[:, i:end] = lp_filter(out_region[:, i:end], sr, cutoff)
    fade_out = np.linspace(1.0, 0.0, n).astype(np.float32)
    fade_in = np.linspace(0.0, 1.0, n).astype(np.float32)
    mixed = out_filtered * fade_out + in_region * fade_in
    pre = out_full[:, :-n] if out_full.shape[1] > n else np.zeros((2, 0), dtype=np.float32)
    post = in_full[:, n:] if in_full.shape[1] > n else np.zeros((2, 0), dtype=np.float32)
    return np.concatenate([pre, mixed, post], axis=1)


def silence_drop_transition(out_full: np.ndarray, in_full: np.ndarray,
                            sr: int, silence_beats: float, beat_dur: float) -> np.ndarray:
    """Drop — cut to silence for N beats then full re-entry. Tension-release."""
    n_silence = max(0, int(silence_beats * beat_dur * sr))
    silence = np.zeros((2, n_silence), dtype=np.float32)
    return np.concatenate([out_full, silence, in_full], axis=1)


def drum_break_transition(out_drums: np.ndarray, in_full: np.ndarray,
                          sr: int, bars: int, beat_dur: float,
                          out_full_remainder: np.ndarray | None = None) -> np.ndarray:
    """
    Drum Break — drums-only N bars then incoming full.
    out_drums is the drum stem of OUTGOING clip, last N bars used.
    out_full_remainder = outgoing's full mix everything BEFORE the break region (optional).
    """
    n_break = int(bars * 4 * beat_dur * sr)
    drum_only = out_drums[:, -n_break:] if out_drums.shape[1] >= n_break else out_drums
    parts = []
    if out_full_remainder is not None and out_full_remainder.shape[1] > 0:
        parts.append(out_full_remainder.astype(np.float32))
    parts.append(drum_only.astype(np.float32))
    parts.append(in_full.astype(np.float32))
    return np.concatenate(parts, axis=1)


def stem_swap_transition(out_inst: np.ndarray, in_vox: np.ndarray,
                         in_full: np.ndarray, sr: int, bars: int, beat_dur: float,
                         out_full_remainder: np.ndarray | None = None) -> np.ndarray:
    """
    Stem swap — outgoing instrumental + incoming vocals overlaid for N bars,
    then incoming full.
    """
    n_overlay = int(bars * 4 * beat_dur * sr)
    n = min(out_inst.shape[1], in_vox.shape[1], n_overlay)
    if n <= 0:
        return cut_transition(out_inst, in_full)
    overlay = (out_inst[:, :n] * 0.7 + in_vox[:, :n] * 1.0).astype(np.float32)
    parts = []
    if out_full_remainder is not None and out_full_remainder.shape[1] > 0:
        parts.append(out_full_remainder.astype(np.float32))
    parts.append(overlay)
    parts.append(in_full.astype(np.float32))
    return np.concatenate(parts, axis=1)


def mashup_transition(out_inst: np.ndarray, in_vocals: np.ndarray,
                      in_full: np.ndarray, sr: int, bars: int, beat_dur: float,
                      out_full_remainder: np.ndarray | None = None) -> np.ndarray:
    """
    Mashup — sustained vocals-of-A over instrumental-of-B for N bars,
    then crossfade-resolve to incoming full over last 4 bars.
    """
    n_overlay = int(bars * 4 * beat_dur * sr)
    n = min(out_inst.shape[1], in_vocals.shape[1], n_overlay)
    if n <= 0:
        return cut_transition(out_inst, in_full)
    overlay = (out_inst[:, :n] * 0.65 + in_vocals[:, :n] * 1.0).astype(np.float32)
    xfade_n = min(int(4 * 4 * beat_dur * sr), n, in_full.shape[1])
    if xfade_n <= 0:
        body = overlay
    else:
        body = equal_power_xfade(overlay, in_full[:, :n], xfade_n)
    parts = []
    if out_full_remainder is not None and out_full_remainder.shape[1] > 0:
        parts.append(out_full_remainder.astype(np.float32))
    parts.append(body)
    if in_full.shape[1] > n:
        parts.append(in_full[:, n:].astype(np.float32))
    return np.concatenate(parts, axis=1)


def echo_out_transition(out_full: np.ndarray, in_full: np.ndarray,
                        sr: int, bars: int, beat_dur: float,
                        delay_beats: float = 0.5,
                        feedback: float = 0.55,
                        tail_extra_sec: float = 2.0) -> np.ndarray:
    """
    Echo Out — feedback delay tail on last N bars of outgoing. Outgoing dry
    fades to zero across second half of region; tail trails into incoming.
    """
    region_n = int(bars * 4 * beat_dur * sr)
    if out_full.shape[1] < region_n:
        return cut_transition(out_full, in_full)
    delay_samp = max(1, int(delay_beats * beat_dur * sr))
    region = out_full[:, -region_n:].astype(np.float32).copy()
    tail_len = region_n + int(tail_extra_sec * sr)
    tail = np.zeros((2, tail_len), dtype=np.float32)
    tail[:, :region_n] = region
    fade = np.concatenate([
        np.ones(region_n // 2, dtype=np.float32),
        np.linspace(1.0, 0.0, region_n - region_n // 2).astype(np.float32),
    ])
    tail[:, :region_n] *= fade
    for i in range(delay_samp, tail_len):
        tail[:, i] += tail[:, i - delay_samp] * feedback
    tail = np.clip(tail, -1.0, 1.0)
    body = out_full[:, :-region_n].astype(np.float32)
    overlap = min(tail_len, in_full.shape[1])
    mixed_overlap = (tail[:, :overlap] + in_full[:, :overlap].astype(np.float32))
    rest = in_full[:, overlap:].astype(np.float32) if in_full.shape[1] > overlap else np.zeros((2, 0), dtype=np.float32)
    return np.concatenate([body, mixed_overlap, rest], axis=1)


def spinback_transition(out_full: np.ndarray, in_full: np.ndarray,
                        sr: int, spinback_beats: float, beat_dur: float,
                        n_chunks: int = 40, reverse_tail_sec: float = 0.3) -> np.ndarray:
    """
    Spinback — outgoing tail pitch-bends down to ~zero (vinyl-stop emulation)
    + reverse smear, then incoming hits. Big-moment punctuation.
    """
    region_n = int(spinback_beats * beat_dur * sr)
    if out_full.shape[1] < region_n:
        return cut_transition(out_full, in_full)
    region = out_full[:, -region_n:].astype(np.float32)
    chunk_size = max(1, region_n // n_chunks)
    out_chunks: list[np.ndarray] = []
    for i in range(n_chunks):
        rate = 1.0 - (i / n_chunks)
        chunk = region[:, i * chunk_size:(i + 1) * chunk_size]
        if chunk.shape[1] == 0 or rate < 0.05:
            break
        new_len = max(1, int(chunk.shape[1] * (1.0 + (1.0 - rate))))
        idx = np.linspace(0, chunk.shape[1] - 1, new_len).astype(int)
        out_chunks.append(chunk[:, idx])
    tail_n = int(reverse_tail_sec * sr)
    if region.shape[1] >= tail_n:
        out_chunks.append((region[:, -tail_n:][:, ::-1] * 0.5).astype(np.float32))
    spinback = (np.concatenate(out_chunks, axis=1)
                if out_chunks else np.zeros((2, 0), dtype=np.float32))
    body = out_full[:, :-region_n].astype(np.float32)
    return np.concatenate([body, spinback, in_full.astype(np.float32)], axis=1)


def pitch_bend_transition(out_full: np.ndarray, in_full: np.ndarray,
                          sr: int, bars: int, beat_dur: float,
                          semitones: float = 1.0,
                          stages: int = 8) -> np.ndarray:
    """
    Pitch Control — gradually bend outgoing ±semitones over N bars, then
    crossfade into incoming over last 4 bars.
    """
    import pyrubberband as pyrb
    region_n = int(bars * 4 * beat_dur * sr)
    if out_full.shape[1] < region_n or stages < 1:
        return cut_transition(out_full, in_full)
    region = out_full[:, -region_n:].astype(np.float32)
    stage_n = max(1, region_n // stages)
    out_stages: list[np.ndarray] = []
    for i in range(stages):
        progress = i / max(1, stages - 1)
        st = float(semitones * progress)
        chunk = region[:, i * stage_n:(i + 1) * stage_n]
        if chunk.shape[1] == 0:
            continue
        try:
            shifted = pyrb.pitch_shift(chunk.T, sr, st).T
        except Exception:
            shifted = chunk
        if shifted.shape[1] > stage_n:
            shifted = shifted[:, :stage_n]
        elif shifted.shape[1] < stage_n:
            pad = stage_n - shifted.shape[1]
            shifted = np.pad(shifted, ((0, 0), (0, pad)))
        out_stages.append(shifted.astype(np.float32))
    bent = (np.concatenate(out_stages, axis=1)
            if out_stages else np.zeros((2, 0), dtype=np.float32))
    body = out_full[:, :-region_n].astype(np.float32)
    full_bent = np.concatenate([body, bent], axis=1)
    xfade_n = int(4 * 4 * beat_dur * sr)
    return equal_power_xfade(full_bent, in_full.astype(np.float32), xfade_n)


def loop_tighten_transition(out_full: np.ndarray, in_full: np.ndarray,
                            sr: int, beat_dur: float,
                            start_bars: int = 4) -> np.ndarray:
    """
    Looping & Tightening — last N bars looped at halving lengths
    (N -> N/2 -> N/4 -> N/8 -> 0.5 bar) then drop into incoming.
    """
    n_loop_full = int(start_bars * 4 * beat_dur * sr)
    if out_full.shape[1] < n_loop_full:
        return cut_transition(out_full, in_full)
    base = out_full[:, -n_loop_full:].astype(np.float32)
    sequence: list[np.ndarray] = []
    bars_now = float(start_bars)
    while bars_now >= 0.5:
        n = int(bars_now * 4 * beat_dur * sr)
        if n < 1:
            break
        sequence.append(base[:, :n])
        bars_now /= 2.0
    tightened = (np.concatenate(sequence, axis=1)
                 if sequence else np.zeros((2, 0), dtype=np.float32))
    body = out_full[:, :-n_loop_full].astype(np.float32)
    return np.concatenate([body, tightened, in_full.astype(np.float32)], axis=1)


def scratch_fill(hook: np.ndarray, sr: int, beat_dur: float,
                 n_jogs: int = 4) -> np.ndarray:
    """Synthetic scratch — forward/reverse jogs on a hook segment. Returns the fill only."""
    jog_dur = beat_dur * 0.5
    jog_n = max(1, int(jog_dur * sr))
    if hook.shape[1] < jog_n * 2:
        return hook.astype(np.float32)
    jog = hook[:, :jog_n].astype(np.float32)
    out: list[np.ndarray] = []
    for _ in range(n_jogs):
        out.append(jog)
        out.append(jog[:, ::-1])
    return np.concatenate(out, axis=1)


def loop_callback(hook: np.ndarray, repetitions: int = 2) -> np.ndarray:
    """Loop a hook segment N times. Returns the looped block only."""
    return np.tile(hook.astype(np.float32), (1, max(1, repetitions)))


def riser_bridge(duration_sec: float, sr: int) -> np.ndarray:
    """Synthetic white-noise riser, LP cutoff sweeps up over duration."""
    n = max(1, int(duration_sec * sr))
    noise = (np.random.randn(2, n) * 0.1).astype(np.float32)
    out = np.zeros_like(noise)
    chunk = max(1, sr // 50)
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        progress = i / max(1, n)
        cutoff = 200 + (8000 - 200) * progress
        gain = 0.3 + 0.7 * progress
        out[:, i:end] = lp_filter(noise[:, i:end], sr, cutoff) * gain
    return out


# ---------------------------------------------------------------------------
# Sample bank — Sampling technique
# ---------------------------------------------------------------------------

def load_sample_bank(samples_dir: str = 'samples') -> dict[str, list[tuple[str, np.ndarray]]]:
    """
    Load samples per manifest. Returns {type: [(filename, audio_np), ...]}.
    Returns empty dict if manifest missing.
    """
    bank: dict[str, list[tuple[str, np.ndarray]]] = {}
    sd = Path(samples_dir)
    manifest_path = sd / 'manifest.json'
    if not manifest_path.exists():
        return bank
    with open(manifest_path) as f:
        manifest = json.load(f)
    for entry in manifest:
        fp = sd / entry['file']
        if not fp.exists():
            print(f"warn: sample missing: {fp}")
            continue
        wav, sr = torchaudio.load(str(fp))
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.size(0) == 1:
            wav = wav.repeat(2, 1)
        bank.setdefault(entry['type'], []).append(
            (entry['file'], wav.numpy().astype(np.float32))
        )
    return bank


def sample_trigger(bank: dict, sample_type: str, idx: int = 0) -> np.ndarray:
    items = bank.get(sample_type, [])
    if not items:
        return np.zeros((2, 0), dtype=np.float32)
    return items[idx % len(items)][1]


def overlay_sample(host: np.ndarray, sample: np.ndarray, at_sample_idx: int,
                   gain: float = 0.7) -> np.ndarray:
    """Mix sample on top of host audio at given index. In-place safe via copy."""
    if sample.shape[1] == 0 or at_sample_idx >= host.shape[1]:
        return host
    end = min(at_sample_idx + sample.shape[1], host.shape[1])
    seg = sample[:, :end - at_sample_idx] * gain
    out = host.astype(np.float32).copy()
    out[:, at_sample_idx:end] += seg
    return np.clip(out, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Extended catalog (catalog.json status='implemented' upgrades).
# All accept stereo (2, T), return full stitched audio, same SR convention.
# ---------------------------------------------------------------------------


def _split_lo_hi(x: np.ndarray, sr: int, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-band split. Returns (low, high) such that low + high ≈ x."""
    lo = lp_filter(x, sr, cutoff)
    hi = hp_filter(x, sr, cutoff)
    return lo, hi


def bass_swap_transition(out_full: np.ndarray, in_full: np.ndarray,
                          sr: int, bars: int, beat_dur: float,
                          cutoff: float = 200.0) -> np.ndarray:
    """EQ Mixing — only the BASS swaps. Highs keep flowing from outgoing
    throughout the overlap. Avoids low-end mud without changing top-end.
    """
    n = int(bars * 4 * beat_dur * sr)
    overlap_n = min(n, out_full.shape[1], in_full.shape[1])
    if overlap_n <= 0:
        return cut_transition(out_full, in_full)
    out_tail = out_full[:, -overlap_n:]
    in_head = in_full[:, :overlap_n]
    out_lo, out_hi = _split_lo_hi(out_tail, sr, cutoff)
    in_lo, _ = _split_lo_hi(in_head, sr, cutoff)
    t = np.linspace(0.0, 1.0, overlap_n, dtype=np.float32)
    bass = out_lo * (1.0 - t) + in_lo * t
    overlap = bass + out_hi
    body = out_full[:, :-overlap_n] if out_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    rest = in_full[:, overlap_n:] if in_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    return np.concatenate([body, overlap.astype(np.float32), rest], axis=1)


def highs_swap_transition(out_full: np.ndarray, in_full: np.ndarray,
                           sr: int, bars: int, beat_dur: float,
                           cutoff: float = 4000.0) -> np.ndarray:
    """Top-end (cymbals/hihats) swap; lows + mids continue from outgoing."""
    n = int(bars * 4 * beat_dur * sr)
    overlap_n = min(n, out_full.shape[1], in_full.shape[1])
    if overlap_n <= 0:
        return cut_transition(out_full, in_full)
    out_tail = out_full[:, -overlap_n:]
    in_head = in_full[:, :overlap_n]
    out_lo, out_hi = _split_lo_hi(out_tail, sr, cutoff)
    _, in_hi = _split_lo_hi(in_head, sr, cutoff)
    t = np.linspace(0.0, 1.0, overlap_n, dtype=np.float32)
    highs = out_hi * (1.0 - t) + in_hi * t
    overlap = highs + out_lo
    body = out_full[:, :-overlap_n] if out_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    rest = in_full[:, overlap_n:] if in_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    return np.concatenate([body, overlap.astype(np.float32), rest], axis=1)


def highpass_sweep_in_transition(out_full: np.ndarray, in_full: np.ndarray,
                                  sr: int, bars: int, beat_dur: float,
                                  min_cutoff: float = 400.0,
                                  max_cutoff: float = 20.0) -> np.ndarray:
    """Mirror of filter_fade. Incoming enters HIGHPASSED (top-end only) and
    sweeps DOWN to full spectrum. Builds anticipation into a drop / chorus.
    """
    n = int(bars * 4 * beat_dur * sr)
    overlap_n = min(n, out_full.shape[1], in_full.shape[1])
    if overlap_n <= 0:
        return cut_transition(out_full, in_full)
    in_head = in_full[:, :overlap_n]
    chunk = max(1, int(beat_dur * sr / 2))     # 1/8-note chunks
    pieces = []
    n_chunks = max(1, overlap_n // chunk)
    for i in range(n_chunks):
        # cutoff sweeps from min_cutoff (heavy hp) toward max_cutoff (open)
        # max_cutoff < min_cutoff numerically, so interpolate inverted
        frac = i / max(1, n_chunks - 1)
        cutoff = min_cutoff * (1.0 - frac) + max_cutoff * frac
        cutoff = max(20.0, min(sr * 0.45, cutoff))
        seg = in_head[:, i * chunk:(i + 1) * chunk]
        if seg.shape[1] == 0:
            continue
        if cutoff > 50.0:
            seg = hp_filter(seg, sr, cutoff)
        pieces.append(seg)
    if not pieces:
        return cut_transition(out_full, in_full)
    swept = np.concatenate(pieces, axis=1)[:, :overlap_n]
    if swept.shape[1] < overlap_n:
        swept = np.pad(swept, ((0, 0), (0, overlap_n - swept.shape[1])))
    out_tail = out_full[:, -overlap_n:]
    t = np.linspace(0.0, np.pi / 2.0, overlap_n, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    overlap = out_tail * fade_out + swept * fade_in
    body = out_full[:, :-overlap_n] if out_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    rest = in_full[:, overlap_n:] if in_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    return np.concatenate([body, overlap.astype(np.float32), rest], axis=1)


def punch_in_transition(out_full: np.ndarray, in_full: np.ndarray,
                         sr: int = SR, anti_click_ms: float = 5.0) -> np.ndarray:
    """Hard cut on the 1 with anti-click ramps (~5 ms each side). Like cut
    but eliminates the discontinuity-pop at non-zero-crossing splices.
    """
    n_click = max(1, int(anti_click_ms * sr / 1000.0))
    n_click = min(n_click, out_full.shape[1], in_full.shape[1])
    if n_click < 4:
        return cut_transition(out_full, in_full)
    out_tail = out_full[:, -n_click:].astype(np.float32) * np.linspace(1.0, 0.0, n_click, dtype=np.float32)
    in_head = in_full[:, :n_click].astype(np.float32) * np.linspace(0.0, 1.0, n_click, dtype=np.float32)
    body = out_full[:, :-n_click]
    rest = in_full[:, n_click:]
    junction = out_tail + in_head
    return np.concatenate([body, junction, rest], axis=1)


def chop_transition(out_full: np.ndarray, in_full: np.ndarray,
                     sr: int, beat_dur: float,
                     n_chops: int = 4, period_beats: float = 0.5) -> np.ndarray:
    """Rapid alternation between outgoing and incoming. Each chop is
    `period_beats` long. Total chop region = n_chops * period_beats. After
    the last chop, incoming continues full. Limit to 1-2 per set.
    """
    period = max(1, int(period_beats * beat_dur * sr))
    chop_total = period * n_chops
    if chop_total >= out_full.shape[1] or chop_total >= in_full.shape[1]:
        return cut_transition(out_full, in_full)
    body = out_full[:, :-chop_total]
    pieces = []
    for i in range(n_chops):
        if i % 2 == 0:
            # Pull from outgoing (advancing within its tail)
            src_start = out_full.shape[1] - chop_total + i * period
            chop = out_full[:, src_start:src_start + period]
        else:
            # Pull from incoming
            in_idx = (i // 2) * period * 2 + period   # interleave
            chop = in_full[:, in_idx:in_idx + period] if in_idx < in_full.shape[1] else None
        if chop is None or chop.shape[1] == 0:
            continue
        # 1ms ramps on each chop edge to avoid clicks
        ramp = max(1, int(0.001 * sr))
        if chop.shape[1] > 2 * ramp:
            chop = chop.astype(np.float32, copy=True)
            chop[:, :ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
            chop[:, -ramp:] *= np.linspace(1.0, 0.0, ramp, dtype=np.float32)
        pieces.append(chop)
    if not pieces:
        return cut_transition(out_full, in_full)
    chop_region = np.concatenate(pieces, axis=1)
    rest = in_full[:, chop_total:]
    return np.concatenate([body, chop_region.astype(np.float32), rest], axis=1)


def loop_roll_transition(out_full: np.ndarray, in_full: np.ndarray,
                          sr: int, beat_dur: float,
                          steps: int = 4) -> np.ndarray:
    """Progressive halving: 1/2 beat → 1/4 → 1/8 → 1/16 over outgoing's
    tail, then hard handoff to incoming. More aggressive than loop_tighten.
    """
    durations_beats = [0.5 / (2 ** i) for i in range(steps)]
    durations_samples = [max(1, int(d * beat_dur * sr)) for d in durations_beats]
    total = sum(durations_samples)
    if total >= out_full.shape[1]:
        return cut_transition(out_full, in_full)
    body = out_full[:, :-total]
    seed = out_full[:, -durations_samples[0]:].astype(np.float32)
    pieces = []
    for d in durations_samples:
        # Take the LAST d samples of seed and ramp them
        loop = seed[:, -d:].astype(np.float32, copy=True)
        ramp = max(1, int(0.001 * sr))
        if loop.shape[1] > 2 * ramp:
            loop[:, :ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
            loop[:, -ramp:] *= np.linspace(1.0, 0.0, ramp, dtype=np.float32)
        pieces.append(loop)
    rolled = np.concatenate(pieces, axis=1)
    return np.concatenate([body, rolled.astype(np.float32),
                            in_full.astype(np.float32)], axis=1)


def beat_juggle_transition(out_full: np.ndarray, in_full: np.ndarray,
                            sr: int, beat_dur: float,
                            n_juggles: int = 2,
                            period_beats: float = 1.0) -> np.ndarray:
    """Alternate `period_beats`-long loops between outgoing and incoming
    `n_juggles` times. Each side plays its own slice in turn. Then full
    handoff to incoming.
    """
    period = max(1, int(period_beats * beat_dur * sr))
    juggle_total = period * n_juggles * 2
    if juggle_total >= out_full.shape[1] or juggle_total >= in_full.shape[1]:
        return cut_transition(out_full, in_full)
    body = out_full[:, :-period]
    pieces = []
    for i in range(n_juggles * 2):
        if i % 2 == 0:
            chop = out_full[:, -period:]
        else:
            chop = in_full[:, :period]
        ramp = max(1, int(0.001 * sr))
        c = chop.astype(np.float32, copy=True)
        if c.shape[1] > 2 * ramp:
            c[:, :ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
            c[:, -ramp:] *= np.linspace(1.0, 0.0, ramp, dtype=np.float32)
        pieces.append(c)
    return np.concatenate([body] + pieces + [in_full.astype(np.float32)], axis=1)


def acapella_drop_transition(out_full: np.ndarray, out_vox: np.ndarray,
                              in_full: np.ndarray, sr: int,
                              vocal_only_bars: int = 4,
                              beat_dur: float = 0.5) -> np.ndarray:
    """Strip outgoing to vocals only over N bars, then incoming drops on
    the 1. `out_vox` is the pre-computed vocal stem of outgoing.
    """
    vocal_n = int(vocal_only_bars * 4 * beat_dur * sr)
    vocal_n = min(vocal_n, out_full.shape[1], out_vox.shape[1])
    if vocal_n <= 0:
        return cut_transition(out_full, in_full)
    body = out_full[:, :-vocal_n] if out_full.shape[1] > vocal_n else np.zeros((2, 0), dtype=np.float32)
    vox_only = out_vox[:, -vocal_n:].astype(np.float32)
    return np.concatenate([body, vox_only, in_full.astype(np.float32)], axis=1)


def instrumental_swap_transition(out_full: np.ndarray, out_vox: np.ndarray,
                                  in_inst: np.ndarray, sr: int, bars: int,
                                  beat_dur: float) -> np.ndarray:
    """Mirror of mashup. Outgoing's vocals continue while incoming's
    instrumental backing replaces outgoing's. Vocals on top of new bed.
    """
    n_overlay = int(bars * 4 * beat_dur * sr)
    n_overlay = min(n_overlay, out_vox.shape[1], in_inst.shape[1])
    if n_overlay <= 0 or out_full.shape[1] < n_overlay:
        return cut_transition(out_full, in_full=in_inst)
    body = out_full[:, :-n_overlay] if out_full.shape[1] > n_overlay else np.zeros((2, 0), dtype=np.float32)
    vox_tail = out_vox[:, -n_overlay:].astype(np.float32)
    inst_head = in_inst[:, :n_overlay].astype(np.float32)
    overlay = vox_tail + inst_head
    rest_inst = in_inst[:, n_overlay:] if in_inst.shape[1] > n_overlay else np.zeros((2, 0), dtype=np.float32)
    return np.concatenate([body, np.clip(overlay, -1.0, 1.0), rest_inst.astype(np.float32)], axis=1)


def kickless_swap_transition(out_full: np.ndarray, out_drums: np.ndarray,
                              in_full: np.ndarray, in_drums: np.ndarray,
                              sr: int, bars: int, beat_dur: float) -> np.ndarray:
    """Remove kick (low-band of drums) on both during overlap. Re-add full
    drums on incoming downbeat. Avoids competing kick patterns.
    """
    n = int(bars * 4 * beat_dur * sr)
    overlap_n = min(n, out_full.shape[1], in_full.shape[1])
    if overlap_n <= 0 or out_drums.shape[1] < overlap_n or in_drums.shape[1] < overlap_n:
        return crossfade_transition(out_full, in_full, sr, bars, beat_dur)
    out_tail = out_full[:, -overlap_n:].astype(np.float32)
    in_head = in_full[:, :overlap_n].astype(np.float32)
    # Remove kick band (sub-130 Hz) from drum stems → subtract from full mix
    out_kick = lp_filter(out_drums[:, -overlap_n:], sr, 130.0)
    in_kick = lp_filter(in_drums[:, :overlap_n], sr, 130.0)
    out_no_kick = out_tail - out_kick
    in_no_kick = in_head - in_kick
    t = np.linspace(0.0, np.pi / 2.0, overlap_n, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    overlap = out_no_kick * fade_out + in_no_kick * fade_in
    body = out_full[:, :-overlap_n] if out_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    rest = in_full[:, overlap_n:] if in_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    return np.concatenate([body, overlap.astype(np.float32), rest], axis=1)


def drum_replace_transition(out_full: np.ndarray, out_drums: np.ndarray,
                             in_full: np.ndarray, in_drums: np.ndarray,
                             sr: int, bars: int, beat_dur: float) -> np.ndarray:
    """Swap drum stem from outgoing to incoming early; non-drum stems of
    outgoing continue, then full handoff to incoming. Tempo-match assumed.
    """
    n = int(bars * 4 * beat_dur * sr)
    overlap_n = min(n, out_full.shape[1], in_full.shape[1])
    if overlap_n <= 0 or out_drums.shape[1] < overlap_n or in_drums.shape[1] < overlap_n:
        return crossfade_transition(out_full, in_full, sr, bars, beat_dur)
    # Outgoing minus its own drums = "inst-no-drums" backing
    out_tail = out_full[:, -overlap_n:].astype(np.float32)
    out_drums_tail = out_drums[:, -overlap_n:].astype(np.float32)
    in_drums_head = in_drums[:, :overlap_n].astype(np.float32)
    backing = out_tail - out_drums_tail
    # Half overlap: backing + incoming drums; second half: full incoming
    half = overlap_n // 2
    if half <= 0:
        return cut_transition(out_full, in_full)
    swap_region = np.concatenate([
        backing[:, :half] + in_drums_head[:, :half],
        in_full[:, half:overlap_n].astype(np.float32),
    ], axis=1)
    body = out_full[:, :-overlap_n] if out_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    rest = in_full[:, overlap_n:] if in_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    return np.concatenate([body, np.clip(swap_region, -1.0, 1.0), rest], axis=1)


def reverb_wash_transition(out_full: np.ndarray, in_full: np.ndarray,
                            sr: int, bars: int, beat_dur: float,
                            decay_sec: float = 2.5,
                            wet_gain: float = 0.6) -> np.ndarray:
    """Outgoing tail soaked in long reverb (synthetic exp-decay convolution)
    as a bridge. Atmospheric / cinematic.
    """
    n = int(bars * 4 * beat_dur * sr)
    overlap_n = min(n, out_full.shape[1], in_full.shape[1])
    if overlap_n <= 0:
        return cut_transition(out_full, in_full)
    # Build a quick exponential-decay impulse response (synthetic reverb)
    ir_n = max(1, int(decay_sec * sr))
    rng = np.random.default_rng(42)
    ir_mono = rng.standard_normal(ir_n).astype(np.float32)
    ir_mono *= np.exp(-np.linspace(0, 6, ir_n, dtype=np.float32))
    ir_mono /= max(1e-6, np.linalg.norm(ir_mono))
    out_tail = out_full[:, -overlap_n:].astype(np.float32)
    # Convolve each channel
    wet = np.stack([
        np.convolve(out_tail[c], ir_mono, mode='same') for c in range(out_tail.shape[0])
    ]).astype(np.float32)
    t = np.linspace(0.0, np.pi / 2.0, overlap_n, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    overlap = (out_tail * fade_out
                + wet * (wet_gain * fade_out)
                + in_full[:, :overlap_n].astype(np.float32) * fade_in)
    body = out_full[:, :-overlap_n] if out_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    rest = in_full[:, overlap_n:] if in_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    return np.concatenate([body, np.clip(overlap, -1.0, 1.0), rest], axis=1)


def forward_spin_transition(out_full: np.ndarray, in_full: np.ndarray,
                             sr: int, beat_dur: float,
                             accel_beats: float = 1.0,
                             max_pitch_ratio: float = 1.6) -> np.ndarray:
    """Pitch-up acceleration of outgoing into incoming hard cut. Mirror of
    spinback. Plays last `accel_beats` at progressively higher pitch.
    """
    n = max(1, int(accel_beats * beat_dur * sr))
    n = min(n, out_full.shape[1])
    body = out_full[:, :-n] if out_full.shape[1] > n else np.zeros((2, 0), dtype=np.float32)
    src = out_full[:, -n:].astype(np.float32)
    out_n = max(1, int(n / max_pitch_ratio))
    idx = np.linspace(0, n - 1, out_n).astype(np.int64)
    spun = src[:, idx]
    # Quick fade-out to avoid click into incoming
    ramp_n = max(1, int(0.005 * sr))
    if spun.shape[1] > ramp_n:
        spun[:, -ramp_n:] *= np.linspace(1.0, 0.0, ramp_n, dtype=np.float32)
    return np.concatenate([body, spun, in_full.astype(np.float32)], axis=1)


def tape_stop_transition(out_full: np.ndarray, in_full: np.ndarray,
                          sr: int, beat_dur: float,
                          slow_beats: float = 1.0,
                          curve: str = 'exp') -> np.ndarray:
    """Vari-speed slowdown that drops pitch + tempo to zero over `slow_beats`.
    More extreme than spinback. `curve`='exp' for exponential decay,
    'linear' for constant deceleration.
    """
    n = max(1, int(slow_beats * beat_dur * sr))
    n = min(n, out_full.shape[1])
    body = out_full[:, :-n] if out_full.shape[1] > n else np.zeros((2, 0), dtype=np.float32)
    src = out_full[:, -n:].astype(np.float32)
    if curve == 'exp':
        # Exponentially decreasing rate
        rate = np.exp(np.linspace(0.0, -3.0, n, dtype=np.float32))
    else:
        rate = np.linspace(1.0, 0.05, n, dtype=np.float32)
    pos = np.cumsum(rate)
    pos = pos / pos[-1] * (n - 1)
    idx = pos.astype(np.int64)
    stopped = src[:, idx]
    # Fade tail down to silence to seal the slowdown
    fade = np.linspace(1.0, 0.0, n, dtype=np.float32)
    stopped = stopped * fade
    return np.concatenate([body, stopped, in_full.astype(np.float32)], axis=1)


def spectral_hold_transition(out_full: np.ndarray, in_full: np.ndarray,
                              sr: int, bars: int, beat_dur: float,
                              hold_bars: int = 2) -> np.ndarray:
    """Freeze harmonic content of outgoing's last bar (FFT magnitude held
    over `hold_bars`) while drums change underneath. Glue layer for
    cross-genre jumps. Cheap rule-based phase-cancel mitigation.
    """
    bar_n = int(4 * beat_dur * sr)
    hold_n = min(hold_bars * bar_n, out_full.shape[1] // 2)
    if hold_n <= 0:
        return cut_transition(out_full, in_full)
    # Take last bar of outgoing as the spectral seed
    seed_n = min(bar_n, out_full.shape[1])
    seed = out_full[:, -seed_n:].astype(np.float32)
    # Build held audio by repeating the seed `hold_bars` times with random
    # phase to avoid perfect periodicity (which would sound mechanical).
    rng = np.random.default_rng(0)
    pieces = []
    for _ in range(max(1, hold_n // seed_n)):
        # Apply a small random phase shift via tiny random rotation in time
        shift = rng.integers(0, max(1, seed_n // 32))
        rolled = np.roll(seed, int(shift), axis=1)
        pieces.append(rolled.astype(np.float32))
    held = np.concatenate(pieces, axis=1)[:, :hold_n]
    # Crossfade out the hold and crossfade in the incoming
    overlap_n = min(hold_n, in_full.shape[1])
    body = out_full[:, :-seed_n] if out_full.shape[1] > seed_n else np.zeros((2, 0), dtype=np.float32)
    t = np.linspace(0.0, np.pi / 2.0, overlap_n, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    bridge = held[:, :overlap_n] * fade_out + in_full[:, :overlap_n].astype(np.float32) * fade_in
    rest = in_full[:, overlap_n:] if in_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    return np.concatenate([body, bridge.astype(np.float32), rest], axis=1)


def bpm_warp_transition(out_full: np.ndarray, in_full: np.ndarray,
                         sr: int, bars: int, beat_dur: float,
                         out_bpm: float, in_bpm: float) -> np.ndarray:
    """Gradual BPM blend over `bars` for cross-tempo bridging. Uses simple
    resample (rate-warp) rather than pyrubberband (cheap, slight pitch
    coupling — acceptable for short bridges). Outgoing slows / incoming
    speeds toward midpoint, then incoming continues at its own rate.
    """
    if out_bpm <= 0 or in_bpm <= 0 or abs(out_bpm - in_bpm) < 1.0:
        return crossfade_transition(out_full, in_full, sr, bars, beat_dur)
    overlap_n = min(int(bars * 4 * beat_dur * sr),
                     out_full.shape[1], in_full.shape[1])
    if overlap_n <= 0:
        return cut_transition(out_full, in_full)
    midpoint = (out_bpm + in_bpm) / 2.0
    # Outgoing speed ramp 1.0 → midpoint/out_bpm
    out_target_ratio = midpoint / out_bpm
    in_source_ratio = midpoint / in_bpm
    out_tail = out_full[:, -overlap_n:].astype(np.float32)
    in_head = in_full[:, :overlap_n].astype(np.float32)
    # Resample by linear interp with a varying rate
    out_rates = np.linspace(1.0, out_target_ratio, overlap_n, dtype=np.float32)
    out_pos = np.cumsum(out_rates)
    out_pos *= (overlap_n - 1) / out_pos[-1]
    out_idx = np.clip(out_pos.astype(np.int64), 0, overlap_n - 1)
    out_warped = out_tail[:, out_idx]
    in_rates = np.linspace(in_source_ratio, 1.0, overlap_n, dtype=np.float32)
    in_pos = np.cumsum(in_rates)
    in_pos *= (overlap_n - 1) / in_pos[-1]
    in_idx = np.clip(in_pos.astype(np.int64), 0, overlap_n - 1)
    in_warped = in_head[:, in_idx]
    t = np.linspace(0.0, np.pi / 2.0, overlap_n, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    overlap = out_warped * fade_out + in_warped * fade_in
    body = out_full[:, :-overlap_n] if out_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    rest = in_full[:, overlap_n:] if in_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    return np.concatenate([body, overlap.astype(np.float32), rest], axis=1)


def harmonic_overlay_transition(out_full: np.ndarray, in_full: np.ndarray,
                                 sr: int, bars: int, beat_dur: float,
                                 pad_freqs: tuple[float, ...] = (220.0, 277.0, 330.0),
                                 pad_gain: float = 0.15) -> np.ndarray:
    """Sustained pad layer on bridge between tracks. Synthesizes a major
    chord drone (defaults A3-C#4-E4) over the overlap; both tracks
    crossfade underneath. Glue for genre-jump or key-jump junctions.
    """
    overlap_n = min(int(bars * 4 * beat_dur * sr),
                     out_full.shape[1], in_full.shape[1])
    if overlap_n <= 0:
        return cut_transition(out_full, in_full)
    t_axis = np.arange(overlap_n, dtype=np.float32) / sr
    pad_mono = np.zeros(overlap_n, dtype=np.float32)
    for f in pad_freqs:
        pad_mono += np.sin(2 * np.pi * f * t_axis)
    pad_mono /= max(1, len(pad_freqs))
    # Long attack/release envelope so the pad doesn't pop in
    env = np.ones(overlap_n, dtype=np.float32)
    attack_n = min(overlap_n // 4, int(beat_dur * sr * 2))
    if attack_n > 0:
        env[:attack_n] = np.linspace(0.0, 1.0, attack_n, dtype=np.float32)
        env[-attack_n:] = np.linspace(1.0, 0.0, attack_n, dtype=np.float32)
    pad_mono = pad_mono * env * pad_gain
    pad = np.stack([pad_mono, pad_mono])
    out_tail = out_full[:, -overlap_n:].astype(np.float32)
    in_head = in_full[:, :overlap_n].astype(np.float32)
    t = np.linspace(0.0, np.pi / 2.0, overlap_n, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    overlap = out_tail * fade_out + in_head * fade_in + pad
    body = out_full[:, :-overlap_n] if out_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    rest = in_full[:, overlap_n:] if in_full.shape[1] > overlap_n else np.zeros((2, 0), dtype=np.float32)
    return np.concatenate([body, np.clip(overlap, -1.0, 1.0), rest], axis=1)
