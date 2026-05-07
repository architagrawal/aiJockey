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


def equal_power_xfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """
    Equal-power crossfade. a fades out, b fades in over n samples of overlap.
    Output length = a.shape[1] + b.shape[1] - n.
    """
    n = min(n, a.shape[1], b.shape[1])
    if n <= 0:
        return np.concatenate([a, b], axis=1)
    t = np.linspace(0, np.pi / 2, n)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    out_len = a.shape[1] + b.shape[1] - n
    out = np.zeros((a.shape[0], out_len), dtype=np.float32)
    pre = a.shape[1] - n
    if pre > 0:
        out[:, :pre] = a[:, :pre]
    out[:, pre:pre + n] = a[:, -n:] * fade_out + b[:, :n] * fade_in
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
