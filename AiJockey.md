# AiJockey — AI DJ Set Generator

Single source of truth. Architecture + working code + quality strategy + MI300X plan.

---

## TL;DR

- **Input**: pool of audio clips (5-50, 30s-5min each)
- **Output**: cohesive DJ set (5-30 min) with pro-level mix engineering
- **Approach**: subset-select + non-sequential reorder + craft transitions + master
- **Phase A (3mo)**: rule-based pipeline, no generation
- **Phase B (+2mo)**: generative fills via Stable Audio Open for hard transitions
- **Hardware**: dev on laptop, MI300X for batch + Phase B fine-tune
- **Quality target**: blind A/B beats djay Pro auto-mix; approaches human DJ on transition quality
- **Honest ceiling**: not Daft Punk. Decades of craft + signature production cannot be matched. Realistic = "competent professional DJ doing a curated set."

---

## Quality strategy — where investment matters

Most auto-DJ tools fail because they treat mixing as "crossfade between songs." Pro DJs:

1. **Phrase-align** — transitions on 16/32-bar boundaries, not just downbeats
2. **Energy-match** — never crossfade peak into intro; build then drop
3. **Frequency-swap** — kill outgoing bass + raise incoming bass (avoid mud)
4. **Stem-tricks** — a-cappella overlay, drum-only breaks, percussive risers
5. **Tension-release** — silence drops, snare rolls, filter sweeps
6. **Callbacks** — repeat earlier hook later for cohesion
7. **Master to club LUFS** (-9 to -7) with side-chain ducking on bass

Investment priority (where quality is won/lost):

| Layer | Investment | Why |
|-------|------------|-----|
| Stem separation | high (Demucs v4 htdemucs_ft) | unlocks all stem-tricks |
| Beat + downbeat detection | high (madmom + multi-algo cross-check) | wrong beat = unusable |
| Phrase boundary detection | **highest** (most auto-DJs skip this) | distinguishes pro from amateur |
| Key detection + Camelot | medium (librosa + Krumhansl) | wrong key = clash |
| Transition library | **highest** (parametric, 15+ techniques) | this IS the product |
| Sample bank | medium (one-shots: risers, impacts, vinyl FX) | enables spinback, scratching, impacts |
| Phrase enforcement | **highest** (transitions ONLY on 16/32-bar grid) | most auto-DJs ignore phrase, sounds amateur |
| Mastering chain | high (multiband + sidechain + LUFS norm) | club playback ready |
| Planner scoring | medium-high (tunable weights, eval-driven) | iterate via metrics |

---

## Architecture

```
clips/ (M clips)
  │
  ▼
[ANALYZE] per clip → cached features (.npz + .json)
  - Demucs v4 stems: vocals, drums, bass, other
  - madmom: beats, downbeats, tempo (cross-check librosa)
  - librosa: key (Krumhansl), structure boundaries
  - hook detection: self-similarity matrix → recurring 4/8-bar loops
  - energy curve: RMS over time, smoothed
  - phrase grid: downbeats grouped into 4/8/16/32-bar phrases
  - CLAP embedding: 512-d vector for timbre/genre similarity
  │
  ▼
[PLAN] beam search → timeline.json
  - target: duration, energy arc (warmup→peak→cooldown), surprise budget
  - subset-select + order: M clips → K used in arbitrary order
  - per clip: which SEGMENT (intro 16 bars, drop 32 bars, outro 8 bars)
  - per transition: TECHNIQUE from full library (15+) — see Transitions section
  - schedule callbacks: repeat earlier hook at planned moment
  - enforce: transitions land on 16- or 32-bar phrase boundaries only
  │
  ▼
[EXECUTE] render timeline → raw_mix.wav
  - per clip segment: load stems
  - tempo-sync: rubberband to target BPM
  - pitch-shift: rubberband to compatible key
  - apply transition technique at planned bar
  - mix stems with per-stem volume + EQ automation
  │
  ▼
[MASTER] raw_mix → final_mix.wav
  - high-pass at 30Hz (remove sub-rumble)
  - multiband compression (3 bands)
  - sidechain ducking on bass triggered by kick
  - true-peak limiter
  - LUFS normalize to -9 (club target)
  │
  ▼
final_mix.wav + timeline.json (auditable)
```

---

## Repo layout

```
aijockey/
├── clips/                        # input audio
├── cache/                        # analysis features per clip
├── output/                       # mix.wav + timeline.json
├── src/
│   ├── analyze.py               # stem sep + beat + key + structure + CLAP
│   ├── hooks.py                 # self-similarity hook detection
│   ├── camelot.py               # key compatibility logic
│   ├── planner.py               # beam search set construction
│   ├── transitions.py           # 7 transition techniques
│   ├── execute.py               # render timeline to raw_mix
│   ├── master.py                # mastering chain
│   ├── eval.py                  # objective metrics
│   └── main.py                  # CLI orchestrator
├── tests/
│   └── fixtures/                # 5 test clips
└── pyproject.toml
```

---

## Environment

```bash
python -m venv .venv && source .venv/bin/activate

# CPU dev (laptop)
pip install torch==2.3.1 torchaudio==2.3.1

# OR MI300X (ROCm 6.0)
# pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/rocm6.0

pip install demucs==4.0.1
pip install madmom==0.16.1 librosa==0.10.2 soundfile==0.12.1
pip install pyrubberband==0.3.0  # requires rubberband CLI installed (apt: rubberband-cli)
pip install pyloudnorm==0.1.1    # LUFS normalization
pip install scipy numpy tqdm
pip install laion-clap==1.1.4    # CLAP embeddings (audio-text)
pip install msaf==0.1.80         # structure segmentation
```

System: `apt install rubberband-cli ffmpeg`

---

## ROCm sanity (first MI300X session, budget $5)

```python
# scripts/00_rocm_sanity.py
import torch
print(f"torch {torch.__version__}, hip={torch.version.hip}, devices={torch.cuda.device_count()}")
# Demucs
from demucs.pretrained import get_model
m = get_model('htdemucs')
m.cuda()
import torchaudio
wav, sr = torchaudio.load('clips/test.wav')
wav = wav.cuda().unsqueeze(0)
with torch.no_grad():
    stems = m(wav)
print(f"stems shape: {stems.shape}")  # (1, 4, 2, T)
# CLAP
from laion_clap import CLAP_Module
c = CLAP_Module(enable_fusion=False); c.load_ckpt()
emb = c.get_audio_embedding_from_data(wav.squeeze(0).mean(0, keepdim=True).cpu().numpy(), use_tensor=False)
print(f"clap emb shape: {emb.shape}")  # (1, 512)
```

If anything fails: stop, fix ROCm setup, do not proceed.

---

## Layer 1 — Analyze

```python
# src/analyze.py
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
import torchaudio
import librosa
from demucs.pretrained import get_model as get_demucs
from demucs.apply import apply_model
import madmom
from laion_clap import CLAP_Module

SR = 44100  # working sample rate (matches Demucs)


@dataclass
class ClipAnalysis:
    clip_id: str
    path: str
    duration: float
    sample_rate: int
    tempo: float
    beats: list           # downbeats AND on-beats, in seconds
    downbeats: list       # in seconds
    key: str              # e.g. "8A" Camelot
    key_confidence: float
    sections: list        # [{type, start, end, energy}]
    hooks: list           # [{start, end, repetition_count, strength}]
    energy_curve: list    # RMS samples at 10Hz
    clap_embedding: list  # 512-d


class Analyzer:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.demucs = get_demucs('htdemucs_ft').to(device)
        self.demucs.eval()
        self.clap = CLAP_Module(enable_fusion=False)
        self.clap.load_ckpt()  # downloads weights first time
        self.beat_proc = madmom.features.beats.RNNBeatProcessor()
        self.beat_track = madmom.features.beats.BeatTrackingProcessor(fps=100)
        self.dbn_proc = madmom.features.downbeats.RNNDownBeatProcessor()
        self.dbn_track = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4], fps=100)

    def load(self, path: str) -> tuple[torch.Tensor, int]:
        wav, sr = torchaudio.load(path)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.size(0) == 1:
            wav = wav.repeat(2, 1)  # demucs expects stereo
        return wav, SR

    def stems(self, wav: torch.Tensor) -> dict:
        with torch.no_grad():
            x = wav.unsqueeze(0).to(self.device)
            sources = apply_model(self.demucs, x, split=True, overlap=0.25)[0]
        names = self.demucs.sources  # ['drums','bass','other','vocals']
        return {n: sources[i].cpu() for i, n in enumerate(names)}

    def beats_and_downbeats(self, wav: torch.Tensor) -> tuple[float, list, list]:
        mono = wav.mean(0).numpy()
        # Beats
        beat_act = self.beat_proc(mono)
        beats = self.beat_track(beat_act).tolist()
        # Downbeats
        db_act = self.dbn_proc(mono)
        db_out = self.dbn_track(db_act)
        downbeats = [float(t) for t, b in db_out if int(b) == 1]
        # Tempo from median IBI
        if len(beats) > 1:
            ibis = np.diff(beats)
            tempo = 60.0 / float(np.median(ibis))
        else:
            tempo = 0.0
        return tempo, beats, downbeats

    def key_camelot(self, wav: torch.Tensor) -> tuple[str, float]:
        mono = wav.mean(0).numpy()
        chroma = librosa.feature.chroma_cqt(y=mono, sr=SR)
        chroma_mean = chroma.mean(axis=1)
        from camelot import krumhansl_key, to_camelot
        pitch_class, mode, conf = krumhansl_key(chroma_mean)
        return to_camelot(pitch_class, mode), float(conf)

    def sections(self, wav: torch.Tensor) -> list:
        # Use librosa segmentation as fallback (msaf can be flaky)
        mono = wav.mean(0).numpy()
        bounds = librosa.segment.agglomerative(
            librosa.feature.mfcc(y=mono, sr=SR), k=8)
        bound_times = librosa.frames_to_time(bounds, sr=SR).tolist()
        bound_times = [0.0] + bound_times + [len(mono) / SR]
        # Compute energy per section
        rms = librosa.feature.rms(y=mono)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=SR)
        out = []
        for i in range(len(bound_times) - 1):
            s, e = bound_times[i], bound_times[i+1]
            mask = (rms_times >= s) & (rms_times < e)
            energy = float(rms[mask].mean()) if mask.any() else 0.0
            out.append({'start': s, 'end': e, 'energy': energy})
        # Label sections by energy + position (heuristic)
        if out:
            energies = [s['energy'] for s in out]
            peak_e = max(energies)
            for i, s in enumerate(out):
                rel = s['energy'] / peak_e if peak_e > 0 else 0
                if i == 0 and rel < 0.5:
                    s['type'] = 'intro'
                elif i == len(out) - 1 and rel < 0.5:
                    s['type'] = 'outro'
                elif rel > 0.85:
                    s['type'] = 'drop'
                elif rel > 0.6:
                    s['type'] = 'verse'
                else:
                    s['type'] = 'breakdown'
        return out

    def energy_curve(self, wav: torch.Tensor, hop_hz: int = 10) -> list:
        mono = wav.mean(0).numpy()
        hop = SR // hop_hz
        rms = librosa.feature.rms(y=mono, frame_length=hop * 2, hop_length=hop)[0]
        return rms.tolist()

    def clap_embedding(self, wav: torch.Tensor) -> list:
        mono = wav.mean(0).numpy().astype(np.float32)
        # CLAP wants 48kHz
        if SR != 48000:
            mono = librosa.resample(mono, orig_sr=SR, target_sr=48000)
        emb = self.clap.get_audio_embedding_from_data(mono[None, :], use_tensor=False)
        return emb[0].tolist()

    def analyze(self, path: str, clip_id: str) -> ClipAnalysis:
        wav, sr = self.load(path)
        stems = self.stems(wav)
        # Save stems for later use in execute
        Path(f"cache/stems/{clip_id}").mkdir(parents=True, exist_ok=True)
        for name, s in stems.items():
            torchaudio.save(f"cache/stems/{clip_id}/{name}.wav", s, sr)

        tempo, beats, downbeats = self.beats_and_downbeats(wav)
        key, key_conf = self.key_camelot(wav)
        sections = self.sections(wav)
        energy = self.energy_curve(wav)
        from hooks import detect_hooks
        hook_list = detect_hooks(wav.mean(0).numpy(), sr, downbeats)
        clap = self.clap_embedding(wav)

        return ClipAnalysis(
            clip_id=clip_id,
            path=path,
            duration=wav.size(1) / sr,
            sample_rate=sr,
            tempo=tempo,
            beats=beats,
            downbeats=downbeats,
            key=key,
            key_confidence=key_conf,
            sections=sections,
            hooks=hook_list,
            energy_curve=energy,
            clap_embedding=clap,
        )


def analyze_pool(clip_dir: str, cache_dir: str, device: str = 'cuda'):
    analyzer = Analyzer(device=device)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(clip_dir).glob('*.wav')) + sorted(Path(clip_dir).glob('*.mp3')):
        clip_id = path.stem
        cache_path = Path(cache_dir) / f"{clip_id}.json"
        if cache_path.exists():
            continue
        print(f"analyzing {clip_id}")
        result = analyzer.analyze(str(path), clip_id)
        # Save (clap_embedding separate as npz for size)
        d = asdict(result)
        np.savez_compressed(Path(cache_dir) / f"{clip_id}.npz",
                            clap=np.array(d.pop('clap_embedding'), dtype=np.float32),
                            energy=np.array(d.pop('energy_curve'), dtype=np.float32))
        with open(cache_path, 'w') as f:
            json.dump(d, f, indent=2)


if __name__ == '__main__':
    import sys
    analyze_pool(sys.argv[1], sys.argv[2], device=sys.argv[3] if len(sys.argv) > 3 else 'cuda')
```

### Camelot key logic

```python
# src/camelot.py
import numpy as np

# Krumhansl-Schmuckler profiles
MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

# Camelot wheel mapping: (pitch_class, mode) -> code
CAMELOT = {
    (0,'maj'):'8B',(7,'maj'):'9B',(2,'maj'):'10B',(9,'maj'):'11B',(4,'maj'):'12B',
    (11,'maj'):'1B',(6,'maj'):'2B',(1,'maj'):'3B',(8,'maj'):'4B',(3,'maj'):'5B',
    (10,'maj'):'6B',(5,'maj'):'7B',
    (9,'min'):'8A',(4,'min'):'9A',(11,'min'):'10A',(6,'min'):'11A',(1,'min'):'12A',
    (8,'min'):'1A',(3,'min'):'2A',(10,'min'):'3A',(5,'min'):'4A',(0,'min'):'5A',
    (7,'min'):'6A',(2,'min'):'7A',
}


def krumhansl_key(chroma_mean: np.ndarray) -> tuple[int, str, float]:
    scores_maj = [np.corrcoef(np.roll(MAJ, i), chroma_mean)[0,1] for i in range(12)]
    scores_min = [np.corrcoef(np.roll(MIN, i), chroma_mean)[0,1] for i in range(12)]
    best_maj = int(np.argmax(scores_maj)); best_min = int(np.argmax(scores_min))
    if scores_maj[best_maj] >= scores_min[best_min]:
        return best_maj, 'maj', float(scores_maj[best_maj])
    return best_min, 'min', float(scores_min[best_min])


def to_camelot(pitch_class: int, mode: str) -> str:
    return CAMELOT.get((pitch_class, mode), '?')


def camelot_distance(a: str, b: str) -> int:
    """0=same, 1=adjacent or relative, 2=2 steps, ..."""
    if a == '?' or b == '?': return 6
    if a == b: return 0
    num_a, letter_a = int(a[:-1]), a[-1]
    num_b, letter_b = int(b[:-1]), b[-1]
    # Same letter, adjacent number
    if letter_a == letter_b:
        diff = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
        return diff
    # Different letter (maj/min relative)
    if num_a == num_b:
        return 1  # relative key, compatible
    # Both differ
    diff = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
    return diff + 1
```

### Hook detection

```python
# src/hooks.py
import numpy as np
import librosa


def detect_hooks(mono: np.ndarray, sr: int, downbeats: list,
                 min_bars: int = 4, max_bars: int = 16) -> list:
    """
    Find recurring segments via self-similarity matrix on chroma+MFCC.
    Returns list of {start, end, repetition_count, strength}.
    """
    if len(downbeats) < min_bars * 2:
        return []
    # Features per downbeat-aligned frame
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=13)
    feat = np.concatenate([chroma, mfcc], axis=0)
    # Pool to per-bar features
    times = librosa.frames_to_time(np.arange(feat.shape[1]), sr=sr)
    bar_feats = []
    for i in range(len(downbeats) - 1):
        s, e = downbeats[i], downbeats[i+1]
        mask = (times >= s) & (times < e)
        if mask.any():
            bar_feats.append(feat[:, mask].mean(axis=1))
    if len(bar_feats) < min_bars * 2:
        return []
    bar_feats = np.array(bar_feats)
    # Self-similarity matrix
    norm = bar_feats / (np.linalg.norm(bar_feats, axis=1, keepdims=True) + 1e-8)
    sim = norm @ norm.T
    # Find recurring N-bar segments
    hooks = []
    used = set()
    for L in range(min_bars, max_bars + 1, 4):
        for i in range(len(bar_feats) - L):
            if i in used: continue
            # Sum similarity of segment [i:i+L] vs all other [j:j+L]
            sims = []
            for j in range(i + L, len(bar_feats) - L):
                if j in used: continue
                s = sim[i:i+L, j:j+L].diagonal().mean()
                if s > 0.7:
                    sims.append((j, float(s)))
            if len(sims) >= 1:
                hooks.append({
                    'start': float(downbeats[i]),
                    'end': float(downbeats[i + L]),
                    'repetition_count': len(sims) + 1,
                    'strength': float(np.mean([s for _, s in sims])),
                    'bars': L,
                })
                used.update(range(i, i + L))
                for j, _ in sims:
                    used.update(range(j, j + L))
    # Sort by strength
    hooks.sort(key=lambda h: -h['strength'])
    return hooks[:5]
```

---

## Layer 2 — Planner (beam search)

```python
# src/planner.py
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import heapq

from camelot import camelot_distance


@dataclass
class TimelineEntry:
    clip_id: str
    segment: dict        # {start_sec, end_sec, type}
    target_bpm: float
    target_key: str      # camelot, may differ from clip key (we'll pitch-shift)
    transition_in: dict  # technique + params
    play_at: float = 0.0 # set during scheduling


@dataclass
class PlannerConfig:
    target_duration: float = 600.0   # 10 min
    energy_arc: list = field(default_factory=lambda: [0.3, 0.5, 0.8, 1.0, 0.9, 0.6, 0.3])
    surprise_budget: int = 1         # # of unexpected drops/jumps allowed
    callback_budget: int = 1         # # of hook-callbacks
    beam_width: int = 12
    max_clips: int = 20
    transition_min_bars: int = 16    # phrase-aligned
    weights: dict = field(default_factory=lambda: dict(
        key=0.25, tempo=0.20, energy=0.20, timbre=0.15,
        variety=0.10, surprise=0.10,
    ))


@dataclass
class State:
    sequence: list                   # list[TimelineEntry]
    cumulative_duration: float = 0.0
    used_clip_ids: set = field(default_factory=set)
    surprises_used: int = 0
    callbacks_used: int = 0
    score: float = 0.0


def load_clips(cache_dir: str) -> dict:
    out = {}
    for jp in Path(cache_dir).glob('*.json'):
        with open(jp) as f:
            d = json.load(f)
        npz = np.load(Path(cache_dir) / f"{jp.stem}.npz")
        d['clap'] = npz['clap']
        d['energy_arr'] = npz['energy']
        out[jp.stem] = d
    return out


def transition_score(prev: dict, prev_seg: dict, prev_target_bpm: float, prev_target_key: str,
                     cand: dict, cand_seg: dict, weights: dict) -> tuple[float, dict]:
    # Tempo: penalize stretch beyond ±8%
    cand_bpm_native = cand['tempo']
    target_bpm = prev_target_bpm  # try to keep continuity
    stretch = abs(target_bpm - cand_bpm_native) / max(cand_bpm_native, 1)
    tempo_score = max(0, 1 - stretch / 0.12)  # 0 at 12% stretch
    # Key: Camelot distance (we'll pitch-shift candidate to compatible key)
    # Allow ±1 step on Camelot wheel cheaply
    key_dist = camelot_distance(prev_target_key, cand['key'])
    key_score = max(0, 1 - key_dist / 4)
    # Energy continuity (incoming intro should match outgoing outro)
    out_energy = prev_seg.get('energy', 0.5)
    in_energy = cand_seg.get('energy', 0.5)
    energy_score = 1 - min(1, abs(out_energy - in_energy) * 2)
    # Timbre similarity via CLAP cosine
    a = np.asarray(prev['clap']); b = np.asarray(cand['clap'])
    timbre_score = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    # Variety: penalty if very similar (we want some change)
    variety_score = 1 - timbre_score if timbre_score > 0.95 else 1.0
    # Aggregate
    score = (weights['key'] * key_score + weights['tempo'] * tempo_score
             + weights['energy'] * energy_score + weights['timbre'] * timbre_score
             + weights['variety'] * variety_score)
    # Pick technique by context. Priority decision tree.
    # 1. Hard incompatibility → neutralizer or echo-out / spinback
    if key_score < 0.3 and tempo_score < 0.4:
        tech = {'name': 'echo_out', 'bars': 8, 'delay_beats': 0.5, 'feedback': 0.55}
    elif key_score < 0.4:
        # Try pitch_bend if small gap, else drum_break
        if 0.3 < key_score < 0.6:
            tech = {'name': 'pitch_bend', 'bars': 8, 'semitones': 1.0}
        else:
            tech = {'name': 'drum_break', 'bars': 8}
    # 2. Big-moment punctuation: huge energy jump down→up after sustained peak
    elif out_energy > 0.85 and in_energy > 0.85 and timbre_score < 0.5:
        tech = {'name': 'spinback', 'spinback_beats': 4}
    # 3. Climactic build into incoming drop
    elif in_energy > 0.85 and out_energy > 0.6:
        tech = {'name': 'loop_tighten', 'start_bars': 4}
    # 4. Tension-release: low → high energy
    elif in_energy > 0.85 and out_energy < 0.5:
        tech = {'name': 'silence_drop', 'bars': 4}
    # 5. Vocal-rich pair → mashup
    elif timbre_score > 0.7 and in_energy > 0.5 and out_energy > 0.5:
        tech = {'name': 'mashup', 'bars': 16}
    # 6. Mood shift / breakdown intro
    elif in_energy < out_energy - 0.2:
        tech = {'name': 'filter_fade', 'bars': 16}
    # 7. High-energy sustained continuity
    elif energy_score > 0.7 and tempo_score > 0.6 and in_energy > 0.7:
        tech = {'name': 'eq_swap', 'bars': 32}
    # 8. Default smooth blend
    elif energy_score > 0.5:
        tech = {'name': 'crossfade', 'bars': 16}
    else:
        tech = {'name': 'eq_swap', 'bars': 16}
    return score, tech


def pick_segment(clip: dict, prefer: str = None) -> dict:
    sections = clip.get('sections', [])
    if not sections:
        return {'start': 0.0, 'end': clip['duration'], 'type': 'unknown', 'energy': 0.5}
    if prefer:
        for s in sections:
            if s.get('type') == prefer:
                return s
    # Default: longest non-intro/outro
    body = [s for s in sections if s.get('type') in ('drop', 'verse', 'breakdown')]
    return max(body or sections, key=lambda s: s['end'] - s['start'])


def plan(clips: dict, config: PlannerConfig) -> list:
    # Initial states: each clip as start
    starts = []
    for cid, clip in clips.items():
        seg = pick_segment(clip, prefer='intro') or pick_segment(clip)
        target_bpm = clip['tempo']
        entry = TimelineEntry(
            clip_id=cid, segment=seg, target_bpm=target_bpm,
            target_key=clip['key'],
            transition_in={'name': 'fade_in', 'bars': 4},
        )
        st = State(sequence=[entry],
                   cumulative_duration=seg['end'] - seg['start'],
                   used_clip_ids={cid}, score=0.0)
        starts.append(st)
    # Beam search
    beam = sorted(starts, key=lambda s: -s.score)[:config.beam_width]
    finished = []
    while beam:
        next_beam = []
        for st in beam:
            if (st.cumulative_duration >= config.target_duration
                or len(st.sequence) >= config.max_clips):
                finished.append(st)
                continue
            # Where in arc are we?
            progress = st.cumulative_duration / config.target_duration
            arc_idx = min(int(progress * len(config.energy_arc)),
                          len(config.energy_arc) - 1)
            target_e = config.energy_arc[arc_idx]
            last_entry = st.sequence[-1]
            last_clip = clips[last_entry.clip_id]
            # Candidates: unused clips
            for cid, cand in clips.items():
                if cid in st.used_clip_ids:
                    continue
                # Pick segment for candidate matching target energy
                seg = min(cand['sections'],
                          key=lambda s: abs(s.get('energy', 0.5) - target_e),
                          default=pick_segment(cand))
                score, tech = transition_score(
                    last_clip, last_entry.segment, last_entry.target_bpm,
                    last_entry.target_key, cand, seg, config.weights,
                )
                # Surprise: low-score transition allowed if budget permits
                is_surprise = score < 0.4
                if is_surprise and st.surprises_used >= config.surprise_budget:
                    continue
                new_entry = TimelineEntry(
                    clip_id=cid, segment=seg,
                    target_bpm=last_entry.target_bpm,  # maintain BPM
                    target_key=last_entry.target_key,  # maintain key
                    transition_in=tech,
                )
                new_st = State(
                    sequence=st.sequence + [new_entry],
                    cumulative_duration=st.cumulative_duration + (seg['end'] - seg['start']),
                    used_clip_ids=st.used_clip_ids | {cid},
                    surprises_used=st.surprises_used + (1 if is_surprise else 0),
                    callbacks_used=st.callbacks_used,
                    score=st.score + score,
                )
                next_beam.append(new_st)
        beam = sorted(next_beam, key=lambda s: -s.score)[:config.beam_width]
    if not finished:
        finished = beam
    best = max(finished, key=lambda s: s.score / max(1, len(s.sequence)))

    # Schedule callbacks: insert hook from earlier clip back later
    if config.callback_budget > 0 and len(best.sequence) > 4:
        from copy import deepcopy
        seq = deepcopy(best.sequence)
        # Find best hook in first half
        candidates = []
        for i, e in enumerate(seq[:len(seq)//2]):
            for h in clips[e.clip_id].get('hooks', []):
                candidates.append((h['strength'], i, e.clip_id, h))
        if candidates:
            candidates.sort(key=lambda x: -x[0])
            _, _, cid, hook = candidates[0]
            insert_at = len(seq) * 3 // 4
            callback = TimelineEntry(
                clip_id=cid,
                segment={'start': hook['start'], 'end': hook['end'], 'type': 'callback'},
                target_bpm=seq[insert_at].target_bpm,
                target_key=seq[insert_at].target_key,
                transition_in={'name': 'loop_callback', 'bars': hook['bars']},
            )
            seq.insert(insert_at, callback)
            best.sequence = seq

    # Set play_at times
    t = 0.0
    for e in best.sequence:
        e.play_at = t
        t += e.segment['end'] - e.segment['start']

    return [asdict(e) for e in best.sequence]


def save_timeline(timeline: list, path: str):
    with open(path, 'w') as f:
        json.dump({'timeline': timeline}, f, indent=2)
```

---

## Layer 3 — Transition library (full catalogue)

### DJ technique → implementation map

| DJ technique | Implemented as | When planner picks it |
|--------------|----------------|----------------------|
| **Beatmatching** | `stretch_and_pitch` (rubberband) on every clip render | always, foundational |
| **Harmonic Mixing** | Camelot wheel + pitch-shift in `execute.camelot_to_semitones` | always (key-aware planner) |
| **Phrase Matching** | planner enforces transitions on 16/32-bar boundaries via `snap_to_phrase` | always (mandatory) |
| **EQ Mixing / Blending** | `eq_swap_transition` (kill out-bass, raise in-bass) | high-energy continuity |
| **The Fade** | `crossfade_transition` (equal-power) | medium-energy continuity |
| **Filter Fade** | `filter_fade_transition` (LP cutoff sweep on outgoing) | mood-shift, breakdown intro |
| **Drop** | `silence_drop_transition` (cut to silence N beats then full re-entry) | tension-release moments |
| **Mashup** | `mashup_transition` (sustained vocals-of-A over instrumental-of-B) | vocal-rich → vocal-rich pairs |
| **Looping & Tightening** | `loop_tighten_transition` (last bar → 1/2 → 1/4 → 1/8 then drop) | climactic build into incoming |
| **Echo Out** | `echo_out_transition` (delay/feedback tail on outgoing last bars) | abrupt key/genre change |
| **Spinback** | `spinback_transition` (pitch-down to 0 + reverse on outgoing tail) | big-moment punctuation |
| **Pitch Control (bend)** | `pitch_bend_transition` (gradual ±1-2 semitone over N bars) | reach harmonic compatibility |
| **Scratching** | `scratch_fill` (forward/reverse jog on hook, 1-2 bars) | filler, energy injection |
| **Sampling** | `sample_trigger` (one-shot from `samples/` bank: impacts, risers, vinyl FX) | embellishment, drops |
| **Drum Break** | `drum_break_transition` (drums-only N bars then incoming) | reset between contrasting tracks |
| **Loop Callback** | `loop_callback` (repeat earlier hook from prior clip) | cohesion, structural recall |

15 distinct techniques. Planner selects per transition based on context score.

### Sample bank

Curate `samples/` directory with one-shots: kick impacts, snare rolls, white-noise risers, downsweeps, vinyl-stop FX, air-horn, reverb-impact. Free CC0 sources: Cymatics free packs, freesound.org. ~50 samples covers most needs.

```
samples/
├── impacts/          # kick.wav, sub_drop.wav, reverse_crash.wav
├── risers/           # white_noise_4bar.wav, synth_riser_8bar.wav
├── sweeps/           # downsweep_2bar.wav, filter_sweep_4bar.wav
├── vinyl/            # spinback_fx.wav, scratch_loop.wav
└── manifest.json     # {file, type, length_bars, bpm_native (or 'agnostic')}
```

### Phrase enforcement

Pro DJs transition on 16- or 32-bar boundaries — almost never mid-phrase. Enforce in planner:

```python
# src/phrase.py
def snap_to_phrase(t_sec: float, downbeats: list, bars_per_phrase: int = 16) -> float:
    """Snap time to nearest phrase boundary (every N downbeats)."""
    phrase_dbs = downbeats[::bars_per_phrase]
    if not phrase_dbs:
        return t_sec
    import numpy as np
    return float(phrase_dbs[np.argmin(np.abs(np.array(phrase_dbs) - t_sec))])

def detect_phrase_length(downbeats: list, energy_curve: list, sr_hz: int = 10) -> int:
    """Heuristic: 16 bars default; 32 if novelty changes align at 32-bar grid."""
    # Most EDM/house = 16-bar phrases. Hip-hop = 8 or 16. Jazz = irregular.
    # For MVP, default 16. Detect 32 via autocorrelation of energy at 32-bar lag.
    import numpy as np
    if len(downbeats) < 64:
        return 16
    bar_energies = []
    energy = np.array(energy_curve)
    for i in range(len(downbeats) - 1):
        s = int(downbeats[i] * sr_hz); e = int(downbeats[i+1] * sr_hz)
        if e > s and e <= len(energy):
            bar_energies.append(float(energy[s:e].mean()))
    if len(bar_energies) < 64:
        return 16
    bar_energies = np.array(bar_energies)
    # Compare 16-bar autocorrelation vs 32-bar
    def autocorr(x, lag):
        if lag >= len(x): return 0
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
    return 32 if autocorr(bar_energies, 32) > autocorr(bar_energies, 16) + 0.1 else 16
```

Use `snap_to_phrase` when computing transition start times in execute.py. Use `detect_phrase_length` per clip in analyze.py and store in metadata.

### transitions.py — full implementation

```python
# src/transitions.py
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
from scipy.signal import butter, sosfilt

SR = 44100


# ---------- DSP helpers ----------

def equal_power_xfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Equal-power crossfade over n samples. a fades out, b fades in."""
    n = min(n, len(a), len(b))
    t = np.linspace(0, np.pi/2, n)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    out_len = max(len(a), len(b))
    out = np.zeros((2, out_len))
    out[:, :len(a)] += a
    a_tail = a[:, -n:] * fade_out
    out[:, len(a)-n:len(a)] = a_tail
    b_head = b[:, :n] * fade_in
    out[:, :n] += np.concatenate([np.zeros((2, max(0, len(a)-n))), b_head], axis=1)[:, :n] if len(a)-n > 0 else b_head[:, :n]
    return out


def lp_filter(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(4, cutoff, btype='low', fs=sr, output='sos')
    return np.stack([sosfilt(sos, ch) for ch in x])


def hp_filter(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    sos = butter(4, cutoff, btype='high', fs=sr, output='sos')
    return np.stack([sosfilt(sos, ch) for ch in x])


def eq_swap_transition(out_full: np.ndarray, in_full: np.ndarray,
                       sr: int, bars: int, beat_dur: float) -> np.ndarray:
    """
    Kill outgoing bass over N bars, raise incoming bass simultaneously.
    out_full, in_full: stereo (2, T) numpy arrays.
    Returns concatenated mix region.
    """
    region_dur = bars * 4 * beat_dur
    n_region = int(region_dur * sr)
    out_region = out_full[:, -n_region:] if out_full.shape[1] >= n_region else out_full
    in_region = in_full[:, :n_region] if in_full.shape[1] >= n_region else in_full
    n = min(out_region.shape[1], in_region.shape[1])
    # Linear EQ ramp
    bass_cutoff = 200.0
    out_low = lp_filter(out_region[:, :n], sr, bass_cutoff)
    out_high = hp_filter(out_region[:, :n], sr, bass_cutoff)
    in_low = lp_filter(in_region[:, :n], sr, bass_cutoff)
    in_high = hp_filter(in_region[:, :n], sr, bass_cutoff)
    ramp = np.linspace(1.0, 0.0, n)
    inv = 1.0 - ramp
    mixed = (out_low * ramp + in_low * inv) + (out_high * ramp * 0.5 + in_high * inv * 0.5)
    return mixed


def silence_drop_transition(out_full: np.ndarray, in_full: np.ndarray,
                            sr: int, silence_beats: float, beat_dur: float) -> np.ndarray:
    n_silence = int(silence_beats * beat_dur * sr)
    silence = np.zeros((2, n_silence))
    return np.concatenate([out_full, silence, in_full], axis=1)


def drum_break_transition(out_drums: np.ndarray, in_full: np.ndarray,
                          sr: int, bars: int, beat_dur: float) -> np.ndarray:
    """Strip everything except drums for N bars, then full incoming."""
    n_break = int(bars * 4 * beat_dur * sr)
    drum_only = out_drums[:, -n_break:] if out_drums.shape[1] >= n_break else out_drums
    return np.concatenate([drum_only, in_full], axis=1)


def stem_swap_transition(out_inst: np.ndarray, in_vox: np.ndarray,
                         in_full: np.ndarray, sr: int, bars: int, beat_dur: float) -> np.ndarray:
    """Outgoing instrumental + incoming vocals overlaid for N bars, then incoming full."""
    n_overlay = int(bars * 4 * beat_dur * sr)
    n = min(out_inst.shape[1], in_vox.shape[1], n_overlay)
    overlay = out_inst[:, :n] * 0.7 + in_vox[:, :n] * 1.0
    return np.concatenate([overlay, in_full], axis=1)


def cut_transition(out_full: np.ndarray, in_full: np.ndarray) -> np.ndarray:
    return np.concatenate([out_full, in_full], axis=1)


def crossfade_transition(out_full: np.ndarray, in_full: np.ndarray,
                         sr: int, bars: int, beat_dur: float) -> np.ndarray:
    n = int(bars * 4 * beat_dur * sr)
    return equal_power_xfade(out_full, in_full, n)


def loop_callback(hook: np.ndarray, repetitions: int) -> np.ndarray:
    """Loop a hook segment N times."""
    return np.tile(hook, (1, repetitions))


def riser_bridge(duration_sec: float, sr: int) -> np.ndarray:
    """Synthesized white-noise riser (filtered up over time)."""
    n = int(duration_sec * sr)
    noise = np.random.randn(2, n) * 0.1
    out = np.zeros_like(noise)
    chunk = sr // 50
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        progress = i / n
        cutoff = 200 + (8000 - 200) * progress
        gain = 0.3 + 0.7 * progress
        out[:, i:end] = lp_filter(noise[:, i:end], sr, cutoff) * gain
    return out


# ---------- New: Filter Fade ----------

def filter_fade_transition(out_full: np.ndarray, in_full: np.ndarray,
                           sr: int, bars: int, beat_dur: float) -> np.ndarray:
    """
    Sweep LP cutoff DOWN on outgoing (8kHz → 200Hz) over N bars, simultaneously
    fade outgoing volume + raise incoming. Classic moody transition.
    """
    n_region = int(bars * 4 * beat_dur * sr)
    out_region = out_full[:, -n_region:] if out_full.shape[1] >= n_region else out_full
    in_region = in_full[:, :n_region] if in_full.shape[1] >= n_region else in_full
    n = min(out_region.shape[1], in_region.shape[1])
    if n == 0: return np.concatenate([out_full, in_full], axis=1)
    chunk = sr // 50  # 20ms
    out_filtered = np.zeros_like(out_region[:, :n])
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        progress = i / n
        cutoff = 8000 - (8000 - 200) * progress
        out_filtered[:, i:end] = lp_filter(out_region[:, i:end], sr, cutoff)
    fade_out = np.linspace(1.0, 0.0, n)
    fade_in = np.linspace(0.0, 1.0, n)
    mixed = out_filtered * fade_out + in_region[:, :n] * fade_in
    return np.concatenate([out_full[:, :-n] if out_full.shape[1] >= n else np.zeros((2,0)),
                           mixed,
                           in_full[:, n:]], axis=1)


# ---------- New: Echo Out ----------

def echo_out_transition(out_full: np.ndarray, in_full: np.ndarray,
                        sr: int, bars: int, beat_dur: float,
                        delay_beats: float = 0.5, feedback: float = 0.55) -> np.ndarray:
    """
    Apply feedback delay to last N bars of outgoing, let tail trail into incoming.
    Outgoing dry signal cuts at end of region; only echo tail remains.
    """
    region_n = int(bars * 4 * beat_dur * sr)
    delay_samp = int(delay_beats * beat_dur * sr)
    if out_full.shape[1] < region_n:
        return np.concatenate([out_full, in_full], axis=1)
    region = out_full[:, -region_n:].copy()
    # Generate decaying echo tail (extends past region)
    tail_len = region_n + int(2 * sr)  # +2s of tail
    tail = np.zeros((2, tail_len))
    tail[:, :region_n] = region
    # Fade outgoing dry signal to zero over second half of region
    fade = np.concatenate([np.ones(region_n // 2),
                           np.linspace(1.0, 0.0, region_n - region_n // 2)])
    tail[:, :region_n] *= fade
    # Apply feedback delay
    for i in range(delay_samp, tail_len):
        tail[:, i] += tail[:, i - delay_samp] * feedback
    # Clip to prevent runaway
    tail = np.clip(tail, -1.0, 1.0)
    # Mix tail with incoming
    body = out_full[:, :-region_n]
    overlap = min(tail_len, in_full.shape[1])
    mixed_overlap = tail[:, :overlap]
    in_first = in_full[:, :overlap]
    mixed_overlap = mixed_overlap + in_first
    return np.concatenate([body, mixed_overlap, in_full[:, overlap:]], axis=1)


# ---------- New: Spinback ----------

def spinback_transition(out_full: np.ndarray, in_full: np.ndarray,
                        sr: int, spinback_beats: float, beat_dur: float) -> np.ndarray:
    """
    Pitch-bend outgoing tail down to zero (like vinyl stop) + reverse last 0.5s,
    then immediately incoming. Big-moment punctuation.
    """
    region_n = int(spinback_beats * beat_dur * sr)
    if out_full.shape[1] < region_n:
        return np.concatenate([out_full, in_full], axis=1)
    region = out_full[:, -region_n:]
    # Variable-rate playback: rate goes 1.0 → 0.0 over region (resampling effect)
    # Simple approximation: gradual time-stretch + downward pitch sweep via decimation
    out_chunks = []
    n_chunks = 40
    chunk_size = region_n // n_chunks
    for i in range(n_chunks):
        rate = 1.0 - (i / n_chunks)  # 1.0 → 0.0
        chunk = region[:, i * chunk_size:(i + 1) * chunk_size]
        if rate < 0.05:
            break
        # Pitch-down by playing slower (resample shorter)
        new_len = max(1, int(chunk.shape[1] * (1.0 + (1.0 - rate))))
        idx = np.linspace(0, chunk.shape[1] - 1, new_len).astype(int)
        out_chunks.append(chunk[:, idx])
    # Reverse final short tail (vinyl-stop reverse smear)
    tail_rev_n = int(0.3 * sr)
    if region.shape[1] >= tail_rev_n:
        out_chunks.append(region[:, -tail_rev_n:][:, ::-1] * 0.5)
    spinback = np.concatenate(out_chunks, axis=1) if out_chunks else np.zeros((2, 0))
    return np.concatenate([out_full[:, :-region_n], spinback, in_full], axis=1)


# ---------- New: Pitch Bend (gradual) ----------

def pitch_bend_transition(out_full: np.ndarray, in_full: np.ndarray,
                          sr: int, bars: int, beat_dur: float,
                          semitones: float = 1.0) -> np.ndarray:
    """
    Gradually pitch-bend outgoing ±semitones over N bars to reach incoming key.
    Implemented via per-chunk resampling (cheap, audible artifacts at extremes).
    """
    import pyrubberband as pyrb
    region_n = int(bars * 4 * beat_dur * sr)
    if out_full.shape[1] < region_n:
        return np.concatenate([out_full, in_full], axis=1)
    region = out_full[:, -region_n:]
    # Process in 8 stages, each pitched a fraction more
    stages = 8
    stage_n = region_n // stages
    out_stages = []
    for i in range(stages):
        progress = i / (stages - 1) if stages > 1 else 1.0
        st = semitones * progress
        chunk = region[:, i * stage_n:(i + 1) * stage_n]
        shifted = pyrb.pitch_shift(chunk.T, sr, st).T
        # Re-trim to original chunk length to avoid drift
        if shifted.shape[1] > stage_n:
            shifted = shifted[:, :stage_n]
        elif shifted.shape[1] < stage_n:
            shifted = np.pad(shifted, ((0,0),(0,stage_n - shifted.shape[1])))
        out_stages.append(shifted)
    bent = np.concatenate(out_stages, axis=1)
    # Crossfade into incoming over last 4 bars
    xfade_n = int(4 * 4 * beat_dur * sr)
    return equal_power_xfade(
        np.concatenate([out_full[:, :-region_n], bent], axis=1),
        in_full,
        xfade_n,
    )


# ---------- New: Mashup ----------

def mashup_transition(out_inst: np.ndarray, in_vocals: np.ndarray, in_full: np.ndarray,
                      sr: int, bars: int, beat_dur: float) -> np.ndarray:
    """
    Sustained mashup: incoming vocals over outgoing instrumental for N bars,
    then resolve to incoming full. Like stem_swap but longer + more deliberate.
    """
    n_overlay = int(bars * 4 * beat_dur * sr)
    n = min(out_inst.shape[1], in_vocals.shape[1], n_overlay)
    # Duck instrumental slightly so vocal sits forward
    overlay = out_inst[:, :n] * 0.65 + in_vocals[:, :n] * 1.0
    # Crossfade overlay→incoming over last 4 bars of overlay
    xfade_n = int(4 * 4 * beat_dur * sr)
    xfade_n = min(xfade_n, n)
    transition_region = equal_power_xfade(overlay, in_full[:, :n], xfade_n)
    return np.concatenate([transition_region, in_full[:, n:]], axis=1)


# ---------- New: Looping & Tightening ----------

def loop_tighten_transition(out_full: np.ndarray, in_full: np.ndarray,
                            sr: int, beat_dur: float, start_bars: int = 4) -> np.ndarray:
    """
    Take last N bars of outgoing, loop with halving length: N → N/2 → N/4 → N/8,
    then drop into incoming. Climactic build.
    """
    n_loop_full = int(start_bars * 4 * beat_dur * sr)
    if out_full.shape[1] < n_loop_full:
        return np.concatenate([out_full, in_full], axis=1)
    base = out_full[:, -n_loop_full:]
    sequence = []
    bars = start_bars
    while bars >= 0.5:
        n = int(bars * 4 * beat_dur * sr)
        if n < 1: break
        loop_seg = base[:, :n]
        sequence.append(loop_seg)
        bars /= 2
    tightened = np.concatenate(sequence, axis=1)
    return np.concatenate([out_full[:, :-n_loop_full], tightened, in_full], axis=1)


# ---------- New: Scratch fill ----------

def scratch_fill(hook: np.ndarray, sr: int, beat_dur: float, n_jogs: int = 4) -> np.ndarray:
    """
    Synthetic scratch: forward/reverse jogs on a short hook segment.
    Each jog = forward N samples then reverse N samples.
    """
    jog_dur = beat_dur * 0.5  # half-beat per jog
    jog_n = int(jog_dur * sr)
    if hook.shape[1] < jog_n * 2:
        return hook
    jog = hook[:, :jog_n]
    out = []
    for i in range(n_jogs):
        out.append(jog)
        out.append(jog[:, ::-1])
    return np.concatenate(out, axis=1)


# ---------- New: Sample trigger ----------

def load_sample_bank(samples_dir: str = 'samples') -> dict:
    """Load all samples + manifest. Returns {type: [(name, np_audio), ...]}"""
    bank = {}
    manifest_path = Path(samples_dir) / 'manifest.json'
    if not manifest_path.exists():
        return bank
    with open(manifest_path) as f:
        manifest = json.load(f)
    for entry in manifest:
        wav, sr = torchaudio.load(str(Path(samples_dir) / entry['file']))
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.size(0) == 1:
            wav = wav.repeat(2, 1)
        bank.setdefault(entry['type'], []).append((entry['file'], wav.numpy()))
    return bank


def sample_trigger(bank: dict, sample_type: str, idx: int = 0) -> np.ndarray:
    """Get a sample from bank by type."""
    items = bank.get(sample_type, [])
    if not items:
        return np.zeros((2, 0))
    return items[idx % len(items)][1]


def overlay_sample(host: np.ndarray, sample: np.ndarray, at_sample_idx: int,
                   gain: float = 0.7) -> np.ndarray:
    """Mix sample on top of host audio at given sample index."""
    if sample.shape[1] == 0 or at_sample_idx >= host.shape[1]:
        return host
    end = min(at_sample_idx + sample.shape[1], host.shape[1])
    host = host.copy()
    host[:, at_sample_idx:end] += sample[:, :end - at_sample_idx] * gain
    return np.clip(host, -1.0, 1.0)
```

---

## Layer 4 — Execute

```python
# src/execute.py
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
import pyrubberband as pyrb

import transitions as T

SR = 44100


def load_stems(clip_id: str, cache_dir: str = 'cache') -> dict:
    base = Path(cache_dir) / 'stems' / clip_id
    out = {}
    for name in ['drums', 'bass', 'other', 'vocals']:
        wav, sr = torchaudio.load(str(base / f'{name}.wav'))
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        out[name] = wav.numpy()
    return out


def stretch_and_pitch(wav: np.ndarray, sr: int, src_bpm: float, dst_bpm: float,
                      semitones: float) -> np.ndarray:
    # rubberband expects (T, channels)
    x = wav.T
    if abs(src_bpm - dst_bpm) > 0.01 and src_bpm > 0:
        x = pyrb.time_stretch(x, sr, dst_bpm / src_bpm)
    if abs(semitones) > 0.01:
        x = pyrb.pitch_shift(x, sr, semitones)
    return x.T


def camelot_to_semitones(src: str, dst: str) -> float:
    """Compute pitch-shift in semitones to move from src key to dst key."""
    # Camelot to pitch class
    REVERSE = {}
    from camelot import CAMELOT
    for (pc, mode), code in CAMELOT.items():
        REVERSE[code] = pc
    if src not in REVERSE or dst not in REVERSE:
        return 0.0
    diff = (REVERSE[dst] - REVERSE[src]) % 12
    if diff > 6:
        diff -= 12
    # Cap at ±3 semitones (beyond = audible artifacts)
    return float(np.clip(diff, -3, 3))


def render_segment(entry: dict, clips_meta: dict) -> tuple[np.ndarray, dict]:
    """Load, slice, time-stretch, pitch-shift one timeline entry. Returns (mix, stems_dict)."""
    cid = entry['clip_id']
    seg = entry['segment']
    target_bpm = entry['target_bpm']
    target_key = entry['target_key']

    meta = clips_meta[cid]
    src_bpm = meta['tempo']
    src_key = meta['key']
    semitones = camelot_to_semitones(src_key, target_key)

    stems = load_stems(cid)
    s_start = int(seg['start'] * SR)
    s_end = int(seg['end'] * SR)
    sliced = {n: s[:, s_start:s_end] for n, s in stems.items()}
    # Stretch+pitch each stem with same params (preserves alignment)
    processed = {n: stretch_and_pitch(s, SR, src_bpm, target_bpm, semitones)
                 for n, s in sliced.items()}
    full = sum(processed.values())
    return full, processed


def execute(timeline_path: str, cache_dir: str, out_path: str):
    with open(timeline_path) as f:
        tl = json.load(f)['timeline']

    # Load all metadata
    clips_meta = {}
    for entry in tl:
        cid = entry['clip_id']
        if cid not in clips_meta:
            with open(Path(cache_dir) / f'{cid}.json') as f:
                clips_meta[cid] = json.load(f)

    rendered = []
    for entry in tl:
        full, stems = render_segment(entry, clips_meta)
        rendered.append({'entry': entry, 'full': full, 'stems': stems,
                         'meta': clips_meta[entry['clip_id']]})

    # Load sample bank (if exists)
    sample_bank = T.load_sample_bank('samples')

    # Stitch with transitions
    output = rendered[0]['full']
    for i in range(1, len(rendered)):
        prev = rendered[i-1]
        cur = rendered[i]
        tech = cur['entry']['transition_in']
        name = tech['name']
        bars = tech.get('bars', 16)
        target_bpm = cur['entry']['target_bpm']
        beat_dur = 60.0 / target_bpm

        if name == 'cut':
            output = T.cut_transition(output, cur['full'])
        elif name == 'crossfade':
            output = T.crossfade_transition(output, cur['full'], SR, bars, beat_dur)
        elif name == 'eq_swap':
            mix = T.eq_swap_transition(output, cur['full'], SR, bars, beat_dur)
            n_region = int(bars * 4 * beat_dur * SR)
            output = np.concatenate([output[:, :-n_region], mix, cur['full'][:, n_region:]], axis=1)
        elif name == 'filter_fade':
            output = T.filter_fade_transition(output, cur['full'], SR, bars, beat_dur)
        elif name == 'silence_drop':
            silence_beats = tech.get('silence_beats', tech.get('bars', 1) * 4)
            output = T.silence_drop_transition(output, cur['full'], SR, silence_beats, beat_dur)
            # Embellish with impact sample on the re-entry
            impact = T.sample_trigger(sample_bank, 'impacts', 0)
            if impact.shape[1] > 0:
                impact_at = output.shape[1] - cur['full'].shape[1]
                output = T.overlay_sample(output, impact, impact_at, gain=0.6)
        elif name == 'drum_break':
            output = T.drum_break_transition(prev['stems']['drums'], cur['full'], SR, bars, beat_dur)
        elif name == 'mashup':
            inst = sum(v for k, v in prev['stems'].items() if k != 'vocals')
            output = T.mashup_transition(inst, cur['stems']['vocals'], cur['full'], SR, bars, beat_dur)
        elif name == 'stem_swap':
            inst = sum(v for k, v in prev['stems'].items() if k != 'vocals')
            output = T.stem_swap_transition(inst, cur['stems']['vocals'], cur['full'], SR, bars, beat_dur)
        elif name == 'echo_out':
            output = T.echo_out_transition(
                output, cur['full'], SR, bars, beat_dur,
                delay_beats=tech.get('delay_beats', 0.5),
                feedback=tech.get('feedback', 0.55))
        elif name == 'spinback':
            sb_beats = tech.get('spinback_beats', 4)
            output = T.spinback_transition(output, cur['full'], SR, sb_beats, beat_dur)
            # Add vinyl FX sample on the spinback if available
            vinyl = T.sample_trigger(sample_bank, 'vinyl', 0)
            if vinyl.shape[1] > 0:
                fx_at = max(0, output.shape[1] - cur['full'].shape[1] - int(sb_beats * beat_dur * SR))
                output = T.overlay_sample(output, vinyl, fx_at, gain=0.5)
        elif name == 'pitch_bend':
            output = T.pitch_bend_transition(
                output, cur['full'], SR, bars, beat_dur,
                semitones=tech.get('semitones', 1.0))
        elif name == 'loop_tighten':
            output = T.loop_tighten_transition(
                output, cur['full'], SR, beat_dur,
                start_bars=tech.get('start_bars', 4))
            # Riser embellishment over the tighten region
            riser = T.sample_trigger(sample_bank, 'risers', 0)
            if riser.shape[1] > 0:
                tighten_n = int(tech.get('start_bars', 4) * 4 * beat_dur * SR * 2)
                riser_at = max(0, output.shape[1] - cur['full'].shape[1] - tighten_n)
                output = T.overlay_sample(output, riser, riser_at, gain=0.4)
        elif name == 'scratch_fill':
            hook_seg = prev['full'][:, -int(2 * beat_dur * SR):]
            scratch = T.scratch_fill(hook_seg, SR, beat_dur,
                                     n_jogs=tech.get('n_jogs', 4))
            output = np.concatenate([output, scratch, cur['full']], axis=1)
        elif name == 'loop_callback':
            output = T.cut_transition(output, T.loop_callback(cur['full'], 2))
        else:
            output = T.crossfade_transition(output, cur['full'], SR, bars, beat_dur)

    # Save raw
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(out_path, torch.from_numpy(output.astype(np.float32)), SR)
    return output
```

---

## Layer 5 — Master

```python
# src/master.py
import numpy as np
import torch
import torchaudio
import pyloudnorm as pyln
from scipy.signal import butter, sosfilt

SR = 44100


def hp(x, sr, cutoff):
    sos = butter(4, cutoff, btype='high', fs=sr, output='sos')
    return np.stack([sosfilt(sos, ch) for ch in x])


def split_bands(x, sr):
    sos_low = butter(4, 200, btype='low', fs=sr, output='sos')
    sos_mid_lp = butter(4, 4000, btype='low', fs=sr, output='sos')
    sos_mid_hp = butter(4, 200, btype='high', fs=sr, output='sos')
    sos_high = butter(4, 4000, btype='high', fs=sr, output='sos')
    low = np.stack([sosfilt(sos_low, ch) for ch in x])
    mid_lp = np.stack([sosfilt(sos_mid_lp, ch) for ch in x])
    mid = np.stack([sosfilt(sos_mid_hp, ch) for ch in mid_lp])
    high = np.stack([sosfilt(sos_high, ch) for ch in x])
    return low, mid, high


def compress(x, threshold_db=-20, ratio=4.0, attack_ms=10, release_ms=100, sr=44100):
    eps = 1e-10
    abs_x = np.abs(x).max(axis=0)
    db = 20 * np.log10(abs_x + eps)
    over = np.maximum(0, db - threshold_db)
    gain_red_db = -over * (1 - 1/ratio)
    # Smooth envelope
    a_a = np.exp(-1 / (attack_ms * sr / 1000))
    a_r = np.exp(-1 / (release_ms * sr / 1000))
    env = np.zeros_like(gain_red_db)
    g = 0.0
    for i, target in enumerate(gain_red_db):
        coef = a_a if target < g else a_r
        g = coef * g + (1 - coef) * target
        env[i] = g
    gain_lin = 10 ** (env / 20)
    return x * gain_lin


def limit(x, ceiling_db=-1.0, lookahead_ms=5, sr=44100):
    ceiling = 10 ** (ceiling_db / 20)
    lookahead = int(lookahead_ms * sr / 1000)
    abs_x = np.abs(x).max(axis=0)
    # Find max in lookahead window
    pad = np.concatenate([abs_x, np.zeros(lookahead)])
    rolling = np.array([pad[i:i+lookahead].max() for i in range(len(abs_x))])
    gain = np.where(rolling > ceiling, ceiling / (rolling + 1e-10), 1.0)
    # Smooth
    a = np.exp(-1 / (lookahead * 0.5))
    smoothed = np.zeros_like(gain)
    g = 1.0
    for i, target in enumerate(gain):
        g = min(target, a * g + (1 - a) * target)
        smoothed[i] = g
    return x * smoothed


def lufs_normalize(x, sr, target_lufs=-9.0):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(x.T)
    return pyln.normalize.loudness(x.T, loudness, target_lufs).T


def master(in_path: str, out_path: str, target_lufs: float = -9.0):
    wav, sr = torchaudio.load(in_path)
    x = wav.numpy()
    if x.shape[0] == 1:
        x = np.concatenate([x, x], axis=0)
    # HP at 30Hz to remove sub-rumble
    x = hp(x, sr, 30)
    # Multiband compression
    low, mid, high = split_bands(x, sr)
    low = compress(low, threshold_db=-18, ratio=3, sr=sr)
    mid = compress(mid, threshold_db=-20, ratio=2.5, sr=sr)
    high = compress(high, threshold_db=-22, ratio=2, sr=sr)
    x = low + mid + high
    # Glue compressor
    x = compress(x, threshold_db=-12, ratio=2, sr=sr)
    # LUFS
    x = lufs_normalize(x, sr, target_lufs)
    # Limit
    x = limit(x, ceiling_db=-1.0, sr=sr)
    torchaudio.save(out_path, torch.from_numpy(x.astype(np.float32)), sr)
```

---

## Layer 6 — CLI

```python
# src/main.py
import argparse
from pathlib import Path

from analyze import analyze_pool
from planner import load_clips, plan, save_timeline, PlannerConfig
from execute import execute
from master import master


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p_an = sub.add_parser('analyze'); p_an.add_argument('--clips', required=True); p_an.add_argument('--cache', default='cache'); p_an.add_argument('--device', default='cuda')
    p_pl = sub.add_parser('plan'); p_pl.add_argument('--cache', default='cache'); p_pl.add_argument('--out', default='output/timeline.json'); p_pl.add_argument('--duration', type=float, default=600); p_pl.add_argument('--surprises', type=int, default=1); p_pl.add_argument('--callbacks', type=int, default=1)
    p_ex = sub.add_parser('execute'); p_ex.add_argument('--timeline', default='output/timeline.json'); p_ex.add_argument('--cache', default='cache'); p_ex.add_argument('--out', default='output/raw_mix.wav')
    p_ms = sub.add_parser('master'); p_ms.add_argument('--in_path', default='output/raw_mix.wav'); p_ms.add_argument('--out', default='output/final_mix.wav'); p_ms.add_argument('--lufs', type=float, default=-9.0)
    p_all = sub.add_parser('all'); p_all.add_argument('--clips', required=True); p_all.add_argument('--cache', default='cache'); p_all.add_argument('--out_dir', default='output'); p_all.add_argument('--duration', type=float, default=600)
    args = ap.parse_args()

    if args.cmd == 'analyze':
        analyze_pool(args.clips, args.cache, args.device)
    elif args.cmd == 'plan':
        clips = load_clips(args.cache)
        cfg = PlannerConfig(target_duration=args.duration,
                            surprise_budget=args.surprises,
                            callback_budget=args.callbacks)
        tl = plan(clips, cfg)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_timeline(tl, args.out)
    elif args.cmd == 'execute':
        execute(args.timeline, args.cache, args.out)
    elif args.cmd == 'master':
        master(args.in_path, args.out, args.lufs)
    elif args.cmd == 'all':
        analyze_pool(args.clips, args.cache, 'cuda')
        clips = load_clips(args.cache)
        cfg = PlannerConfig(target_duration=args.duration)
        tl = plan(clips, cfg)
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        save_timeline(tl, str(out_dir / 'timeline.json'))
        execute(str(out_dir / 'timeline.json'), args.cache, str(out_dir / 'raw_mix.wav'))
        master(str(out_dir / 'raw_mix.wav'), str(out_dir / 'final_mix.wav'))


if __name__ == '__main__':
    main()
```

Usage:

```bash
python src/main.py all --clips clips/ --duration 600 --out_dir output/
# or step by step:
python src/main.py analyze --clips clips/ --cache cache/
python src/main.py plan --cache cache/ --duration 900 --surprises 2
python src/main.py execute
python src/main.py master --lufs -9
```

---

## Phase B — generative augmentation

After Phase A produces listenable mixes, identify failure cases:
- Transition score < 0.3 → fill needed
- Long silence → riser/sweep needed
- Outro→intro tempo gap > 12% → bridge needed

**Stable Audio Open** (1.2B diffusion, Stability AI Community License):

```python
# src/generative.py
from diffusers import StableAudioPipeline
import torch

def load_sao(device='cuda'):
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16,
    ).to(device)
    return pipe

def generate_bridge(pipe, prompt: str, duration_sec: float = 4.0, bpm: int = 128):
    audio = pipe(
        prompt=f"{prompt}, {bpm} BPM, seamless transition, no vocals",
        negative_prompt="low quality, vocals, lyrics",
        num_inference_steps=200,
        audio_end_in_s=duration_sec,
    ).audios[0]
    return audio  # (channels, T) at 44100 Hz
```

**Last-layer fine-tune on MI300X**:

Replace SAO's final projection layer + train on dataset of (transition_context, transition_audio) pairs:

```python
# Roughly:
# 1. Freeze pipe.transformer except last block + final projection
# 2. Build dataset: (outgoing_clip_clap_emb, incoming_clip_clap_emb, ground_truth_bridge_audio)
# 3. Loss: MSE on diffusion noise prediction conditioned on both embeddings
# 4. ~3-5 days on MI300X, $200-400 credits
```

Skip until Phase A clearly hits ceiling.

---

## MI300X usage plan

| Activity | Where | Est. cost |
|----------|-------|-----------|
| ROCm sanity (00_rocm_sanity.py) | MI300X 1hr | $5 |
| Phase A dev | laptop CPU | $0 |
| Phase A end-to-end test (10 clips) | laptop GPU or 1 MI300X hr | $5 |
| Mass analysis (1000+ clips) | MI300X batch job | $20-50 |
| Phase B SAO inference experiments | MI300X 10hr | $50 |
| Phase B last-layer fine-tune | MI300X 3-5 days | $200-400 |

**Total Phase A budget**: $30-60. **Phase B budget**: $250-450.

Save credits by: developing on laptop, batching analysis jobs, no idle sessions.

---

## Eval

### Objective metrics (compute on 10 generated mixes)

```python
# src/eval.py snippet
import librosa
import numpy as np

def beat_continuity_score(mix, sr, transition_times):
    """% of transitions where downbeat continues across boundary."""
    proc = madmom.features.beats.RNNBeatProcessor()
    _, beat_frames = librosa.beat.beat_track(y=mix, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    hits = 0
    for t in transition_times:
        nearest = beat_times[np.argmin(np.abs(beat_times - t))]
        if abs(nearest - t) < 0.05:
            hits += 1
    return hits / max(1, len(transition_times))


def energy_arc_match(mix, sr, target_arc):
    """Correlation between actual RMS curve and target arc."""
    rms = librosa.feature.rms(y=mix, frame_length=44100, hop_length=22050)[0]
    # Resample target_arc to same length
    target = np.interp(np.linspace(0, 1, len(rms)),
                       np.linspace(0, 1, len(target_arc)), target_arc)
    return float(np.corrcoef(rms, target)[0, 1])


def lufs_in_range(mix, sr, low=-10, high=-7):
    import pyloudnorm as pyln
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(mix.T if mix.ndim > 1 else mix)
    return low <= lufs <= high, lufs
```

### Targets

| Metric | Target |
|--------|--------|
| Beat continuity at transitions | >85% within ±50ms |
| Energy arc correlation | r > 0.6 |
| LUFS in club range | -10 to -7 |
| Key clash count (detected dissonance) | 0 per mix |
| Subjective: blind A/B vs djay Pro auto | wins ≥70% of the time |
| Subjective: vs human DJ amateur set | comparable on 5 of 10 mixes |

---

## Timeline

| Week | Deliverable |
|------|-------------|
| 1 | env setup, ROCm sanity, project skeleton |
| 2 | analyze.py: stems + beats + key + structure on 5 test clips |
| 3 | hooks.py + camelot.py + CLAP integration |
| 4 | planner.py beam search produces sane timeline.json |
| 5-7 | transitions.py: 15 techniques + listenable individually |
| 8 | sample bank curation (~50 one-shots), phrase-length detection |
| 9 | execute.py end-to-end render with all techniques + sample embellishment |
| 10 | master.py chain |
| 11-12 | full pipeline on 20 clips → first listenable 10-min mix |
| 13-14 | eval metrics + scoring weight tuning, technique-selection refinement |
| 15-16 | A/B vs djay Pro, iterate on weak transitions, fix artifacts |
| 17+ | Phase B if needed |

**Phase A total: ~4-5 months realistic** (15 techniques is more work than 7).

---

## Risks

| Risk | P | Mitigation |
|------|---|------------|
| Demucs ROCm slow/broken | medium | use small Demucs variant, batch on MI300X |
| madmom downbeat detection fails on EDM | medium | multi-algo cross-check, manual override |
| Phrase boundary detection imprecise | high | start with downbeat-only, add phrase later |
| rubberband artifacts on extreme stretch | medium | cap at ±8%, choose tracks accordingly |
| Mastering chain over-compresses | medium | A/B with raw, expose params |
| Planner picks bad orderings | high | tunable weights, manual timeline edit allowed |
| MI300X credits exhausted before Phase B | medium | budget table above, monitor weekly |
| Output sounds amateurish despite all this | possible | accept ceiling; the pro DJ moves are decades of taste |

---

## Abandon if

- Week 4: planner can't produce sane timeline on test set
- Week 8: no transition technique sounds clean on any test pair
- Week 12: full pipeline mix is unlistenable

## Push to Phase B if

- Phase A mixes sound competent on most transitions
- Specific transition types consistently fail (e.g. cross-genre)
- Want to add filler/bridge content not in source clips

---

## Product / UX

### Form factor

**Local web app**. FastAPI backend (Python, owns analyze/plan/execute/master) + lightweight React or vanilla-JS frontend. User runs `aijockey serve`, browser opens at `localhost:7860`. No cloud, no signup, files stay local.

Why web-app not CLI: visual timeline editing is essential. Why not desktop app: shipping Electron + ROCm/CUDA torch is painful; localhost web works everywhere.

### User journey (happy path)

```
1. LAUNCH      → `aijockey serve` → browser opens
2. POOL        → drag clips folder onto window → list populates
3. ANALYZE     → click "Analyze" → progress bar per clip (~30s/clip on GPU)
                 → table fills with detected BPM, key, sections, hooks
4. CONFIGURE   → set duration (10 min), energy arc (warmup→peak→cooldown),
                 surprise budget (1), callback budget (1)
5. PLAN        → click "Generate Set" → timeline appears as block diagram
                 → see clip order, segment used, transition technique per junction
6. AUDITION    → click any block to hear that clip's segment
                 → click any junction to hear ONLY that transition (10-sec preview)
7. EDIT        → swap clip, change technique, drag transition bar count
                 → manual override of detected key/BPM if wrong
                 → re-render only affected region (cached neighbors)
8. RENDER      → "Render Full Mix" → progress → audition raw → audition mastered
9. EXPORT      → download final_mix.wav + timeline.json (reproducible)
```

### Feature list

**MVP (v0.1) — month 4-5**
- Drag-drop clip pool (folder or files)
- Analyze pipeline (Demucs + madmom + librosa + CLAP) with progress
- Per-clip table: BPM, key (Camelot), duration, sections detected
- Manual override: BPM, key, section boundaries
- Configure: target duration, energy arc shape, surprise/callback budgets
- One-click "Generate Set" → timeline visualization
- Audition: per-clip segment, per-transition preview (10s)
- Render full mix → audition → download

**v0.2 — month 6**
- Timeline editor: drag clips to reorder, swap technique via dropdown, drag bar count
- Energy arc editor: drag curve shape
- Per-transition tweak panel (full param exposure)
- Re-render only edited region (incremental)
- Sample bank manager: list, audition, add/remove samples
- Mix versioning: save N versions of same set, A/B compare

**v0.3 — month 8**
- Library mode: persist clip pool across sessions, tag clips, search
- Smart pool subset: "give me 8 EDM clips around 128 BPM in compatible keys"
- Crowd-energy preset templates (warmup, peak-hour, after-hours, sunset)
- Export DAW-compatible: stems per region + Reaper/Ableton timeline import
- Generative fills (Phase B integration): mark transition as "regenerate via SAO"

**Not planned (out of scope)**
- Live performance mode (real-time crowd reading) — different product
- Vocal generation
- Multi-user / cloud / collaboration
- Mobile app

### UI sketch

```
┌─ AiJockey ──────────────────────────────────────────────────┐
│ [Pool] [Set: Untitled*]                       [Save] [Help] │
├─────────────────────────────────────────────────────────────┤
│ POOL (12 clips)                          [+ Add Clips]      │
│ ┌──────────────────────────────────────────────────────┐    │
│ │ ☑ track_001.wav   128 BPM  8A   3:42  ●drop ●hook    │    │
│ │ ☑ track_002.wav   126 BPM  9A   4:10  ●breakdown     │    │
│ │ ☑ track_003.wav   130 BPM  10A  3:55  ●drop          │    │
│ │ ☐ track_004.wav   124 BPM  3B   5:20  ●verse  [edit] │    │
│ │ ...                                                   │    │
│ └──────────────────────────────────────────────────────┘    │
│                                                              │
│ CONFIG                                                       │
│ Duration: [10:00▾]   Energy arc: ╱‾‾‾╲___ [edit]            │
│ Surprises: [1▾]      Callbacks: [1▾]                        │
│                              [Generate Set]                  │
├─────────────────────────────────────────────────────────────┤
│ TIMELINE (10:23)                                             │
│ ┌─────┐⤳eq_swap⤳┌──────┐⤳filter_fade⤳┌────┐⤳silence_drop⤳ │
│ │ 001 │           │ 003  │              │ 007│              │
│ │intro│           │ drop │              │bdwn│              │
│ │ 0:42│           │ 2:10 │              │1:30│              │
│ └─────┘           └──────┘              └────┘  ...          │
│  ▶ play          ▶ preview              ▶                    │
│                                                              │
│ Energy: ╱╲___╱‾‾‾╲___                                       │
│ Key:    8A→8A→9A→9A→2B→2B→...   (Camelot wheel ✓)           │
│ BPM:    128 ──── 128 ──── 128 ──── 128                      │
│                                                              │
│ [▶ Audition Raw]  [▶ Audition Mastered]  [Render]  [Export] │
└─────────────────────────────────────────────────────────────┘
```

### Editing model

Every edit re-runs only affected stages:
- Change clip selection → re-plan only (cached analyses reused)
- Edit detected BPM → re-analyze that clip + re-plan
- Swap transition technique → re-execute only that junction (~5 sec)
- Edit master settings → re-master only (~2 sec)

Cache layout: `cache/<clip_id>.json|.npz` (analysis), `cache/segments/<hash>.wav` (rendered segments), `cache/transitions/<hash>.wav` (rendered transitions). Hashing on inputs makes incremental render trivial.

### API surface (FastAPI)

```
POST /pool/add        # upload files OR scan folder
GET  /pool            # list with analysis status
POST /pool/analyze    # trigger analysis (SSE progress stream)
PATCH /clip/{id}      # override BPM, key, sections
POST /plan            # body: PlannerConfig → returns timeline
POST /timeline/edit   # apply edit op (swap, change_tech, reorder)
POST /audition        # body: timeline + range → returns wav stream
POST /render          # full mix render (SSE progress)
POST /master          # apply mastering chain
GET  /export          # download final_mix.wav + timeline.json
```

### Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | FastAPI | async, SSE for progress, simple |
| Frontend | Vanilla JS + Alpine.js or HTMX | no bundler complexity |
| Audio playback | Web Audio API + WaveSurfer.js | timeline waveforms, scrubbing |
| State | localStorage (sessions) + sqlite (library mode v0.3) | local-only |
| Process | uvicorn single worker | one user, one machine |
| Distribution | `pip install aijockey` + `aijockey serve` | one command |

### What user sees on errors

- Demucs OOM → "GPU out of memory. Try fewer clips at once or use --device cpu"
- Beat detection low confidence → flag clip with ⚠ in pool, suggest manual override
- Key detection ambiguous → show top-2 keys with confidence, let user pick
- Transition score < 0.3 → flag junction in timeline with ⚠, suggest alternative technique
- Generative fill needed (Phase B) → button: "Regenerate this 4-bar bridge"

### Performance targets

| Operation | Target |
|-----------|--------|
| Analyze 1 clip (3-min, GPU) | <30 sec |
| Analyze 1 clip (3-min, CPU) | <90 sec |
| Plan (12 clips, 10-min set) | <2 sec |
| Audition single transition (10s) | <3 sec |
| Render full 10-min mix | <2 min on GPU, <8 min CPU |
| Master | <30 sec |

### Distribution

- `pip install aijockey` (PyPI)
- Bundle: backend + static frontend in single wheel
- ROCm/CUDA torch as extras: `pip install aijockey[rocm]` / `[cuda]` / `[cpu]`
- Docker image for cloud users: `aijockey/aijockey:latest`
- Open source under MIT (your code) + respect upstream licenses (Demucs MIT, madmom BSD, etc.)

---

## Bottom line

Buildable. Honest target: **competent professional auto-DJ that beats existing automated tools and approaches mid-tier human DJ on transition quality**. Not Daft Punk. Daft Punk = signature production + decades of craft on original material. This = arrangement + execution intelligence on YOUR clips.

Where this wins: **15-technique transition library, stem-aware mashups, phrase-aligned mixing, sample-bank embellishments (impacts/risers/vinyl FX), callback structure, club-LUFS mastering**. Most auto-DJs do at most 2-3 of these.

### Full technique coverage

✅ Beatmatching · ✅ Harmonic Mixing · ✅ Phrase Matching · ✅ EQ Mixing / Blending · ✅ The Fade · ✅ Filter Fade · ✅ Drop · ✅ Mashup · ✅ Looping & Tightening · ✅ Echo Out · ✅ Spinback · ✅ Pitch Control · ✅ Scratching · ✅ Sampling · ✅ Drum Break · ✅ Loop Callback
