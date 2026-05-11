"""Multi-target mastering — Spotify/Apple/broadcast LUFS variants.

Re-runs LUFS normalization at multiple target levels, writes each
to a sibling file. Default targets:
    -9 LUFS  (festival / DJ)
    -14 LUFS (Spotify / Tidal / Amazon Music)
    -16 LUFS (Apple Music / YouTube Music)
    -23 LUFS (EBU R128 broadcast)

Toggle: AIJOCKEY_MULTI_TARGET=1
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    import pyloudnorm as pyln  # type: ignore
    import torchaudio
    import torch
    _HAS_DEPS = True
except Exception:
    _HAS_DEPS = False


DEFAULT_TARGETS: dict[str, float] = {
    "festival": -9.0,
    "spotify": -14.0,
    "apple": -16.0,
    "broadcast": -23.0,
}


def enabled() -> bool:
    return os.environ.get("AIJOCKEY_MULTI_TARGET", "0") == "1"


def write_targets(master_path: str | Path,
                    targets: dict[str, float] | None = None) -> dict[str, str]:
    """Read mastered audio, write variants at each target LUFS.

    Output filenames: `<stem>_<label>.<ext>` next to master_path.
    Returns {label: path}.
    """
    if not enabled() or not _HAS_DEPS:
        return {}
    p = Path(master_path)
    if not p.exists():
        return {}
    targets = targets or DEFAULT_TARGETS
    out_map: dict[str, str] = {}
    try:
        wav, sr = torchaudio.load(str(p))
        x = wav.numpy().astype(np.float32)
        meter = pyln.Meter(sr)
        cur_loud = meter.integrated_loudness(x.T)
        if not np.isfinite(cur_loud):
            return {}
        for label, tgt in targets.items():
            if abs(tgt - cur_loud) < 0.1:
                continue   # close enough, skip
            y = pyln.normalize.loudness(x.T, cur_loud, tgt).T.astype(np.float32)
            out_p = p.with_name(f"{p.stem}_{label}{p.suffix}")
            torchaudio.save(str(out_p), torch.from_numpy(y), sr)
            out_map[label] = str(out_p)
    except Exception as e:
        print(f"[master_targets] failed: {e}")
        return out_map
    return out_map
