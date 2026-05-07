"""Phrase boundary detection + snap helpers.

DJs almost always transition on 16- or 32-bar phrase boundaries. Most auto-DJs
ignore phrase. Enforcing this is a key quality differentiator.
"""
from __future__ import annotations
import numpy as np


def snap_to_phrase(t_sec: float, downbeats: list[float],
                   bars_per_phrase: int = 16) -> float:
    """Snap time to nearest phrase boundary (every Nth downbeat)."""
    if not downbeats:
        return t_sec
    phrase_dbs = downbeats[::bars_per_phrase]
    if not phrase_dbs:
        return t_sec
    arr = np.asarray(phrase_dbs)
    return float(arr[np.argmin(np.abs(arr - t_sec))])


def detect_phrase_length(downbeats: list[float], energy_curve: list[float],
                         energy_hop_hz: float = 10.0) -> int:
    """
    Heuristic phrase length detection (16 vs 32 bars).
    Returns 16 if too few bars to decide.
    """
    if len(downbeats) < 64:
        return 16
    energy = np.asarray(energy_curve, dtype=np.float32)
    if energy.size == 0:
        return 16
    bar_energies: list[float] = []
    for i in range(len(downbeats) - 1):
        s = int(downbeats[i] * energy_hop_hz)
        e = int(downbeats[i + 1] * energy_hop_hz)
        if 0 <= s < e <= energy.size:
            bar_energies.append(float(energy[s:e].mean()))
    if len(bar_energies) < 64:
        return 16
    arr = np.asarray(bar_energies)

    def autocorr(x: np.ndarray, lag: int) -> float:
        if lag <= 0 or lag >= len(x):
            return 0.0
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

    return 32 if autocorr(arr, 32) > autocorr(arr, 16) + 0.1 else 16
