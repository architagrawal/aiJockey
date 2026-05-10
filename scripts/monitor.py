"""Pipeline health monitor — queue depth + GPU + cost ticker.

Runs in tmux window 9. Prints a refreshing dashboard so user (or other agents)
see at a glance what stages are alive, queue depths, GPU utilization, total
spend so far.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.common import scratch_dir, queue_count


PRICE_PER_HR = 1.99
QUEUES = [
    ('raw',           'audio downloads',          '*'),
    ('cache',         'analyzed clips',           '*.json'),
    ('transitions',   'DJ-set triplets',          '**/t*.json'),
    ('embed',         'CLAP index files',         '*'),
    ('renders',       'self-play renders',        '*/k*/mix.wav'),
    ('preferences',   'preference jsonl',         '*.jsonl'),
    ('models',        'checkpoints',              '*'),
    ('output',        'final mixes',              '*/card.json'),
]


def _gpu_smi() -> str:
    if not shutil.which('rocm-smi'):
        return 'rocm-smi: n/a'
    try:
        out = subprocess.check_output(
            ['rocm-smi', '--showuse', '--showmemuse'],
            timeout=5, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return 'rocm-smi: error'
    # Compact relevant lines
    lines = [ln for ln in out.splitlines()
             if 'GPU use' in ln or 'GPU memory use' in ln or 'GPU%' in ln]
    return ' | '.join(lines[:4]) or out.strip()[:200]


def _disk_free() -> str:
    try:
        out = subprocess.check_output(['df', '-h', str(scratch_dir())],
                                       text=True, stderr=subprocess.DEVNULL)
        return out.splitlines()[-1].split()[3]
    except Exception:
        return 'n/a'


def render_dashboard(start_t: float) -> str:
    rows = []
    for q, desc, pat in QUEUES:
        n = queue_count(scratch_dir(q), pat)
        rows.append(f"  {q:15s} {desc:25s} {n:>6d}")
    elapsed_h = (time.time() - start_t) / 3600.0
    spend = elapsed_h * PRICE_PER_HR
    parts = [
        '=' * 70,
        f"AiJockey Pipeline   uptime={elapsed_h:.2f}h   spend=${spend:.2f}",
        f"GPU: {_gpu_smi()}",
        f"scratch free: {_disk_free()}",
        'queues:',
        *rows,
        '=' * 70,
    ]
    return '\n'.join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', type=float, default=30.0)
    args = ap.parse_args()
    start_t = time.time()
    while True:
        print(render_dashboard(start_t), flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
