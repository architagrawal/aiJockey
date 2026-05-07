"""
PyTorch Dataset wrapping the built transition samples.

Yields per-sample dicts ready for fine-tuning a generative model
(MusicGen / Stable Audio Open) on (context -> transition_audio) pairs.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


SR = 44100


class TransitionDataset(Dataset):
    """
    Loads (pre_audio, transition_audio, post_audio, features, technique_label)
    samples produced by dataset_builder.py.
    """

    def __init__(self, dataset_dir: str = 'datasets/transitions_real',
                 master_index: str = 'master_index.json'):
        self.root = Path(dataset_dir)
        index_path = self.root / master_index
        if not index_path.exists():
            raise FileNotFoundError(f"missing {index_path}. Run dataset_builder.py first.")
        with open(index_path) as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_wav(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        return wav

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        pre = self._load_wav(s['pre_path'])
        trans = self._load_wav(s['transition_path'])
        post = self._load_wav(s['post_path'])

        # Load CLAP features
        feat_path = Path(s['pre_path']).parent / f"{int(s['idx']):03d}_features.npz"
        if feat_path.exists():
            f = np.load(str(feat_path))
            clap_pre = torch.from_numpy(f['clap_pre']).float()
            clap_post = torch.from_numpy(f['clap_post']).float()
            tempo_pre = float(f['tempo_pre'])
            tempo_post = float(f['tempo_post'])
        else:
            clap_pre = torch.zeros(512)
            clap_post = torch.zeros(512)
            tempo_pre = tempo_post = 0.0

        return {
            'pre_audio': pre,                      # (2, T_pre)
            'transition_audio': trans,             # (2, T_trans) -- target
            'post_audio': post,                    # (2, T_post)
            'clap_pre': clap_pre,                  # (512,)
            'clap_post': clap_post,                # (512,)
            'tempo_pre': tempo_pre,
            'tempo_post': tempo_post,
            'technique_label': s.get('technique_guess', 'crossfade'),
            'mix_id': s['mix_id'],
            'idx': s['idx'],
        }


def collate_padded(batch: list[dict]) -> dict:
    """Pad audio tensors to max length in batch. Useful for variable-length training."""
    out: dict = {}
    keys_audio = ('pre_audio', 'transition_audio', 'post_audio')
    for k in keys_audio:
        max_len = max(b[k].size(-1) for b in batch)
        padded = torch.zeros(len(batch), 2, max_len)
        lengths = torch.zeros(len(batch), dtype=torch.long)
        for i, b in enumerate(batch):
            n = b[k].size(-1)
            padded[i, :, :n] = b[k]
            lengths[i] = n
        out[k] = padded
        out[f'{k}_lengths'] = lengths
    out['clap_pre'] = torch.stack([b['clap_pre'] for b in batch])
    out['clap_post'] = torch.stack([b['clap_post'] for b in batch])
    out['tempo_pre'] = torch.tensor([b['tempo_pre'] for b in batch])
    out['tempo_post'] = torch.tensor([b['tempo_post'] for b in batch])
    out['technique_label'] = [b['technique_label'] for b in batch]
    return out


if __name__ == '__main__':
    ds = TransitionDataset()
    print(f"loaded {len(ds)} transition samples")
    if len(ds) > 0:
        s = ds[0]
        print(f"first sample:")
        print(f"  pre_audio:        {tuple(s['pre_audio'].shape)}")
        print(f"  transition_audio: {tuple(s['transition_audio'].shape)}")
        print(f"  post_audio:       {tuple(s['post_audio'].shape)}")
        print(f"  clap_pre:         {tuple(s['clap_pre'].shape)}")
        print(f"  technique:        {s['technique_label']}")
