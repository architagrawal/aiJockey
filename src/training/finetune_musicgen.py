"""
Fine-tune MusicGen for transition-bridge generation.

Target: MusicGen Medium (1.5B). Input conditioning replaced/augmented with
DJ-context (CLAP_pre, CLAP_post, tempo, target_section).
Output target: the transition_audio segment.

This is Phase B (MI300X). Skeleton — fill in once dataset is built.

License notes:
- MusicGen weights: CC-BY-NC. AGPL fork = research only, not commercial.
- For commercial: replace with Stable Audio Open (Stability Community License)
  or train from scratch.

Usage:
  python src/training/finetune_musicgen.py \
    --dataset_dir datasets/transitions_real \
    --base_model musicgen-medium \
    --out_dir checkpoints/musicgen_dj \
    --epochs 5 \
    --batch_size 2 \
    --lr 1e-5 \
    --use_qlora
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import torch
import torch.nn as nn

from transitions_data import TransitionDataset, collate_padded


class DJContextEncoder(nn.Module):
    """
    Maps (clap_pre, clap_post, tempo_pre, tempo_post) -> conditioning embedding
    compatible with MusicGen's cross-attention.

    Output shape: (batch, seq_len=4, d_model). Each token represents:
        [pre_summary, post_summary, tempo_pre, tempo_post]
    """

    def __init__(self, d_model: int = 1024, clap_dim: int = 512):
        super().__init__()
        self.clap_proj = nn.Linear(clap_dim, d_model)
        self.tempo_proj = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, clap_pre: torch.Tensor, clap_post: torch.Tensor,
                tempo_pre: torch.Tensor, tempo_post: torch.Tensor) -> torch.Tensor:
        # All inputs shape (B, ...). Project each to d_model.
        pre_emb = self.clap_proj(clap_pre)                    # (B, d)
        post_emb = self.clap_proj(clap_post)                  # (B, d)
        tempo_pre_emb = self.tempo_proj(tempo_pre.unsqueeze(-1) / 200.0)  # normalize
        tempo_post_emb = self.tempo_proj(tempo_post.unsqueeze(-1) / 200.0)
        seq = torch.stack([pre_emb, post_emb, tempo_pre_emb, tempo_post_emb], dim=1)
        return self.norm(seq)


def setup_model(base_name: str = 'medium', use_qlora: bool = True,
                device: str = 'cuda'):
    """Load MusicGen + attach DJ context encoder + apply QLoRA to top blocks."""
    from audiocraft.models import MusicGen
    print(f"loading MusicGen {base_name}...")
    mg = MusicGen.get_pretrained(f'facebook/musicgen-{base_name}')
    # Replace text conditioner cross-attention with our DJ context
    # NOTE: requires modifying mg.lm internals. Real implementation:
    #   1. Hook into condition_provider
    #   2. Register custom 'dj_context' provider that uses DJContextEncoder
    #   3. Disable text conditioner OR run alongside
    # See https://github.com/facebookresearch/audiocraft/blob/main/docs/CONDITIONERS.md
    # for the conditioner extension pattern.

    if use_qlora:
        try:
            from peft import LoraConfig, get_peft_model
            target_modules = ['out_proj', 'in_proj_weight']  # AC-specific names; adjust
            lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                  target_modules=target_modules,
                                  bias='none', task_type='CAUSAL_LM')
            mg.lm = get_peft_model(mg.lm, lora_cfg)
            print("applied QLoRA to mg.lm")
        except Exception as e:
            print(f"warn: QLoRA setup failed ({e}), proceeding full-tune")

    encoder = DJContextEncoder(d_model=1024).to(device)
    mg.lm.to(device)
    return mg, encoder


def fine_tune(args):
    """Main fine-tune loop. SKELETON — needs MI300X + audiocraft + working dataset."""
    print(f"=== fine-tune MusicGen-{args.base_model} on transition dataset ===")
    print(f"dataset:   {args.dataset_dir}")
    print(f"out_dir:   {args.out_dir}")
    print(f"epochs:    {args.epochs}")
    print(f"batch:     {args.batch_size}")
    print(f"lr:        {args.lr}")
    print(f"qlora:     {args.use_qlora}")

    ds = TransitionDataset(args.dataset_dir)
    if len(ds) == 0:
        raise SystemExit("empty dataset; build it first via dataset_builder.py")
    print(f"dataset:   {len(ds)} transition samples")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm GPU required for MusicGen fine-tune")

    mg, encoder = setup_model(args.base_model, args.use_qlora, 'cuda')

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_padded)

    optim = torch.optim.AdamW(
        list(mg.lm.parameters()) + list(encoder.parameters()),
        lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Skeleton training step (incomplete — TODO: integrate with audiocraft's
    # ConditionProvider, properly compute loss against EnCodec-tokenized target,
    # gradient accumulation for memory, etc.)
    for epoch in range(args.epochs):
        mg.lm.train(); encoder.train()
        for i, batch in enumerate(loader):
            target_audio = batch['transition_audio'].cuda()
            cond = encoder(
                batch['clap_pre'].cuda(),
                batch['clap_post'].cuda(),
                batch['tempo_pre'].cuda(),
                batch['tempo_post'].cuda(),
            )
            # TODO: tokenize target via mg.compression_model
            #       compute LM loss with cond as cross-attention input
            #       backward + step
            # placeholder log:
            if i == 0:
                print(f"epoch {epoch+1}: cond shape {tuple(cond.shape)}, "
                      f"target shape {tuple(target_audio.shape)}")
            break  # remove when real loop implemented
        sched.step()

    print("\nSKELETON COMPLETE. Real training step needs audiocraft "
          "ConditionProvider integration — see in-code TODOs.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', default='datasets/transitions_real')
    ap.add_argument('--base_model', default='medium', choices=['small', 'medium', 'large'])
    ap.add_argument('--out_dir', default='checkpoints/musicgen_dj')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--use_qlora', action='store_true', default=True)
    args = ap.parse_args()
    fine_tune(args)
