"""Smoke tests for Tier 1.5 CLAP compat head."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'training'))

import numpy as np
import torch
from clap_finetune import (
    DJCompatibilityHead, info_nce, project, project_batch, CLAP_DIM, EMBED_DIM,
)


def test_head_shape():
    head = DJCompatibilityHead()
    x = torch.randn(4, CLAP_DIM)
    z = head(x)
    assert z.shape == (4, EMBED_DIM)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_info_nce_decreases_when_pos_close():
    torch.manual_seed(0)
    B, K, D = 4, 5, 16
    a = torch.randn(B, D); a = a / a.norm(dim=-1, keepdim=True)
    p_close = a + 0.01 * torch.randn_like(a); p_close = p_close / p_close.norm(dim=-1, keepdim=True)
    p_far = torch.randn(B, D); p_far = p_far / p_far.norm(dim=-1, keepdim=True)
    n = torch.randn(B, K, D); n = n / n.norm(dim=-1, keepdim=True)
    loss_close = float(info_nce(a, p_close, n, temperature=0.07))
    loss_far = float(info_nce(a, p_far, n, temperature=0.07))
    assert loss_close < loss_far


def test_project_single():
    head = DJCompatibilityHead()
    head.eval()
    z = project(head, np.random.randn(CLAP_DIM).astype(np.float32))
    assert z.shape == (EMBED_DIM,)
    assert abs(np.linalg.norm(z) - 1.0) < 1e-5


def test_project_batch():
    head = DJCompatibilityHead()
    head.eval()
    z = project_batch(head, np.random.randn(7, CLAP_DIM).astype(np.float32))
    assert z.shape == (7, EMBED_DIM)
    norms = np.linalg.norm(z, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)
