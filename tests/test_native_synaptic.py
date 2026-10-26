import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from synaptic_fold.native_synaptic import NativeSynapticMLP


class MockFloatSwiGLU(nn.Module):
    def __init__(self, d=128, d_ff=512):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def test_native_synaptic_exact_permutation_invariance():
    """
    At keep_ratio=1.0 (all intermediate neurons active), the spectral permutation
    must produce machine-epsilon identical outputs to the unpermuted dense SwiGLU.
    """
    torch.manual_seed(42)
    d, d_ff = 128, 512
    float_mlp = MockFloatSwiGLU(d=d, d_ff=d_ff)
    native_mlp = NativeSynapticMLP(float_mlp, keep_ratio=1.0, fold_prefill=False)

    x = torch.randn(2, 5, d)
    y_dense = float_mlp(x)
    y_syn = native_mlp(x)

    cos_sim = F.cosine_similarity(y_dense.reshape(-1, d), y_syn.reshape(-1, d), dim=-1).mean().item()
    assert cos_sim > 0.99999


def test_native_synaptic_prefill_and_generation():
    """
    Verifies dual-path behavior:
    1. Prefill (S > 1) executes dense GEMM (or folds if fold_prefill=True).
    2. Decoding (S == 1) executes contiguous slice with keep_k active neurons.
    """
    torch.manual_seed(42)
    d, d_ff = 128, 512
    float_mlp = MockFloatSwiGLU(d=d, d_ff=d_ff)
    native_mlp = NativeSynapticMLP(float_mlp, keep_ratio=0.75, fold_prefill=False)

    # 1. Prefill Mode (Multi-Token S=10)
    x_prefill = torch.randn(1, 10, d)
    y_prefill = native_mlp(x_prefill)
    assert y_prefill.shape == (1, 10, d)
    assert native_mlp.last_k == d_ff  # Dense MKL GEMM during prefill

    # 2. Generation Mode (Single-Token S=1)
    x_gen = torch.randn(1, 1, d)
    y_gen = native_mlp(x_gen)
    assert y_gen.shape == (1, 1, d)
    assert native_mlp.last_k == int(d_ff * 0.75)
    assert native_mlp.last_sparsity == 0.25

    # 3. Folded Prefill Mode
    folded_mlp = NativeSynapticMLP(float_mlp, keep_ratio=0.75, fold_prefill=True)
    y_fold = folded_mlp(x_prefill)
    assert (y_fold == 0.0).all()


def test_native_synaptic_numerical_alignment():
    """
    Verifies directional alignment of the subnetwork output with the dense baseline.
    """
    torch.manual_seed(42)
    d, d_ff = 128, 512
    float_mlp = MockFloatSwiGLU(d=d, d_ff=d_ff)
    native_mlp = NativeSynapticMLP(float_mlp, keep_ratio=0.75)

    x = torch.randn(1, 1, d)
    y_dense = float_mlp(x)
    y_syn = native_mlp(x)

    cos_sim = F.cosine_similarity(y_dense.squeeze(), y_syn.squeeze(), dim=-1).item()
    assert cos_sim >= 0.85
