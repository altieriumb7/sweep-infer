import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from synaptic_fold.quantized_synaptic import (
    Int4BlockQuantizer,
    Int4SynapticLinear,
    Int4SynapticMLP,
)


def test_int4_pack_and_unpack_reconstruction():
    torch.manual_seed(42)
    out_f, in_f = 256, 128
    block_size = 32
    w = torch.randn(out_f, in_f, dtype=torch.float32)

    packed, scales = Int4BlockQuantizer.pack_tensor(w, block_size=block_size)
    assert packed.shape == (out_f, in_f // block_size, block_size // 2)
    assert packed.dtype == torch.uint8

    # Unpack all rows
    all_idx = torch.arange(out_f)
    w_recon = Int4BlockQuantizer.unpack_rows(packed, scales, all_idx, block_size=block_size)
    assert w_recon.shape == (out_f, in_f)

    # Cosine fidelity should be >= 0.99
    cos_sim = F.cosine_similarity(w.flatten(), w_recon.flatten(), dim=0).item()
    assert cos_sim >= 0.99


def test_selective_synaptic_linear_forward():
    torch.manual_seed(42)
    in_f, out_f = 128, 512
    linear = nn.Linear(in_f, out_f, bias=False)
    x = torch.randn(1, in_f)

    int4_lin = Int4SynapticLinear.from_float_linear(linear, block_size=32)
    assert int4_lin.memory_footprint_mb < (linear.weight.numel() * 4 / (1024 * 1024))

    # Test selective unpacking of 50 active concepts
    k = 50
    active_idx = torch.randperm(out_f)[:k]
    y_selective = int4_lin.forward_selective(x, active_idx)
    assert y_selective.shape == (1, k)

    # Compare against full dequantized slice
    y_full = int4_lin.forward_full(x)
    assert torch.allclose(y_selective, y_full[:, active_idx], atol=1e-4)


class MockSwiGLU(nn.Module):
    def __init__(self, d=64, d_ff=256):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def test_int4_synaptic_mlp_end_to_end():
    torch.manual_seed(42)
    d, d_ff = 64, 256
    float_mlp = MockSwiGLU(d=d, d_ff=d_ff)
    x = torch.randn(1, d)

    # 1. Convert to Int4 Synaptic MLP with Knee mode
    int4_mlp = Int4SynapticMLP.from_float_mlp(float_mlp, mode="knee", block_size=32)
    
    # Verify memory compression
    float_bytes = sum(p.numel() * 4 for p in float_mlp.parameters()) / (1024 * 1024)
    int4_bytes = int4_mlp.total_memory_mb
    assert int4_bytes < float_bytes / 5.0 # At least 5x smaller

    # Execute Synaptic Forward (Knee mode)
    y_syn = int4_mlp(x)
    assert y_syn.shape == (1, d)
    assert int4_mlp.last_sparsity > 0.40 # High quiescence

    # Numerical alignment with float baseline
    y_float = float_mlp(x)
    cos_sim = F.cosine_similarity(y_float, y_syn, dim=-1).item()
    assert cos_sim >= 0.90  # Knee mode preserves > 0.90 directional fidelity

    # 2. Test Full Dense Int4 reconstruction fidelity
    y_dense_int4 = int4_mlp.forward_full(x)
    dense_cos_sim = F.cosine_similarity(y_float, y_dense_int4, dim=-1).item()
    assert dense_cos_sim >= 0.98 # Quantization reconstruction fidelity is > 0.98
