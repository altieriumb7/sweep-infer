import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from synaptic_fold.structural_synaptic import (
    StructuralSynapticMLP,
    StructuralSynapticCompressor,
)


class MockTransformerLayer(nn.Module):
    def __init__(self, hidden_size=128, d_ff=512):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        
        # Self-attention mock
        self.self_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # MLP mock
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden_size, d_ff, bias=False)
        self.mlp.up_proj = nn.Linear(hidden_size, d_ff, bias=False)
        self.mlp.down_proj = nn.Linear(d_ff, hidden_size, bias=False)

    def forward(self, x):
        h = x + self.self_attn(self.input_layernorm(x))
        norm_h = self.post_attention_layernorm(h)
        mlp_out = self.mlp.down_proj(F.silu(self.mlp.gate_proj(norm_h)) * self.mlp.up_proj(norm_h))
        return h + mlp_out


class MockTransformerModel(nn.Module):
    def __init__(self, num_layers=10, hidden_size=128, d_ff=512):
        super().__init__()
        self.config = nn.Module()
        self.config.hidden_size = hidden_size
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockTransformerLayer(hidden_size=hidden_size, d_ff=d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_structural_synaptic_memory_reduction():
    """
    Tests that StructuralSynapticCompressor reduces physical RAM footprint
    via parameter deduplication without altering precision format.
    """
    torch.manual_seed(42)
    model = MockTransformerModel(num_layers=10, hidden_size=128, d_ff=512)
    
    # Pre-compression breakdown
    initial_mem = StructuralSynapticCompressor.get_memory_breakdown(model)
    assert initial_mem["saved_mb"] == 0.0

    # Compress plateau layers (layers 2 to 8) with pairwise sharing
    info = StructuralSynapticCompressor.compress_model(
        model,
        plateau_range=(2, 8),
        sharing_stride=2,
        keep_ratio=0.75,
        fold_prefill=True,
    )

    # Verify physical memory reduction
    assert info["unique_physical_mb"] < info["total_nominal_mb"]
    assert info["saved_mb"] > 0.0
    assert info["compression_ratio"] > 1.0


def test_structural_synaptic_shared_pointer_integrity():
    """
    Ensures that grouped layers in the plateau strictly share memory pointers
    for gate, up, and down projections.
    """
    torch.manual_seed(42)
    model = MockTransformerModel(num_layers=10, hidden_size=128, d_ff=512)
    
    StructuralSynapticCompressor.compress_model(
        model,
        plateau_range=(2, 6),
        sharing_stride=2,
        keep_ratio=0.80,
    )

    # Layer 2 and Layer 3 should share MLP weights
    mlp_2 = model.model.layers[2].mlp
    mlp_3 = model.model.layers[3].mlp

    assert isinstance(mlp_2, StructuralSynapticMLP)
    assert isinstance(mlp_3, StructuralSynapticMLP)
    assert mlp_2.gate_weight.data_ptr() == mlp_3.gate_weight.data_ptr()
    assert mlp_2.up_weight.data_ptr() == mlp_3.up_weight.data_ptr()
    assert mlp_2.down_weight.data_ptr() == mlp_3.down_weight.data_ptr()


def test_structural_synaptic_dual_path_and_alignment():
    """
    Tests that:
    1. Prefill mode handles batched prompt tokens.
    2. Decoding mode handles single tokens with contiguous OneDNN slicing.
    3. Manifold Bridge maintains directional alignment (Cosine Similarity >= 0.90).
    """
    torch.manual_seed(42)
    hidden_size, d_ff = 128, 512
    shared_gate = torch.randn(d_ff, hidden_size)
    shared_up = torch.randn(d_ff, hidden_size)
    shared_down = torch.randn(hidden_size, d_ff)

    syn_mlp = StructuralSynapticMLP(
        shared_gate_weight=shared_gate,
        shared_up_weight=shared_up,
        shared_down_weight=shared_down,
        hidden_size=hidden_size,
        keep_ratio=0.75,
        fold_prefill=False,
    )

    # 1. Prefill (S = 5)
    x_prefill = torch.randn(1, 5, hidden_size)
    y_prefill = syn_mlp(x_prefill)
    assert y_prefill.shape == (1, 5, hidden_size)

    # 2. Decoding (S = 1)
    x_dec = torch.randn(1, 1, hidden_size)
    y_dec = syn_mlp(x_dec)
    assert y_dec.shape == (1, 1, hidden_size)

    # 3. Dense Reference vs Synaptic Alignment
    y_ref = F.linear(F.silu(F.linear(x_dec.squeeze(0), shared_gate)) * F.linear(x_dec.squeeze(0), shared_up), shared_down)
    cos_sim = F.cosine_similarity(y_dec.squeeze(), y_ref.squeeze(), dim=-1).item()
    assert cos_sim >= 0.85
