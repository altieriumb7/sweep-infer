import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from synaptic_fold.intrinsic import (
    kneedle_knee_point,
    donoho_johnstone_threshold,
    IntrinsicSynapticMLPHook,
    IntrinsicModelWrapper,
)
from synaptic_fold.reservoir import (
    GlobalSynapticPool,
    ReservoirConfig,
)


def test_kneedle_knee_point_geometry():
    # Power-law activation distribution: first few values dominate energy
    d_ff = 1000
    ranks = torch.arange(1, d_ff + 1, dtype=torch.float32)
    gate_mag = 1.0 / (ranks ** 1.5)  # Heavy head, long tail
    
    mask = kneedle_knee_point(gate_mag.unsqueeze(0), min_pct=0.02)
    assert mask.shape == (1, d_ff)
    assert mask.dtype == torch.bool
    
    active_count = mask.sum().item()
    assert 20 <= active_count < d_ff // 2  # Knee correctly isolates the head
    
    # Active elements must have strictly greater or equal magnitude than silent ones
    min_active_val = gate_mag[mask[0]].min().item()
    max_silent_val = gate_mag[~mask[0]].max().item()
    assert min_active_val >= max_silent_val


def test_donoho_johnstone_universal_noise_filter():
    d_ff = 2000
    # Background Gaussian noise + 10 strong signal concepts
    noise = torch.randn(1, d_ff).abs() * 0.1
    noise[0, :10] += 5.0  # Clear signal peaks
    
    mask = donoho_johnstone_threshold(noise, min_active=5)
    assert mask.shape == (1, d_ff)
    assert mask[0, :10].all()  # All 10 strong signals preserved
    
    # Noise floor must suppress at least 90% of the pure noise channels
    assert mask.sum().item() <= 100


class MockSwiGLU(nn.Module):
    def __init__(self, d=128, d_ff=512):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def test_intrinsic_synaptic_mlp_hook_swiglu():
    torch.manual_seed(42)
    mlp = MockSwiGLU(d=64, d_ff=256)
    x = torch.randn(2, 64)
    
    # 1. Dense Baseline
    dense_out = mlp(x)
    
    # 2. Attach Hook with Knee Mode
    hook = IntrinsicSynapticMLPHook(mlp, mode="knee", min_pct=0.05)
    hook.attach()
    assert hook.is_attached
    
    sparse_out = mlp(x)
    assert sparse_out.shape == dense_out.shape
    assert hook.last_sparsity > 0.30  # At least 30% quiescent synapses
    
    # High directional alignment with dense representation
    cos_sim = F.cosine_similarity(dense_out, sparse_out, dim=-1).mean().item()
    assert cos_sim > 0.90
    
    # 3. Detach Hook
    hook.detach()
    assert not hook.is_attached
    restored_out = mlp(x)
    assert torch.allclose(restored_out, dense_out, atol=1e-6)


def test_intrinsic_synaptic_mlp_hook_noise_mode():
    torch.manual_seed(42)
    mlp = MockSwiGLU(d=64, d_ff=256)
    x = torch.randn(1, 64)
    
    hook = IntrinsicSynapticMLPHook(mlp, mode="noise")
    hook.attach()
    
    out = mlp(x)
    assert out.shape == (1, 64)
    assert hook.last_sparsity > 0.50  # Noise floor eliminates over 50%
    hook.detach()


def test_global_synaptic_pool_intrinsic_modes():
    torch.manual_seed(42)
    d = 32
    n = 128
    k_gate = torch.randn(n, d)
    u_up = torch.randn(n, d)
    v_down = torch.randn(d, n)
    
    pool = GlobalSynapticPool(k_gate=k_gate, u_up=u_up, v_down=v_down)
    h = torch.randn(1, d)
    
    # Test Knee mode
    delta_knee, sparsity_knee, idx_knee = pool.query_synapses(h, sparsity_mode="intrinsic_knee")
    assert delta_knee.shape == (1, d)
    assert sparsity_knee > 0.0
    
    # Test Noise mode
    delta_noise, sparsity_noise, idx_noise = pool.query_synapses(h, sparsity_mode="intrinsic_noise")
    assert delta_noise.shape == (1, d)
    assert sparsity_noise > 0.0


def test_intrinsic_model_wrapper():
    # Mock model with 4 layers
    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MockSwiGLU(d=32, d_ff=128)

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([MockLayer() for _ in range(4)])

    model = MockModel()
    x = torch.randn(1, 32)
    
    # Wrap all layers
    hooks = IntrinsicModelWrapper.wrap(model, mode="knee")
    assert len(hooks) == 4
    for h in hooks:
        assert h.is_attached
        
    # Execute forward pass through first layer
    _ = model.model.layers[0].mlp(x)
    assert hooks[0].last_sparsity > 0.0
    
    # Unwrap
    IntrinsicModelWrapper.unwrap(hooks)
    for h in hooks:
        assert not h.is_attached
