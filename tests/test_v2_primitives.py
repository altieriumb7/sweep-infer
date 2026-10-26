import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torch.nn as nn

from synaptic_fold.spiking import (
    SpikeIntegrateAndFire,
    TraceSwiGLU,
    WOSSAttention,
    fold_rmsnorm_affine,
    OptimalBrainSpiking,
)
from synaptic_fold.reservoir import (
    GlobalSynapticPool,
    ContinuousAttractorDynamics,
    ReservoirConfig,
)
from synaptic_fold.consensus import (
    AdaptiveConsensusCheck,
)
from synaptic_fold.engine import (
    InPlaceReservoirCache,
)


def test_fs_spike_exact_charge_conservation():
    neuron = SpikeIntegrateAndFire(v_th=1.0, beta=0.8, v_rest=0.0)
    current = torch.tensor([1.5, 0.2, 0.0])
    
    # Step 1
    s1, u1 = neuron(current)
    assert (s1 == torch.tensor([1.0, 0.0, 0.0])).all()
    # At t=1: u1 = 1.5, s1 = 1.0
    
    # Step 2: residue = 1.5 - 1.0 = 0.5; leak = 0.8 * 0.5 = 0.4; next u = 0.4 + 0.2 = 0.6
    s2, u2 = neuron(torch.tensor([0.2, 0.0, 0.0]), state=u1, last_spike=s1)
    assert torch.isclose(u2[0], torch.tensor(0.6), atol=1e-5)


def test_dual_trace_swiglu_curvature():
    swiglu = TraceSwiGLU(trace_fast=0.7, trace_slow=0.3, inhibition_gamma=0.4)
    gate = torch.ones(1, 16)
    up = torch.ones(1, 16)
    
    out1, state1 = swiglu(gate, up)
    assert out1.shape == (1, 16)
    assert (out1 > 0.0).all()
    
    # Step 2: verify asymmetric kinetics divergence
    out2, state2 = swiglu(gate, up, state=state1)
    # Fast trace accumulates faster than slow trace
    assert state2[0].mean() > state2[1].mean()


def test_woss_attention_energy_normalization():
    woss = WOSSAttention(saliency_threshold=4.6, top_k_nucleus=16)
    q = torch.randn(1, 4, 32)
    k = torch.randn(1, 100, 32)
    v = torch.randn(1, 100, 32)
    
    out, sparsity = woss(q, k, v)
    assert out.shape == (1, 4, 32)
    assert 0.0 <= sparsity <= 1.0
    # Output norm must remain bounded and not explode by 100x
    assert torch.norm(out).item() < 50.0


def test_fold_rmsnorm_affine_dimensions():
    # Linear: [out_features=64, in_features=32]
    linear_w = torch.randn(64, 32)
    norm_w = torch.ones(32) * 2.0
    
    folded = fold_rmsnorm_affine(linear_w, norm_w)
    assert folded.shape == (64, 32)
    assert torch.isclose(folded, linear_w * 2.0).all()


def test_zero_scatter_indexed_gemm():
    k_gate = torch.randn(100, 32)
    u_up = torch.randn(100, 32)
    v_down = torch.randn(32, 100)
    
    pool = GlobalSynapticPool(k_gate, u_up, v_down)
    h = torch.randn(1, 32)
    
    delta_h, sparsity, topk_idx = pool.query_synapses(h, sparsity_mode="fixed", k_ratio=0.30)
    assert delta_h.shape == (1, 32)
    assert abs(sparsity - 0.70) < 1e-4
    assert topk_idx.shape == (1, 30)


def test_elastic_synaptic_mobilization():
    k_gate = torch.randn(100, 32)
    u_up = torch.randn(100, 32)
    v_down = torch.randn(32, 100)
    
    pool = GlobalSynapticPool(k_gate, u_up, v_down)
    h = torch.randn(1, 32)
    
    # Elastic query with 90% energy retention
    delta_h, sparsity, active_idx = pool.query_synapses(h, sparsity_mode="elastic", p_energy_mass=0.90)
    assert delta_h.shape == (1, 32)
    # Sparsity ratio is dynamically determined, between 0.0 and 1.0
    assert 0.0 <= sparsity <= 1.0
    # Active index shape matches dynamic k
    k_active = active_idx.size(1)
    assert k_active >= 1 and k_active <= 100


def test_consensus_patience_prevents_premature_halting():
    head = nn.Linear(32, 100)
    checker = AdaptiveConsensusCheck(unembed_head=head, min_patience_steps=2)
    
    s0 = torch.randn(1, 4, 32)
    s1 = s0 + 0.001 * torch.randn(1, 4, 32)  # Tiny velocity
    
    # Step 1: MUST NOT converge because min_patience_steps is 2
    metrics1 = checker(s0, s1, current_step=1)
    assert metrics1.converged is False
    
    # Step 2: stable step
    s2 = s1 + 0.001 * torch.randn(1, 4, 32)
    metrics2 = checker(s1, s2, current_step=2)
    # Step 3: after patience, can converge if conditions hold
    s3 = s2 + 0.0001 * torch.randn(1, 4, 32)
    metrics3 = checker(s2, s3, current_step=3)
    assert metrics3.step_count == 3


def test_in_place_reservoir_cache():
    cache = InPlaceReservoirCache(max_seq_len=16, num_layers=2, num_heads=2, head_dim=8)
    
    # Overwrite slot 0 during 3 relaxation steps
    for step in range(3):
        k_step = torch.full((1, 2, 1, 8), fill_value=float(step))
        v_step = torch.full((1, 2, 1, 8), fill_value=float(step))
        cache.update_slot(layer_idx=0, k_step=k_step, v_step=v_step)
        
        # Valid seq_len (dim 2) must remain 1 throughout all micro-steps!
        k_ctx, v_ctx = cache.get_context(layer_idx=0)
        assert k_ctx.size(2) == 1
    
    # Value in slot 0 must be the final micro-step value
    assert (k_ctx == 2.0).all()
    
    # Advance token
    cache.advance_token()
    assert cache.current_pos == 1
