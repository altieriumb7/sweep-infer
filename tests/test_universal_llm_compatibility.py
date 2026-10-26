import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torch.nn as nn
from synaptic_fold.engine import InPlaceReservoirCache
from synaptic_fold.reservoir import GlobalSynapticPool
from synaptic_fold.spiking import fold_rmsnorm_affine
from synaptic_fold.adapter import UniversalModelAdapter

def test_gqa_in_place_cache():
    # Simulate LLaMA-3: 32 query heads, 8 KV heads, head_dim 128
    cache = InPlaceReservoirCache(
        max_seq_len=256,
        num_layers=16,
        num_heads=32,
        head_dim=128,
        num_kv_heads=8,
    )
    # LLaMA attention emits [1, 8, 1, 128]
    k_step = torch.randn(1, 8, 1, 128)
    v_step = torch.randn(1, 8, 1, 128)

    cache.update_slot(0, k_step, v_step)
    k_ctx, v_ctx = cache.get_context(0)
    assert k_ctx.shape == (1, 8, 1, 128)
    assert v_ctx.shape == (1, 8, 1, 128)
    print("test_gqa_in_place_cache PASSED")

def test_universal_activations_in_reservoir():
    d_model = 64
    n_synapses = 128
    h = torch.randn(2, d_model)
    k_gate = torch.randn(n_synapses, d_model)
    u_up = torch.randn(n_synapses, d_model)
    v_down = torch.randn(d_model, n_synapses)

    # 1. SwiGLU (LLaMA, Qwen, Mistral)
    pool_swiglu = GlobalSynapticPool(k_gate, u_up, v_down, activation_type="silu")
    out_swi, _, _ = pool_swiglu.query_synapses(h, k_ratio=0.5)
    assert out_swi.shape == (2, d_model)

    # 2. GeGLU (Gemma)
    pool_geglu = GlobalSynapticPool(k_gate, u_up, v_down, activation_type="gelu")
    out_ge, _, _ = pool_geglu.query_synapses(h, k_ratio=0.5)
    assert out_ge.shape == (2, d_model)

    # 3. Non-Gated MLP (GPT-2, Falcon, NeoX: u_up is None)
    pool_nongated = GlobalSynapticPool(k_gate, u_up=None, v_down=v_down, activation_type="gelu")
    out_ng, _, _ = pool_nongated.query_synapses(h, k_ratio=0.5)
    assert out_ng.shape == (2, d_model)
    print("test_universal_activations_in_reservoir PASSED")

def test_gemma_unit_offset_rmsnorm():
    d_in = 32
    d_out = 64
    w = torch.randn(d_out, d_in)
    norm_w = torch.tensor([0.1] * d_in) # in Gemma, scale is 1.0 + 0.1 = 1.1

    folded_standard = fold_rmsnorm_affine(w, norm_w, has_unit_offset=False)
    folded_gemma = fold_rmsnorm_affine(w, norm_w, has_unit_offset=True)

    assert torch.isclose(folded_standard, w * 0.1).all()
    assert torch.isclose(folded_gemma, w * 1.1).all()
    print("test_gemma_unit_offset_rmsnorm PASSED")

def test_universal_adapter_mock_models():
    # 1. Mock LLaMA-style model
    class MockLlamaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(1000, 64)
            self.model.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(6)])
            self.model.norm = nn.LayerNorm(64)
            self.lm_head = nn.Linear(64, 1000, bias=False)

    llama = MockLlamaModel()
    info = UniversalModelAdapter.inspect_model(llama)
    assert info["num_layers"] == 6
    assert info["embed_tokens"] is llama.model.embed_tokens
    assert info["lm_head"] is llama.lm_head

    engine = UniversalModelAdapter.wrap_model(llama, split_start=1, split_end=5)
    assert len(engine.sensory_layers) == 1
    assert len(engine.motor_layers) == 1
    print("test_universal_adapter_mock_models PASSED")

if __name__ == "__main__":
    test_gqa_in_place_cache()
    test_universal_activations_in_reservoir()
    test_gemma_unit_offset_rmsnorm()
    test_universal_adapter_mock_models()
