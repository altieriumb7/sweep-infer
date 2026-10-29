import os
import sys
import torch
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.synaptic_cortical_engine import (
    FusedSynapticLinear,
    FusedSynapticAttention,
    FusedSynapticMLP,
    SynapticInPlaceKVCache,
    SynapticCorticalEngine,
    setup_optimal_cpu_threads,
)


def test_fused_linear():
    setup_optimal_cpu_threads()
    in_feat = 3584
    out_feat = 4608
    linear = FusedSynapticLinear(in_feat, out_feat, has_bias=True)
    w = torch.randn(out_feat, in_feat, dtype=torch.bfloat16)
    b = torch.randn(out_feat, dtype=torch.bfloat16)
    linear.load_from_weight(w, b)

    x = torch.randn(1, in_feat, dtype=torch.bfloat16)
    out = linear(x)
    assert out.shape == (1, out_feat)
    assert out.dtype == torch.bfloat16


def test_kv_cache_in_place():
    num_layers = 2
    num_kv_heads = 4
    head_dim = 128
    cache = SynapticInPlaceKVCache(num_layers, num_kv_heads, head_dim, max_seq_len=64)

    # Step 1: prefill 5 tokens
    k1 = torch.randn(1, num_kv_heads, 5, head_dim, dtype=torch.bfloat16)
    v1 = torch.randn(1, num_kv_heads, 5, head_dim, dtype=torch.bfloat16)
    valid_k, valid_v = cache.update(0, k1, v1)
    cache.advance_step(5)
    assert valid_k.shape == (1, num_kv_heads, 5, head_dim)

    # Step 2: decode 1 token
    k2 = torch.randn(1, num_kv_heads, 1, head_dim, dtype=torch.bfloat16)
    v2 = torch.randn(1, num_kv_heads, 1, head_dim, dtype=torch.bfloat16)
    valid_k, valid_v = cache.update(0, k2, v2)
    cache.advance_step(1)
    assert valid_k.shape == (1, num_kv_heads, 6, head_dim)
    assert cache.seq_len == 6


def test_fused_mlp():
    hidden = 3584
    intermediate = 18944
    mlp = FusedSynapticMLP(hidden, intermediate)
    w_g = torch.randn(intermediate, hidden, dtype=torch.bfloat16)
    w_u = torch.randn(intermediate, hidden, dtype=torch.bfloat16)
    w_d = torch.randn(hidden, intermediate, dtype=torch.bfloat16)
    mlp.load_from_separate(w_g, w_u, w_d)

    x = torch.randn(1, hidden, dtype=torch.bfloat16)
    out = mlp(x)
    assert out.shape == (1, hidden)


if __name__ == "__main__":
    test_fused_linear()
    print("test_fused_linear passed!")
    test_kv_cache_in_place()
    print("test_kv_cache_in_place passed!")
    test_fused_mlp()
    print("test_fused_mlp passed!")
