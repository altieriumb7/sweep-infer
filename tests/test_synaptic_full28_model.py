import sys
ROOT_DIR = r"C:\Users\Umberto\synaptic-fold-v2"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import pytest
from synaptic_fold.synaptic_full28_model import (
    SynapticInt4Linear,
    SynapticDecoderLayer,
    SynapticRMSNorm,
    SynapticRotaryEmbedding,
)

def test_synaptic_int4_linear_quant_and_forward():
    torch.manual_seed(42)
    in_features = 256
    out_features = 512
    x = torch.randn(1, 1, in_features, dtype=torch.bfloat16)

    # Reference float linear
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    ref_out = torch.matmul(x, weight.t())

    # INT4 Synaptic linear
    int4_layer = SynapticInt4Linear(in_features, out_features, bias=False)
    int4_layer.load_from_weight(weight)

    out = int4_layer(x)
    assert out.shape == (1, 1, out_features)

    # Check cosine similarity between int4 output and unquantized float
    cos_sim = torch.cosine_similarity(out.float(), ref_out.float(), dim=-1).mean().item()
    assert cos_sim > 0.95, f"Cosine similarity too low: {cos_sim}"

def test_synaptic_decoder_layer_forward():
    torch.manual_seed(42)
    hidden_size = 256
    intermediate_size = 512
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads

    layer = SynapticDecoderLayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )

    # Quantize dummy weights
    for proj in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.self_attn.o_proj]:
        w = torch.randn(proj.out_features, proj.in_features, dtype=torch.bfloat16)
        proj.load_from_weight(w)

    for proj in [layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj]:
        w = torch.randn(proj.out_features, proj.in_features, dtype=torch.bfloat16)
        proj.load_from_weight(w)

    # Create dummy rotary embeddings
    seq_len = 8
    rotary = SynapticRotaryEmbedding(dim=head_dim, max_position_embeddings=128)
    pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(pos_ids, dtype=torch.bfloat16)

    # Prefill forward
    x = torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16)
    out, kv = layer(x, cos, sin, use_cache=True)

    assert out.shape == (1, seq_len, hidden_size)
    assert kv[0].shape == (1, num_kv_heads, seq_len, head_dim)
    assert kv[1].shape == (1, num_kv_heads, seq_len, head_dim)

    # Decoding single token step (S=1)
    x_next = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16)
    pos_ids_dec = torch.tensor([[seq_len]], dtype=torch.long)
    cos_dec, sin_dec = rotary(pos_ids_dec, dtype=torch.bfloat16)

    out_dec, kv_dec = layer(x_next, cos_dec, sin_dec, past_key_value=kv, use_cache=True)
    assert out_dec.shape == (1, 1, hidden_size)
    assert kv_dec[0].shape == (1, num_kv_heads, seq_len + 1, head_dim)
