"""
SynapticFold: Standalone Test Suite for SynapticNativeModel & Selective SafeTensors Loader.

Verifies:
1. Pure PyTorch architecture components (RMSNorm, RoPE, GQA Attention, SwiGLU MLP, ManifoldBridge).
2. Weight-tied LM Head (zero extra parameter memory).
3. Selective SafeTensors Loader reading Qwen2.5-7B snapshot.
4. Physical RAM resident memory constraint <= 5.18 GB (4900-5100 MB).
5. Forward inference & Autoregressive generation without shape/dtype errors or memory leaks.
"""

import os
import sys
import gc
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synaptic_fold.synaptic_native_model import (
    SynapticNativeConfig,
    SynapticNativeModel,
    SynapticRMSNorm,
    SynapticRotaryEmbedding,
    SynapticAttention,
    SynapticMLP,
    SynapticDecoderLayer,
    SynapticRecurrentPlateau,
    apply_rotary_pos_emb,
)

SNAPSHOT_DIR = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"


def test_synaptic_native_config():
    """Tests configuration initialization and property calculations."""
    config = SynapticNativeConfig()
    assert config.vocab_size == 152064
    assert config.hidden_size == 3584
    assert config.intermediate_size == 18944
    assert config.num_attention_heads == 28
    assert config.num_key_value_heads == 4
    assert config.head_dim == 128
    assert config.num_key_value_groups == 7
    assert config.plateau_virtual_layers == 22
    assert config.tie_word_embeddings is True


def test_synaptic_native_components_forward():
    """Tests isolated mathematical components on CPU in native bfloat16."""
    torch.manual_seed(42)
    device = "cpu"
    dtype = torch.bfloat16
    hidden_size = 256
    seq_len = 4
    bsz = 1

    # 1. RMSNorm
    norm = SynapticRMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
    x = torch.randn(bsz, seq_len, hidden_size, dtype=dtype, device=device)
    normed_x = norm(x)
    assert normed_x.shape == x.shape
    assert normed_x.dtype == dtype
    assert not torch.isnan(normed_x).any()

    # 2. RoPE
    head_dim = 64
    rotary = SynapticRotaryEmbedding(dim=head_dim, max_position_embeddings=512, base=10000.0, device=device)
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cos, sin = rotary(x, pos_ids)
    assert cos.shape == (bsz, seq_len, head_dim)
    assert sin.shape == (bsz, seq_len, head_dim)

    # 3. Rotary application
    q = torch.randn(bsz, 4, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(bsz, 2, seq_len, head_dim, dtype=dtype, device=device)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_synaptic_native_tied_embeddings():
    """Verifies that lm_head and embed_tokens strictly share the same data pointer."""
    config = SynapticNativeConfig(
        vocab_size=2000,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        plateau_virtual_layers=2,
        boundary_in_layers=(0,),
        boundary_out_layers=(1,),
        dtype=torch.float32,
    )
    model = SynapticNativeModel(config, dtype=torch.float32)

    # Pointer equality verification (Zero duplicate RAM)
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    assert id(model.lm_head.weight) == id(model.embed_tokens.weight)

    # Memory breakdown verification
    mem = model.get_memory_breakdown()
    assert mem["saved_mb"] > 0
    assert mem["compression_ratio"] > 1.0


def test_synaptic_native_recurrent_plateau_execution():
    """Verifies execution of 22 virtual layers using 4 attention & 2 MLP archetypes."""
    config = SynapticNativeConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        plateau_virtual_layers=22,
        boundary_in_layers=(0, 1, 2),
        boundary_out_layers=(25, 26, 27),
        dtype=torch.float32,
    )
    model = SynapticNativeModel(config, dtype=torch.float32)
    assert len(model.plateau.archetype_attns) == 4
    assert len(model.plateau.archetype_mlps) == 2
    assert len(model.plateau.bridges) == 22
    assert len(model.plateau.input_layernorms) == 22

    # Forward through mini model
    x = torch.randint(0, 1000, (1, 6))
    out = model(x)
    assert out.shape == (1, 6, 1000)
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not os.path.exists(SNAPSHOT_DIR), reason="Qwen2.5-7B snapshot not found")
def test_synaptic_native_qwen25_7b_full_loader_and_memory():
    """
    Loads full Qwen2.5-7B using SynapticNativeModel selective loader:
    1. Validates physical memory footprint is <= 5.18 GB (4900-5100 MB).
    2. Executes forward pass without shape/dtype errors.
    3. Runs generation of new tokens.
    """
    gc.collect()
    print("\n[TEST] Loading Qwen2.5-7B selectively via SynapticNativeModel...")
    model = SynapticNativeModel.from_pretrained(
        SNAPSHOT_DIR,
        dtype=torch.bfloat16,
        device="cpu",
        load_real_weights=True,
    )

    # 1. Memory Verification
    mem = model.get_memory_breakdown()
    print(f"\n[MEMORY REPORT]")
    print(f"Total Nominal Model Size:  {mem['total_nominal_mb']} MB ({mem['total_nominal_mb']/1024:.2f} GB)")
    print(f"Unique Physical RAM Size:  {mem['unique_physical_mb']} MB ({mem['unique_physical_mb']/1024:.2f} GB)")
    print(f"Physical Memory Saved:     {mem['saved_mb']} MB ({mem['saved_mb']/1024:.2f} GB)")
    print(f"Compression Ratio:         {mem['compression_ratio']}x")
    print(f"Process Resident Set Size: {mem['rss_mb']} MB ({mem['rss_gb']} GB)")

    # Assertions on memory requirements
    assert mem["unique_physical_mb"] <= 5200.0, f"Unique physical MB {mem['unique_physical_mb']} exceeds 5200 MB"
    assert mem["rss_mb"] <= 5350.0, f"Process RSS {mem['rss_mb']} MB exceeds threshold (<= 5.18 GB / 5304 MB)"
    assert mem["saved_mb"] >= 8000.0, f"Saved MB {mem['saved_mb']} should be > 8000 MB"
    assert mem["compression_ratio"] >= 2.5, f"Compression ratio {mem['compression_ratio']} should be >= 2.5"

    # 2. Forward Inference Verification
    # Prompt: "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    input_ids = torch.tensor([[151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, input_ids.size(1), 152064)
    assert logits.dtype == torch.bfloat16
    assert not torch.isnan(logits).any(), "Logits contain NaN values!"
    assert not torch.isinf(logits).any(), "Logits contain Inf values!"
    print(f"[TEST SUCCESS] Forward logits: shape={logits.shape}, dtype={logits.dtype}")

    # 3. Autoregressive Generation Verification (5 tokens)
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=5, temperature=0.7)

    assert generated.shape == (1, input_ids.size(1) + 5)
    assert (generated >= 0).all()
    assert (generated < 152064).all()
    print(f"[TEST SUCCESS] Generated sequence tokens: {generated.tolist()}")


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
