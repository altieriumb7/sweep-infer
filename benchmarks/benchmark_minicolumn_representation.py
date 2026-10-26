import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

def benchmark_minicolumn_kernels():
    d_model = 3584
    n_ffn = 18944
    seq_len = 90
    target_pct = 0.05
    k_synapses = int(n_ffn * target_pct)  # ~947 synapses

    print("=" * 80)
    print(" NOVEL REPRESENTATION BENCHMARK: CORTICAL MINICOLUMNS VS UNSTRUCTURED TOP-K")
    print(f" Hidden Dim: {d_model} | Total Synapses: {n_ffn} | Sequence: {seq_len} tokens")
    print(f" Target Sparsity: {target_pct*100:.1f}% active ({k_synapses} active synapses)")
    print("=" * 80)

    # Test weights and activations
    h = torch.randn(seq_len, d_model, dtype=torch.bfloat16)
    w_down = torch.randn(d_model, n_ffn, dtype=torch.bfloat16)
    act = torch.randn(seq_len, n_ffn, dtype=torch.bfloat16)

    # 1. Unstructured Top-K + Scatter
    # Warmup
    for _ in range(5):
        vals, idx = torch.topk(torch.abs(act), k=k_synapses, dim=-1)
        g_act = torch.gather(act, -1, idx)
        sp_act = torch.zeros_like(act)
        sp_act.scatter_(-1, idx, g_act)
        out1 = torch.matmul(sp_act, w_down.t())

    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        vals, idx = torch.topk(torch.abs(act), k=k_synapses, dim=-1)
        g_act = torch.gather(act, -1, idx)
        sp_act = torch.zeros_like(act)
        sp_act.scatter_(-1, idx, g_act)
        out1 = torch.matmul(sp_act, w_down.t())
    t_unstructured = (time.perf_counter() - t0) / iters

    print(f"\n[1] Unstructured Top-K (Scatter + Dense GEMM): {t_unstructured*1000:.2f} ms")

    # 2. Block-Sparse Cortical Minicolumns (Block Size B = 32 and B = 64)
    for block_size in [32, 64]:
        num_blocks = n_ffn // block_size
        k_blocks = max(1, int(num_blocks * target_pct))
        truncated_n = num_blocks * block_size

        act_trunc = act[:, :truncated_n].view(seq_len, num_blocks, block_size)
        w_down_trunc = w_down[:, :truncated_n] # [d_model, num_blocks * block_size]

        # Warmup
        for _ in range(5):
            # Compute minicolumn energy: L1 norm across block
            block_energy = torch.sum(torch.abs(act_trunc), dim=-1)  # [seq_len, num_blocks]
            top_blocks = torch.topk(block_energy, k=k_blocks, dim=-1).indices # [seq_len, k_blocks]
            
            # Mask inactive minicolumns
            block_mask = torch.zeros((seq_len, num_blocks, 1), dtype=act.dtype)
            block_mask.scatter_(1, top_blocks.unsqueeze(-1), 1.0)
            sparse_col_act = (act_trunc * block_mask).view(seq_len, truncated_n)
            out_block = torch.matmul(sparse_col_act, w_down_trunc.t())

        t0 = time.perf_counter()
        for _ in range(iters):
            block_energy = torch.sum(torch.abs(act_trunc), dim=-1)  # [seq_len, num_blocks]
            top_blocks = torch.topk(block_energy, k=k_blocks, dim=-1).indices # [seq_len, k_blocks]
            block_mask = torch.zeros((seq_len, num_blocks, 1), dtype=act.dtype)
            block_mask.scatter_(1, top_blocks.unsqueeze(-1), 1.0)
            sparse_col_act = (act_trunc * block_mask).view(seq_len, truncated_n)
            out_block = torch.matmul(sparse_col_act, w_down_trunc.t())
        t_block = (time.perf_counter() - t0) / iters

        # Check energy retention
        dense_out = torch.matmul(act[:, :truncated_n], w_down_trunc.t())
        rel_energy = (torch.norm(out_block.float()) / torch.norm(dense_out.float())).item()

        print(f"[2] Block-Sparse Minicolumns (B={block_size}, K_blocks={k_blocks}/{num_blocks}): {t_block*1000:.2f} ms | Energy Retention: {rel_energy*100:.1f}%")

if __name__ == "__main__":
    benchmark_minicolumn_kernels()
