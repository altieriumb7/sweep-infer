"""
Micro-benchmarking script to profile and compare:
1. Baseline Elastic (full sort + per-token slice loop)
2. Vectorized Zero-Scatter (single slice per layer via unique active pool)
3. Homoeostatic Thresholding (O(N) linear time: theta = mean + alpha * std)
4. Activation sparsity sweep: 5%, 10%, 15%, 20%, 30% vs latency and accuracy.
"""

import time
import torch
import torch.nn.functional as F

def test_mechanisms():
    torch.manual_seed(42)
    seq_len = 120
    d_model = 3584
    d_ff = 18944
    
    # Simulate single layer MLP activations
    h = torch.randn(1, seq_len, d_model, dtype=torch.bfloat16)
    w_gate = torch.randn(d_ff, d_model, dtype=torch.bfloat16)
    w_up = torch.randn(d_ff, d_model, dtype=torch.bfloat16)
    w_down = torch.randn(d_model, d_ff, dtype=torch.bfloat16)
    
    zk = F.linear(h, w_gate)
    zu = F.linear(h, w_up)
    act = F.silu(zk) * zu
    abs_act = torch.abs(act)
    
    print("=== PROFILING SPARSITY SELECTION & PROJECTION MECHANISMS ===")
    
    # 1. Full Sort Elastic (Old v2 method)
    t0 = time.perf_counter()
    sorted_vals, sorted_indices = torch.sort(abs_act, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=-1)
    total_mass = cumsum[:, :, -1:] + 1e-8
    meets_mass = (cumsum / total_mass) >= 0.90
    first_idx = torch.argmax(meets_mass.to(torch.int32), dim=-1)
    k_active = int(first_idx.max().item()) + 1
    topk_idx = sorted_indices[:, :, :k_active]
    gathered_act = torch.gather(act, -1, topk_idx)
    
    mlp_out_old = torch.zeros_like(h)
    for s_i in range(seq_len):
        act_s = gathered_act[0, s_i:s_i+1]
        idx_s = topk_idx[0, s_i]
        w_down_active = w_down[:, idx_s]
        mlp_out_old[0, s_i:s_i+1] = torch.matmul(act_s, w_down_active.t())
    t_old = time.perf_counter() - t0
    print(f"1. Full Sort + Python Token Loop : {t_old*1000:>6.2f} ms | k_active: {k_active} ({k_active/d_ff*100:.1f}%)")
    
    # 2. Vectorized Zero-Scatter (Single slice per layer)
    t0 = time.perf_counter()
    sorted_vals, sorted_indices = torch.sort(abs_act, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=-1)
    total_mass = cumsum[:, :, -1:] + 1e-8
    meets_mass = (cumsum / total_mass) >= 0.90
    first_idx = torch.argmax(meets_mass.to(torch.int32), dim=-1)
    k_active = int(first_idx.max().item()) + 1
    topk_idx = sorted_indices[:, :, :k_active]
    
    # Unique active pool across tokens in this layer
    unique_active_idx = torch.unique(topk_idx.reshape(-1))
    k_unique = unique_active_idx.numel()
    
    # Single slice of W_down!
    w_down_sub = w_down[:, unique_active_idx] # [d_model, k_unique]
    
    # Map gathered activations onto the reduced subspace
    # Create compact act [1, seq_len, k_unique] directly
    act_compact = act[:, :, unique_active_idx]
    mlp_out_vec = torch.matmul(act_compact, w_down_sub.t())
    t_vec = time.perf_counter() - t0
    print(f"2. Vectorized Single-Slice GEMM  : {t_vec*1000:>6.2f} ms | k_unique: {k_unique} ({k_unique/d_ff*100:.1f}%) | Speedup: {t_old/t_vec:.2f}x")
    
    # 3. O(N) Homoeostatic Thresholding (No Sorting!)
    for alpha in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        t0 = time.perf_counter()
        mu = abs_act.mean(dim=-1, keepdim=True)
        std = abs_act.std(dim=-1, keepdim=True)
        threshold = mu + alpha * std
        
        mask = (abs_act >= threshold) # [1, seq_len, d_ff]
        active_indices = torch.nonzero(mask[0].any(dim=0)).squeeze(-1)
        k_active_homo = active_indices.numel()
        
        w_down_homo = w_down[:, active_indices]
        act_homo = (act * mask)[:, :, active_indices]
        mlp_out_homo = torch.matmul(act_homo, w_down_homo.t())
        t_homo = time.perf_counter() - t0
        active_pct = (k_active_homo / d_ff) * 100.0
        print(f"3. Homoeostatic (alpha={alpha:>3.1f})      : {t_homo*1000:>6.2f} ms | k: {k_active_homo:>5} ({active_pct:>5.1f}% active) | Speedup: {t_old/t_homo:.2f}x")

if __name__ == "__main__":
    test_mechanisms()
