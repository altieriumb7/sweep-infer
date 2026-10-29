"""
Test: Compensated Top-K Sparse Synaptic Activation with Resting Potential.
Verifies if adding resting potential b_resting boosts Cosine Similarity >= 0.99 at 20% sparsity.
"""

import os
import sys
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SNAPSHOT_DIR = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
TRAJ_PATH = r"C:\Users\Umberto\synaptic-fold-v2\synaptic_fold\data\calibration_trajectories_700.pt"

def test_resting_potential():
    print("=" * 75)
    print("🔬 TEST: RIPRISTINO POTENZIALE DI RIPOSO (Resting Potential Compensation)")
    print("=" * 75)

    idx_path = os.path.join(SNAPSHOT_DIR, "model.safetensors.index.json")
    shard = "model-00002-of-00004.safetensors"
    t = load_file(os.path.join(SNAPSHOT_DIR, shard), device="cpu")
    W_gate = t["model.layers.12.mlp.gate_proj.weight"]
    W_up = t["model.layers.12.mlp.up_proj.weight"]
    W_down = t["model.layers.12.mlp.down_proj.weight"]
    norm_w = t["model.layers.12.post_attention_layernorm.weight"]

    d_inter, d = W_gate.shape

    data = torch.load(TRAJ_PATH, map_location="cpu", weights_only=True)
    X = data["X"] # (700, 3584)
    eps = 1e-6
    variance = X.float().pow(2).mean(-1, keepdim=True)
    X_norm = (X.float() * torch.rsqrt(variance + eps) * norm_w.float()).to(torch.bfloat16)

    # Dense ground truth
    with torch.no_grad():
        gate = F.linear(X_norm, W_gate)
        up = F.linear(X_norm, W_up)
        act = F.silu(gate.float()) * up.float()
        y_dense = F.linear(act.to(torch.bfloat16), W_down).float()

    # Split train (500) and test (200)
    X_train, X_test = X_norm[:500], X_norm[500:]
    act_train, act_test = act[:500], act[500:]
    y_dense_train, y_dense_test = y_dense[:500], y_dense[500:]

    for k in [1894, 2841, 3788, 5683]: # 10%, 15%, 20%, 30%
        pct = (k / d_inter) * 100
        
        # 1. Uncompensated Top-K
        topk_vals, topk_idx_train = torch.topk(act_train.abs(), k=k, dim=-1)
        sparse_act_train = torch.zeros_like(act_train)
        sparse_act_train.scatter_(dim=-1, index=topk_idx_train, src=act_train.gather(dim=-1, index=topk_idx_train))
        y_sparse_train = F.linear(sparse_act_train.to(torch.bfloat16), W_down).float()

        # Compute Resting Potential Vector: mean residual of dropped neurons
        b_resting = (y_dense_train - y_sparse_train).mean(dim=0, keepdim=True) # (1, 3584)
        
        # Compute optimal scalar gain alpha to adjust energy
        diff = y_dense_train - b_resting
        alpha = (diff * y_sparse_train).sum() / (y_sparse_train * y_sparse_train).sum().clamp(min=1e-6)

        # 2. Test on UNSEEN 200 tokens
        topk_vals, topk_idx_test = torch.topk(act_test.abs(), k=k, dim=-1)
        sparse_act_test = torch.zeros_like(act_test)
        sparse_act_test.scatter_(dim=-1, index=topk_idx_test, src=act_test.gather(dim=-1, index=topk_idx_test))
        y_sparse_test_raw = F.linear(sparse_act_test.to(torch.bfloat16), W_down).float()

        # Without compensation
        cos_raw = F.cosine_similarity(y_dense_test, y_sparse_test_raw, dim=-1).mean().item()
        l2_raw = (torch.norm(y_dense_test - y_sparse_test_raw, dim=-1) / torch.norm(y_dense_test, dim=-1)).mean().item()

        # With resting potential + scale compensation:
        y_sparse_comp = alpha * y_sparse_test_raw + b_resting
        cos_comp = F.cosine_similarity(y_dense_test, y_sparse_comp, dim=-1).mean().item()
        l2_comp = (torch.norm(y_dense_test - y_sparse_comp, dim=-1) / torch.norm(y_dense_test, dim=-1)).mean().item()

        print(f"  • Top-{k} ({pct:4.1f}% neuroni attivi):")
        print(f"      Senza comp:  CosSim = {cos_raw:.5f} | Errore L2 = {l2_raw:.4f}")
        print(f"      Con RestPot: CosSim = {cos_comp:.5f} | Errore L2 = {l2_comp:.4f} (Target >= 0.99: {'RAGGIUNTO' if cos_comp >= 0.99 else 'QUASI'})")
        print("-" * 65)

if __name__ == "__main__":
    test_resting_potential()
