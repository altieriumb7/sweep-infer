"""
Test: Low-Rank Contextual Neuron Router vs Co-Activation Clustering.
Evaluates how accurately a lightweight rank-64 / rank-128 router predicts top-k neurons,
and measures the resulting output Cosine Similarity.
"""

import os
import sys
import time
import json
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SNAPSHOT_DIR = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
TRAJ_PATH = r"C:\Users\Umberto\synaptic-fold-v2\synaptic_fold\data\calibration_trajectories_700.pt"


def test_low_rank_router():
    print("=" * 75)
    print("🔬 TEST: ROUTER NEURONALE LOW-RANK AD ALTA FEDELTÀ (Rank-64 / Rank-128)")
    print("=" * 75)

    idx_path = os.path.join(SNAPSHOT_DIR, "model.safetensors.index.json")
    with open(idx_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    
    shard = weight_map["model.layers.12.mlp.gate_proj.weight"]
    t = load_file(os.path.join(SNAPSHOT_DIR, shard), device="cpu")
    W_gate = t["model.layers.12.mlp.gate_proj.weight"] # (18944, 3584)
    W_up = t["model.layers.12.mlp.up_proj.weight"]
    W_down = t["model.layers.12.mlp.down_proj.weight"]
    norm_w = t["model.layers.12.post_attention_layernorm.weight"]

    d_inter, d_model = W_gate.shape

    # Load 700 tokens
    data = torch.load(TRAJ_PATH, map_location="cpu", weights_only=True)
    X = data["X"] # (700, 3584)
    N = X.shape[0]

    # Pre-normalize X
    eps = 1e-6
    variance = X.float().pow(2).mean(-1, keepdim=True)
    X_norm = (X.float() * torch.rsqrt(variance + eps) * norm_w.float()).to(torch.bfloat16)

    # Compute dense ground truth activations and output
    print(f"[Dataset] Calcolo attivazioni ground truth su {N} token...")
    with torch.no_grad():
        gate = F.linear(X_norm, W_gate)
        up = F.linear(X_norm, W_up)
        act = F.silu(gate.float()) * up.float() # (N, 18944)
        y_dense = F.linear(act.to(torch.bfloat16), W_down).float() # (N, 3584)

    # Target for router: magnitude of activation |act|
    # Split train (500 tokens) / test (200 tokens)
    X_train, X_test = X_norm[:500].float(), X_norm[500:].float()
    act_train, act_test = act[:500], act[500:]
    y_dense_train, y_dense_test = y_dense[:500], y_dense[500:]
    N_test = X_test.shape[0]

    # Train a low-rank router via Ridge SVD on (X_train -> |act_train|)
    # |act_train|: (500, 18944). We want W_router (3584 x rank), V_router (rank x 18944)
    print("\n[Fase 1] Addestramento Router Low-Rank (SVD + Ridge Regression su traiettorie)...")
    for rank in [64, 128]:
        t0 = time.perf_counter()
        # Truncated SVD of target activations
        # act_train: (500, 18944)
        U_a, S_a, Vh_a = torch.linalg.svd(act_train.abs(), full_matrices=False)
        # Low rank projection of target: Z = act_train @ Vh_a[:rank].T  (500, rank)
        V_router = Vh_a[:rank] # (rank, 18944)
        Z_train = torch.matmul(act_train.abs(), V_router.t()) # (500, rank)

        # Solve Ridge Regression: X_train @ W_router ~ Z_train
        # W_router = (X^T X + lambda I)^-1 X^T Z
        reg_lambda = 1e-2
        d = X_train.shape[1]
        XtX = torch.matmul(X_train.t(), X_train)
        reg_I = reg_lambda * torch.trace(XtX) / d * torch.eye(d)
        XtZ = torch.matmul(X_train.t(), Z_train)
        W_router = torch.linalg.solve(XtX + reg_I, XtZ) # (3584, rank)
        dt_train = time.perf_counter() - t0
        print(f"  -> Router Rank-{rank} calibrato in {dt_train:.2f}s!")

        # Evaluate on UNSEEN test set (200 tokens)
        with torch.no_grad():
            # Router prediction: (N_test, rank) @ (rank, 18944) -> (N_test, 18944)
            z_pred = torch.matmul(X_test, W_router)
            scores_pred = torch.matmul(z_pred, V_router) # (N_test, 18944)

        print(f"\n  🎯 Validazione Router Rank-{rank} su {N_test} token mai visti:")
        for k in [1894, 2841, 3788, 5683]: # 10%, 15%, 20%, 30%
            pct = (k / d_inter) * 100
            
            # Select top-k neurons according to router scores
            topk_indices = torch.topk(scores_pred, k=k, dim=-1).indices # (N_test, k)

            # Ground truth top-k for overlap measurement
            gt_topk_indices = torch.topk(act_test.abs(), k=k, dim=-1).indices
            
            # Compute intersection/overlap
            overlaps = []
            for i in range(N_test):
                s_pred = set(topk_indices[i].tolist())
                s_gt = set(gt_topk_indices[i].tolist())
                overlaps.append(len(s_pred.intersection(s_gt)) / k)
            avg_overlap = sum(overlaps) / len(overlaps) * 100

            # Compute sparse output using predicted active neurons
            sparse_act = torch.zeros_like(act_test)
            sparse_act.scatter_(dim=-1, index=topk_indices, src=act_test.gather(dim=-1, index=topk_indices))
            y_sparse = F.linear(sparse_act.to(torch.bfloat16), W_down).float()

            cos_sim = F.cosine_similarity(y_dense_test, y_sparse, dim=-1).mean().item()
            rel_l2 = (torch.norm(y_dense_test - y_sparse, dim=-1) / torch.norm(y_dense_test, dim=-1)).mean().item()

            print(f"    • Top-{k} ({pct:4.1f}% neuroni): Overlap={avg_overlap:5.1f}% | CosSim={cos_sim:.5f} | Errore L2={rel_l2:.4f}")
        print("-" * 75)


if __name__ == "__main__":
    test_low_rank_router()
