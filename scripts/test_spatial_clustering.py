"""
Prototype & Verification of Spherical Clustering for Spatial Synaptic Routing.
Clusters 18,944 SwiGLU neurons into C directional fascicles and tests router retrieval.
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


def spherical_kmeans(vectors: torch.Tensor, num_clusters: int = 64, num_iters: int = 25, seed: int = 42):
    """
    Spherical K-Means clustering using cosine distance on CPU.
    vectors: (N_neurons, d_model) -> (18944, 3584)
    Returns:
        cluster_ids: (N_neurons,) long
        centroids: (num_clusters, d_model) float32, unit norm
    """
    torch.manual_seed(seed)
    N, d = vectors.shape
    v_norm = F.normalize(vectors.float(), p=2, dim=-1)
    
    # Initialize centroids with k-means++ or random normalized vectors
    perm = torch.randperm(N)[:num_clusters]
    centroids = v_norm[perm].clone() # (C, d)

    for it in range(num_iters):
        # Cosine similarity between each neuron and all centroids
        # (N, C)
        sim = torch.matmul(v_norm, centroids.t())
        cluster_ids = torch.argmax(sim, dim=-1) # (N,)
        
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for c in range(num_clusters):
            mask = (cluster_ids == c)
            if mask.sum() > 0:
                new_centroids[c] = v_norm[mask].mean(dim=0)
            else:
                # Re-seed empty cluster with random vector
                new_centroids[c] = v_norm[torch.randint(0, N, (1,))].squeeze(0)
                
        centroids = F.normalize(new_centroids, p=2, dim=-1)

    return cluster_ids, centroids


def test_spatial_router():
    print("=" * 75)
    print("🔬 TEST SPHERICAL CLUSTERING & SPATIAL SYNAPTIC ROUTER")
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

    N_neurons, d = W_gate.shape
    print(f"[Pesi Caricati] Strato 12: {N_neurons} neuroni x {d} dimensioni.")

    num_clusters = 64
    print(f"\n[Fase A] Clustering Sferico su {N_neurons} neuroni in {num_clusters} fascicoli sinaptici...")
    t0 = time.perf_counter()
    cluster_ids, centroids = spherical_kmeans(W_gate, num_clusters=num_clusters, num_iters=20)
    dt_clust = time.perf_counter() - t0
    print(f"  Clustering completato in {dt_clust:.2f}s!")

    # Check cluster distribution
    counts = torch.bincount(cluster_ids, minlength=num_clusters)
    print(f"  Distribuzione neuroni per cluster: Min={counts.min().item()}, Max={counts.max().item()}, Media={counts.float().mean().item():.1f}")

    # Load 700 tokens of input
    data = torch.load(TRAJ_PATH, map_location="cpu", weights_only=True)
    X = data["X"][:200] # First 200 tokens
    N_tokens = X.shape[0]

    # Pre-normalize X
    eps = 1e-6
    variance = X.float().pow(2).mean(-1, keepdim=True)
    X_norm = (X.float() * torch.rsqrt(variance + eps) * norm_w.float()).to(torch.bfloat16)

    # Dense ground truth
    print(f"\n[Fase B] Calcolo Dense Ground Truth su {N_tokens} token...")
    with torch.no_grad():
        gate_dense = F.linear(X_norm, W_gate)
        up_dense = F.linear(X_norm, W_up)
        act_dense = F.silu(gate_dense.float()) * up_dense.float()
        y_dense = F.linear(act_dense.to(torch.bfloat16), W_down).float()

    # Precompute cluster neuron index masks
    cluster_masks = [torch.where(cluster_ids == c)[0] for c in range(num_clusters)]

    # Test Router with different Top-C selections
    # If num_clusters = 64:
    # top_c = 8 -> 12.5% of neurons
    # top_c = 12 -> 18.7% of neurons
    # top_c = 16 -> 25.0% of neurons
    # top_c = 24 -> 37.5% of neurons
    print("\n" + "=" * 75)
    print("🎯 PERFORMANCE DEL ROUTER TOPOLOGICO (Selezione Dinamica dei Fascicoli)")
    print("=" * 75)

    for top_c in [8, 12, 16, 20, 24, 32]:
        # Router scores: input x vs centroids
        # X_norm: (N_tokens, d), centroids: (num_clusters, d)
        X_dir = F.normalize(X_norm.float(), p=2, dim=-1)
        router_scores = torch.matmul(X_dir, centroids.t()) # (N_tokens, C)
        
        # Pick top_c clusters per token
        top_clusters = torch.topk(router_scores, k=top_c, dim=-1).indices # (N_tokens, top_c)

        # For each token, gather selected neurons
        y_router_list = []
        active_counts = []
        
        for i in range(N_tokens):
            active_clusters = top_clusters[i].tolist()
            # Unique active neurons for this token
            active_neurons = torch.cat([cluster_masks[c] for c in active_clusters])
            active_counts.append(len(active_neurons))

            # Sparse computation: only for active neurons
            w_g_sparse = W_gate[active_neurons] # (K, d)
            w_u_sparse = W_up[active_neurons]   # (K, d)
            w_d_sparse = W_down[:, active_neurons] # (d, K)

            x_i = X_norm[i:i+1] # (1, d)
            g_i = F.linear(x_i, w_g_sparse)
            u_i = F.linear(x_i, w_u_sparse)
            a_i = F.silu(g_i.float()) * u_i.float()
            y_i = F.linear(a_i.to(torch.bfloat16), w_d_sparse).float()
            y_router_list.append(y_i)

        y_router = torch.cat(y_router_list, dim=0)
        avg_active = sum(active_counts) / len(active_counts)
        pct_active = (avg_active / N_neurons) * 100

        cos_sim = F.cosine_similarity(y_dense, y_router, dim=-1).mean().item()
        rel_l2 = (torch.norm(y_dense - y_router, dim=-1) / torch.norm(y_dense, dim=-1)).mean().item()

        print(f"  • Top-{top_c:2d} cluster (~{pct_active:4.1f}% neuroni attivi, {avg_active:.0f}/{N_neurons}):")
        print(f"      Similarita Coseno Output: {cos_sim:.5f}")
        print(f"      Errore L2 Relativo:      {rel_l2:.4f}")
        print(f"      Risparmio FLOPs Teoria:  {100 - pct_active:.1f}%")
        print("-" * 50)


if __name__ == "__main__":
    test_spatial_router()
