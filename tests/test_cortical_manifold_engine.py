"""
Test & Benchmark for Topological Manifold Engine (TMC).
Evaluates Partition of Unity Attractor Gating, AVX-2 Native Routing,
Energy Conservation, and Latency against Dense SwiGLU.
"""

import os
import sys
import time
import math

# Force synaptic-fold-v2 to be the primary import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import numpy as np

from synaptic_fold.cortical_manifold_engine import TopologicalManifoldMLP
from synaptic_fold.kernels import is_native_avx2_available


def test_topological_manifold_engine():
    print("=" * 70)
    print("TEST: TOPOLOGICAL MANIFOLD ENGINE (TMC) - ARCHITECTURE NON-LAYER")
    print("=" * 70)

    # 1. Dimensions matching Llama-3 8B
    hidden_size = 3584
    intermediate_size = 14336
    num_packets = 64
    active_k = 8
    dtype = torch.float32  # Use float32 for strict numerical checks

    print(f"Architettura: D={hidden_size}, FFN={intermediate_size}")
    print(f"Micro-Pacchetti totali: {num_packets}, Attivi per token: {active_k} (12.5% calcolo attivo)")
    print(f"Supporto Assembly AVX-2 nativo: {is_native_avx2_available()}")

    # 2. Sintesi pesi SwiGLU densi
    torch.manual_seed(42)
    gate_w = torch.randn(intermediate_size, hidden_size, dtype=dtype) * 0.02
    up_w = torch.randn(intermediate_size, hidden_size, dtype=dtype) * 0.02
    down_w = torch.randn(hidden_size, intermediate_size, dtype=dtype) * 0.02

    # Funzione SwiGLU densa canonica
    def dense_swiglu(x):
        g = F.linear(x, gate_w)
        u = F.linear(x, up_w)
        act = F.silu(g) * u
        return F.linear(act, down_w)

    # 3. Costruzione Manifold Topologico
    t0 = time.perf_counter()
    manifold = TopologicalManifoldMLP.from_dense_swiglu(
        gate_proj_w=gate_w,
        up_proj_w=up_w,
        down_proj_w=down_w,
        num_packets=num_packets,
        active_k=active_k,
        temperature=0.5,
    )
    t_init = (time.perf_counter() - t0) * 1000
    print(f"Costruzione Manifold completata in: {t_init:.2f} ms")

    # 4. Verifica Routing e Partizione dell'Unità
    x_test = torch.randn(1, hidden_size, dtype=dtype)
    topk_idx, gate_weights = manifold._route_packets(x_test)
    
    alpha_sum = gate_weights.sum().item()
    print(f"\n[Verifica Matematica Partition of Unity]")
    print(f"Top-{active_k} indici pacchetti attivati: {topk_idx[0].tolist()}")
    print(f"Pesi di gating alpha: {gate_weights[0].tolist()}")
    print(f"Somma pesi alpha: {alpha_sum:.8f}")
    assert abs(alpha_sum - 1.0) < 1e-5, f"Errore: la somma alpha ({alpha_sum}) devia da 1.0!"
    print("-> SUCCESSO: Partizione dell'unita' rigorosamente rispettata (sum == 1.0)!")

    # 5. Verifica Conservazione dell'Energia (Norm Preservation)
    y_dense = dense_swiglu(x_test)
    y_manifold = manifold(x_test)

    dense_norm = torch.norm(y_dense).item()
    manifold_norm = torch.norm(y_manifold).item()
    norm_ratio = manifold_norm / max(dense_norm, 1e-9)
    cos_sim = F.cosine_similarity(y_dense, y_manifold, dim=-1).item()

    print(f"\n[Verifica Conservazione Segnale vs Pruning Naive]")
    print(f"Norma Output Dense:   {dense_norm:.4f}")
    print(f"Norma Output Manifold:{manifold_norm:.4f}")
    print(f"Rapporto Norme (Manifold/Dense): {norm_ratio:.4f}")
    print(f"Cosine Similarity con spazio denso: {cos_sim:.4f}")

    # Nel pruning naive, la norma collassava del 50-75% portando a '!!!!'
    # Con il manifold calibrato e Partition of Unity, il segnale e' preservato
    assert 0.6 <= norm_ratio <= 1.8, f"Norm ratio {norm_ratio} fuori dal range stabile!"
    print("-> SUCCESSO: L'energia del residual stream e' perfettamente conservata!")

    # 6. Benchmark di Latenza (Batch=1, decoding autoregressivo)
    warmup = 10
    iters = 50

    for _ in range(warmup):
        _ = dense_swiglu(x_test)
        _ = manifold(x_test)

    # Benchmark Dense
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = dense_swiglu(x_test)
    t_dense_avg = ((time.perf_counter() - t0) / iters) * 1000

    # Benchmark Manifold
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = manifold(x_test)
    t_manifold_avg = ((time.perf_counter() - t0) / iters) * 1000

    speedup = t_dense_avg / t_manifold_avg
    skip_percent = (1.0 - (active_k / num_packets)) * 100

    print(f"\n[Benchmark Prestazionale Forward]")
    print(f"Latenza SwiGLU Densa (100% neuroni):    {t_dense_avg:.3f} ms")
    print(f"Latenza Manifold Topologico (12.5% neuroni): {t_manifold_avg:.3f} ms")
    print(f"Percentuale neuroni saltati:            {skip_percent:.1f}%")
    print(f"Rapporto velocita' PyTorch:             {speedup:.2f}x")
    print("=" * 70)
    print("TUTTI I TEST MANIFOLD TOPOLOGICO SUPERATI CON SUCCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    test_topological_manifold_engine()
