import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

def benchmark_entropy_representations():
    local_snapshot = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
    index_path = os.path.join(local_snapshot, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    lm_head_weight_path = os.path.join(local_snapshot, weight_map["lm_head.weight"])
    with safe_open(lm_head_weight_path, framework="pt", device="cpu") as f:
        lm_head_w = f.get_tensor("lm_head.weight").to(dtype=torch.float32)

    vocab_size, d_model = lm_head_w.shape
    print("=" * 80)
    print(" NOVEL REPRESENTATION: LOW-RANK SKETCH FOR SHANNON PREDICTIVE ENTROPY")
    print(f" Full Vocabulary Size: {vocab_size} | Hidden Dim: {d_model}")
    print("=" * 80)

    # Generate 10 test states simulating attractor relaxation steps
    torch.manual_seed(42)
    base_state = torch.randn(1, 1, d_model)
    noise = torch.randn(10, 1, 1, d_model) * 0.1
    states = [base_state + noise[i] for i in range(10)]

    # 1. Exact Full Vocabulary Entropy
    # Warmup
    _ = F.log_softmax(F.linear(states[0], lm_head_w), dim=-1)
    
    t0 = time.perf_counter()
    full_entropies = []
    for s in states:
        logits = F.linear(s, lm_head_w)[0, 0]
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        h = -torch.sum(probs * log_probs).item()
        full_entropies.append(h)
    t_full = (time.perf_counter() - t0) / 10.0

    print(f"\n[1] Exact Full Vocabulary ({vocab_size} classes): {t_full*1000:.2f} ms / step | Memory: {vocab_size * d_model * 4 / 1e6:.1f} MB")

    # 2. Representation: Johnson-Lindenstrauss Random Projection Sketch (d_sketch = 128)
    d_sketch = 128
    torch.manual_seed(123)
    sketch_matrix = torch.randn(d_model, d_sketch) / (d_model ** 0.5)

    t0 = time.perf_counter()
    sketch_entropies = []
    for s in states:
        proj = torch.matmul(s[0, 0], sketch_matrix) # [128]
        probs = F.softmax(proj, dim=-1)
        log_probs = F.log_softmax(proj, dim=-1)
        h = -torch.sum(probs * log_probs).item()
        sketch_entropies.append(h)
    t_sketch = (time.perf_counter() - t0) / 10.0

    print(f"[2] Low-Rank Sketch (d_sketch = {d_sketch}): {t_sketch*1000:.4f} ms / step | Memory: {d_model * d_sketch * 4 / 1e3:.1f} KB")
    speedup = t_full / max(1e-8, t_sketch)
    print(f"    --> Computational Speedup: {speedup:.1f}x faster!")

    # 3. Dynamic Delta Correlation (relative change across steps)
    full_deltas = [abs(full_entropies[i] - full_entropies[i-1]) for i in range(1, len(full_entropies))]
    sketch_deltas = [abs(sketch_entropies[i] - sketch_entropies[i-1]) for i in range(1, len(sketch_entropies))]

    f_tensor = torch.tensor(full_deltas)
    s_tensor = torch.tensor(sketch_deltas)
    f_norm = f_tensor - f_tensor.mean()
    s_norm = s_tensor - s_tensor.mean()
    corr = (torch.sum(f_norm * s_norm) / (torch.norm(f_norm) * torch.norm(s_norm) + 1e-8)).item()

    print(f"    --> Stabilization Trajectory Tracking Correlation: r = {corr:.3f}")

if __name__ == "__main__":
    benchmark_entropy_representations()
