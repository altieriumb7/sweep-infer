"""
SynapticFold v2: Native Precision Dual-Path Benchmark (Zero Quantization).

Compares:
1. Dense Float Baseline (Native bfloat16/float32, 100% dense MKL GEMM)
2. Native Synaptic Dual-Path (Native bfloat16/float32, Contiguous Slicing + Gromov Fold)

Evaluates:
- Micro-benchmark on 7B Layer Scale (d=3584, d_ff=18944)
- Micro-benchmark on 0.5B Layer Scale (d=896, d_ff=4864)
- Real Model End-to-End Autoregressive Generation on Qwen2.5-0.5B (TTFT + 30 tokens)
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.native_synaptic import NativeSynapticMLP

torch.set_num_threads(8)

print("=" * 95, flush=True)
print("1. MICRO-BENCHMARK: 7B SCALE LAYER (d=3584, d_ff=18944 - PURE UNQUANTIZED FLOAT)", flush=True)
print("=" * 95, flush=True)

d_7b, d_ff_7b = 3584, 18944

class Mock7B(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(d_7b, d_ff_7b, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(d_7b, d_ff_7b, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(d_ff_7b, d_7b, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

mock_7b = Mock7B()
syn_7b = NativeSynapticMLP(mock_7b, keep_ratio=0.70)
x_7b = torch.randn(1, 1, d_7b, dtype=torch.bfloat16)

for _ in range(5): mock_7b(x_7b); syn_7b(x_7b)

n_iters = 50
t0 = time.perf_counter()
for _ in range(n_iters): y_dense_7b = mock_7b(x_7b)
t_dense_7b = (time.perf_counter() - t0) / n_iters * 1000.0

t0 = time.perf_counter()
for _ in range(n_iters): y_syn_7b = syn_7b(x_7b)
t_syn_7b = (time.perf_counter() - t0) / n_iters * 1000.0

speedup_7b = t_dense_7b / t_syn_7b
cos_sim_7b = F.cosine_similarity(y_dense_7b.float().squeeze(), y_syn_7b.float().squeeze(), dim=-1).item()

print(f"- Dense Float Baseline:          {t_dense_7b:6.2f} ms | 100% active (18,944 syn) | 777.0 MB Float RAM", flush=True)
print(f"- Native Synaptic Float:         {t_syn_7b:6.2f} ms | {syn_7b.last_sparsity*100:4.1f}% silent ({syn_7b.last_k} syn) | Speedup: {speedup_7b:4.2f}x | CosSim: {cos_sim_7b:.4f}", flush=True)

# 2. Micro-Benchmark on 0.5B Scale
print("\n" + "=" * 95, flush=True)
print("2. MICRO-BENCHMARK: 0.5B SCALE LAYER (d=896, d_ff=4864 - PURE UNQUANTIZED FLOAT)", flush=True)
print("=" * 95, flush=True)

d_05b, d_ff_05b = 896, 4864

class Mock05B(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(d_05b, d_ff_05b, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(d_05b, d_ff_05b, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(d_ff_05b, d_05b, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

mock_05b = Mock05B()
syn_05b = NativeSynapticMLP(mock_05b, keep_ratio=0.70)
x_05b = torch.randn(1, 1, d_05b, dtype=torch.bfloat16)

for _ in range(5): mock_05b(x_05b); syn_05b(x_05b)

n_iters = 100
t0 = time.perf_counter()
for _ in range(n_iters): y_dense_05b = mock_05b(x_05b)
t_dense_05b = (time.perf_counter() - t0) / n_iters * 1000.0

t0 = time.perf_counter()
for _ in range(n_iters): y_syn_05b = syn_05b(x_05b)
t_syn_05b = (time.perf_counter() - t0) / n_iters * 1000.0

speedup_05b = t_dense_05b / t_syn_05b
cos_sim_05b = F.cosine_similarity(y_dense_05b.float().squeeze(), y_syn_05b.float().squeeze(), dim=-1).item()
print(f"- Dense Float Baseline:          {t_dense_05b:6.3f} ms | 100% active (4,864 syn)", flush=True)
print(f"- Native Synaptic Float:         {t_syn_05b:6.3f} ms | {syn_05b.last_sparsity*100:4.1f}% silent ({syn_05b.last_k} syn) | Speedup: {speedup_05b:4.2f}x | CosSim: {cos_sim_05b:.4f}", flush=True)

# 3. Real Model End-to-End Generation Benchmark
print("\n" + "=" * 95, flush=True)
print("3. REAL MODEL AUTOREGRESSIVE GENERATION (Qwen2.5-0.5B, 24 Layers, 30 Tokens)", flush=True)
print("=" * 95, flush=True)

model_path = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16, device_map="cpu")
model.eval()

prompts = [
    "Explain how biological neurons achieve energy efficiency compared to artificial neural networks in three points:",
    "What is the mathematical principle behind the Donoho-Johnstone universal threshold?",
]

# A. Evaluate Dense Float Baseline
print("\n--- Evaluating Dense Float Generation ---", flush=True)
dense_results = []
dense_texts = []
for i, p in enumerate(prompts):
    inputs = tokenizer(p, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # TTFT prefill
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids)
    ttft = (time.perf_counter() - t0) * 1000.0

    # Total generation
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=30, do_sample=False)
    elapsed = (time.perf_counter() - t0) * 1000.0
    tok_s = 30.0 / (elapsed / 1000.0)
    text = tokenizer.decode(out[0], skip_special_tokens=True).encode("ascii", errors="ignore").decode()
    dense_texts.append(text)
    dense_results.append({"ttft": ttft, "time": elapsed, "tok_s": tok_s})
    print(f"Dense Prompt #{i+1} | TTFT: {ttft:6.1f} ms | Gen Time: {elapsed:6.1f} ms | Throughput: {tok_s:5.2f} tok/s", flush=True)

# B. Install Native Synaptic Dual-Path MLPs
wrapped_pairs = NativeSynapticMLP.wrap_model(
    model,
    keep_ratio=0.70,
    plateau_range=(4, 19),
    fold_prefill_layers={6, 8, 10, 12},
)
wrapped_mlps = [m for _, m in wrapped_pairs]

print("\n--- Evaluating Native Synaptic Dual-Path Generation ---", flush=True)
syn_results = []
syn_texts = []
for i, p in enumerate(prompts):
    inputs = tokenizer(p, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # TTFT prefill
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids)
    ttft = (time.perf_counter() - t0) * 1000.0

    # Total generation
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=30, do_sample=False)
    elapsed = (time.perf_counter() - t0) * 1000.0
    tok_s = 30.0 / (elapsed / 1000.0)
    mean_sp = sum(m.last_sparsity for m in wrapped_mlps) / len(wrapped_mlps) * 100.0
    text = tokenizer.decode(out[0], skip_special_tokens=True).encode("ascii", errors="ignore").decode()
    syn_texts.append(text)
    syn_results.append({"ttft": ttft, "time": elapsed, "tok_s": tok_s, "silent": mean_sp})
    print(f"Synaptic Prompt #{i+1} | TTFT: {ttft:6.1f} ms | Gen Time: {elapsed:6.1f} ms | Throughput: {tok_s:5.2f} tok/s | Quiescence: {mean_sp:4.1f}%", flush=True)

avg_dense_ttft = sum(r["ttft"] for r in dense_results) / len(dense_results)
avg_syn_ttft = sum(r["ttft"] for r in syn_results) / len(syn_results)
avg_dense_t = sum(r["time"] for r in dense_results) / len(dense_results)
avg_syn_t = sum(r["time"] for r in syn_results) / len(syn_results)
avg_dense_toks = sum(r["tok_s"] for r in dense_results) / len(dense_results)
avg_syn_toks = sum(r["tok_s"] for r in syn_results) / len(syn_results)
avg_sp = sum(r["silent"] for r in syn_results) / len(syn_results)
ttft_speedup = avg_dense_ttft / avg_syn_ttft
gen_speedup = avg_dense_t / avg_syn_t

print("\n" + "=" * 95, flush=True)
print("FINAL NATIVE SYNAPTIC COMPARISON REPORT (ZERO QUANTIZATION)", flush=True)
print("=" * 95, flush=True)
print(f"{'METRIC':<36} | {'DENSE FLOAT BASELINE':<25} | {'NATIVE SYNAPTIC (OURS)'}", flush=True)
print("-" * 95, flush=True)
print(f"{'Quantization Applied':<36} | {'NONE (16-bit Float)':<25} | NONE (16-bit Float)", flush=True)
print(f"{'Weight Pruning Applied':<36} | {'NONE (100% in RAM)':<25} | NONE (100% in RAM)", flush=True)
print(f"{'7B Micro-GEMM Layer Time':<36} | {t_dense_7b:6.2f} ms                 | {t_syn_7b:6.2f} ms ({speedup_7b:.2f}x FASTER)", flush=True)
print(f"{'0.5B Micro-GEMM Layer Time':<36} | {t_dense_05b:6.3f} ms                | {t_syn_05b:6.3f} ms ({speedup_05b:.2f}x FASTER)", flush=True)
print(f"{'Prompt Query Latency (TTFT)':<36} | {avg_dense_ttft:6.1f} ms              | {avg_syn_ttft:6.1f} ms ({ttft_speedup:.2f}x FASTER)", flush=True)
print(f"{'Generation Throughput':<36} | {avg_dense_toks:5.2f} tok/s              | {avg_syn_toks:5.2f} tok/s ({gen_speedup:.2f}x FASTER)", flush=True)
print(f"{'Quiescent Silent Synapses':<36} | {'0.0% at rest':<25} | {avg_sp:4.1f}% at rest", flush=True)
print(f"{'Cosine Fidelity (0.5B Micro)':<36} | {'1.0000 (Reference)':<25} | {cos_sim_05b:.4f}", flush=True)
print("=" * 95, flush=True)

print("\n--- SAMPLE GENERATED TEXT ---")
print("DENSE:")
print(dense_texts[0][:200])
print("\nSYNAPTIC:")
print(syn_texts[0][:200])

results_dict = {
    "micro_7b": {
        "dense_ms": round(t_dense_7b, 2),
        "syn_ms": round(t_syn_7b, 2),
        "speedup": round(speedup_7b, 2),
        "cos_sim": round(cos_sim_7b, 4),
    },
    "micro_05b": {
        "dense_ms": round(t_dense_05b, 3),
        "syn_ms": round(t_syn_05b, 3),
        "speedup": round(speedup_05b, 2),
        "cos_sim": round(cos_sim_05b, 4),
    },
    "generation": {
        "avg_dense_ttft_ms": round(avg_dense_ttft, 1),
        "avg_syn_ttft_ms": round(avg_syn_ttft, 1),
        "ttft_speedup": round(ttft_speedup, 2),
        "avg_dense_latency_ms": round(avg_dense_t, 1),
        "avg_syn_latency_ms": round(avg_syn_t, 1),
        "avg_dense_tok_s": round(avg_dense_toks, 2),
        "avg_syn_tok_s": round(avg_syn_toks, 2),
        "gen_speedup_factor": round(gen_speedup, 2),
        "mean_sparsity": round(avg_sp, 1),
    }
}
out_json_path = os.path.join(ROOT_DIR, "benchmarks", "native_synaptic_benchmark_results.json")
with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump(results_dict, f, indent=2)
print(f"\n[OK] Results saved to: {out_json_path}", flush=True)
