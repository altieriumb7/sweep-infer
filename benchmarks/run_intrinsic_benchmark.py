"""
SynapticFold v2: Intrinsic Parameter-Free Mobilization Benchmark.

Evaluates and compares on real LLM layers (Qwen2.5-0.5B):
1. Dense Baseline (100% active synapses)
2. Elastic Mobilization (P_mass = 0.85, heuristic threshold)
3. Intrinsic Kneedle Knee-Point (Parameter-Free maximal curvature)
4. Intrinsic Donoho-Johnstone Universal Noise Floor (Parameter-Free statistical filtering)
"""

import os
import sys
import time
import json
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.intrinsic import IntrinsicModelWrapper

model_path = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16, device_map="cpu")
model.eval()

# BoolQ samples
boolq_ds = load_dataset("google/boolq", split="validation")
random.seed(42)
indices = random.sample(range(len(boolq_ds)), 5)
samples = [boolq_ds[i] for i in indices]

yes_candidates = ["Yes", " Yes", "yes", " True", "True"]
no_candidates = ["No", " No", "no", " False", "False"]
yes_ids = list(set([tokenizer.encode(s, add_special_tokens=False)[0] for s in yes_candidates]))
no_ids = list(set([tokenizer.encode(s, add_special_tokens=False)[0] for s in no_candidates]))

# Calibration offset from empirical null prior
prior_offset = 2.50

def evaluate_regime(name, hooks=None):
    correct = 0
    times = []
    sparsities = []
    
    for s in samples:
        target = bool(s["answer"])
        prompt = f"Passage: {s['passage'].strip()}\nQuestion: {s['question'].strip()}?\nAnswer with Yes or No:\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(inputs["input_ids"]).logits[0, -1, :]
        elapsed = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed)
        
        yes_score = torch.logsumexp(logits[yes_ids].float(), dim=0).item() + prior_offset
        no_score = torch.logsumexp(logits[no_ids].float(), dim=0).item()
        pred = (yes_score > no_score)
        if pred == target:
            correct += 1
            
        if hooks:
            mean_sp = float(sum(h.last_sparsity for h in hooks) / len(hooks)) * 100.0
            sparsities.append(mean_sp)
        else:
            sparsities.append(0.0)
            
    avg_t = sum(times) / len(times)
    avg_sp = sum(sparsities) / len(sparsities)
    acc = correct / len(samples) * 100.0
    print(f"| {name:<30} | {acc:5.1f}% ({correct}/{len(samples)}) | {avg_sp:5.1f}% silent | {avg_t:6.1f} ms |")
    return {"name": name, "acc": acc, "silent_pct": avg_sp, "latency_ms": avg_t}

print("\n=== INTRINSIC PARAMETER-FREE BENCHMARK ON REAL MODEL (Qwen2.5-0.5B, 24 Layers) ===")
print(f"| {'Regime':<30} | {'Accuracy':<14} | {'Quiescence':<14} | {'Latency':<9} |")
print("|" + "-" * 32 + "|" + "-" * 16 + "|" + "-" * 16 + "|" + "-" * 11 + "|")

# 1. Dense Baseline
evaluate_regime("Dense Baseline (100%)", hooks=None)

# 2. Elastic P_mass = 0.85 (old heuristic)
hooks_elastic = IntrinsicModelWrapper.wrap(model, mode="elastic", p_mass=0.85)
evaluate_regime("Elastic (P_mass = 0.85)", hooks=hooks_elastic)
IntrinsicModelWrapper.unwrap(hooks_elastic)

# 3. Intrinsic Kneedle Knee-Point (Parameter-Free)
hooks_knee = IntrinsicModelWrapper.wrap(model, mode="knee")
evaluate_regime("Intrinsic Knee-Point (No Param)", hooks=hooks_knee)
IntrinsicModelWrapper.unwrap(hooks_knee)

# 4. Intrinsic Donoho-Johnstone Noise Floor (Parameter-Free)
hooks_dj = IntrinsicModelWrapper.wrap(model, mode="noise")
evaluate_regime("Intrinsic Noise Floor (No Param)", hooks=hooks_dj)
IntrinsicModelWrapper.unwrap(hooks_dj)

print("=" * 77)
