"""
SynapticFold v2: Head-to-Head Benchmark on 10 Random Samples.
Compares:
1. Dense Baseline (100% active)
2. Intrinsic Knee-Point (Parameter-Free Maximal Curvature)
3. Intrinsic Noise Floor (Parameter-Free Donoho-Johnstone Universal Threshold)
"""

import os
import sys
import time
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.intrinsic import IntrinsicModelWrapper, IntrinsicSynapticMLPHook

# Setup Model
model_path = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
print("[1/3] Loading model into RAM...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16, device_map="cpu")
model.eval()

# Select 10 fresh random samples
print("[2/3] Sampling 10 fresh random questions from BoolQ...", flush=True)
boolq_ds = load_dataset("google/boolq", split="validation")
random.seed(999)  # New seed distinct from previous runs
indices = random.sample(range(len(boolq_ds)), 10)
samples = [boolq_ds[i] for i in indices]

yes_candidates = ["Yes", " Yes", "yes", " True", "True"]
no_candidates = ["No", " No", "no", " False", "False"]
yes_ids = list(set([tokenizer.encode(s, add_special_tokens=False)[0] for s in yes_candidates]))
no_ids = list(set([tokenizer.encode(s, add_special_tokens=False)[0] for s in no_candidates]))

prior_offset = 2.50  # Calibrated null vocabulary prior

# Evaluation Loop
dense_results = []
knee_results = []
noise_results = []

print("\n[3/3] Running Head-to-Head Comparison on 10 Samples...", flush=True)
print("=" * 95, flush=True)
print(f"{'Idx':<4} | {'Target':<6} | {'Dense Base':<15} | {'Intrinsic Knee-Point':<28} | {'Intrinsic Noise Floor':<28}", flush=True)
print("-" * 95, flush=True)

for i, s in enumerate(samples):
    target = bool(s["answer"])
    prompt = f"Passage: {s['passage'].strip()}\nQuestion: {s['question'].strip()}?\nAnswer with Yes or No:\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # 1. Dense Baseline
    t0 = time.perf_counter()
    with torch.no_grad():
        logits_d = model(input_ids).logits[0, -1, :]
    t_d = (time.perf_counter() - t0) * 1000.0
    yes_d = torch.logsumexp(logits_d[yes_ids].float(), dim=0).item() + prior_offset
    no_d = torch.logsumexp(logits_d[no_ids].float(), dim=0).item()
    pred_d = (yes_d > no_d)
    corr_d = (pred_d == target)
    dense_results.append({"pred": pred_d, "corr": corr_d, "time": t_d, "margin": yes_d - no_d})

    # 2. Intrinsic Knee-Point
    hooks_k = IntrinsicModelWrapper.wrap(model, mode="knee")
    t0 = time.perf_counter()
    with torch.no_grad():
        logits_k = model(input_ids).logits[0, -1, :]
    t_k = (time.perf_counter() - t0) * 1000.0
    yes_k = torch.logsumexp(logits_k[yes_ids].float(), dim=0).item() + prior_offset
    no_k = torch.logsumexp(logits_k[no_ids].float(), dim=0).item()
    pred_k = (yes_k > no_k)
    corr_k = (pred_k == target)
    sp_k = IntrinsicModelWrapper.get_mean_sparsity(hooks_k)
    IntrinsicModelWrapper.unwrap(hooks_k)
    knee_results.append({"pred": pred_k, "corr": corr_k, "time": t_k, "silent": sp_k, "margin": yes_k - no_k})

    # 3. Intrinsic Noise Floor
    hooks_n = IntrinsicModelWrapper.wrap(model, mode="noise")
    t0 = time.perf_counter()
    with torch.no_grad():
        logits_n = model(input_ids).logits[0, -1, :]
    t_n = (time.perf_counter() - t0) * 1000.0
    yes_n = torch.logsumexp(logits_n[yes_ids].float(), dim=0).item() + prior_offset
    no_n = torch.logsumexp(logits_n[no_ids].float(), dim=0).item()
    pred_n = (yes_n > no_n)
    corr_n = (pred_n == target)
    sp_n = IntrinsicModelWrapper.get_mean_sparsity(hooks_n)
    IntrinsicModelWrapper.unwrap(hooks_n)
    noise_results.append({"pred": pred_n, "corr": corr_n, "time": t_n, "silent": sp_n, "margin": yes_n - no_n})

    status_d = "OK " if corr_d else "ERR"
    status_k = "OK " if corr_k else "ERR"
    status_n = "OK " if corr_n else "ERR"

    str_d = f"{str(pred_d):<5} [{status_d}]"
    str_k = f"{str(pred_k):<5} [{status_k}] ({sp_k:4.1f}% sil)"
    str_n = f"{str(pred_n):<5} [{status_n}] ({sp_n:4.1f}% sil)"

    print(f"#{i+1:02d} | {str(target):<6} | {str_d:<15} | {str_k:<28} | {str_n:<28}", flush=True)

print("=" * 95, flush=True)

# Final Summary Stats
acc_d = sum(r["corr"] for r in dense_results) / 10.0 * 100.0
acc_k = sum(r["corr"] for r in knee_results) / 10.0 * 100.0
acc_n = sum(r["corr"] for r in noise_results) / 10.0 * 100.0

parity_k = sum(k["pred"] == d["pred"] for k, d in zip(knee_results, dense_results)) / 10.0 * 100.0
parity_n = sum(n["pred"] == d["pred"] for n, d in zip(noise_results, dense_results)) / 10.0 * 100.0

sil_k = sum(r["silent"] for r in knee_results) / 10.0
sil_n = sum(r["silent"] for r in noise_results) / 10.0

time_d = sum(r["time"] for r in dense_results) / 10.0
time_k = sum(r["time"] for r in knee_results) / 10.0
time_n = sum(r["time"] for r in noise_results) / 10.0

print("\n📊 RIEPILOGO FINALE SU 10 CAMPIONI RANDOM:", flush=True)
print(f"- Dense Baseline (100%):         Accuratezza: {acc_d:5.1f}% ({int(acc_d/10)}/10) | Quiescenza:   0.0% silenti | Latenza media: {time_d:6.1f} ms", flush=True)
print(f"- Intrinsic Knee-Point (Ours):   Accuratezza: {acc_k:5.1f}% ({int(acc_k/10)}/10) | Quiescenza: {sil_k:5.1f}% silenti | Parità con Dense: {parity_k:5.1f}%", flush=True)
print(f"- Intrinsic Noise Floor (Ours):  Accuratezza: {acc_n:5.1f}% ({int(acc_n/10)}/10) | Quiescenza: {sil_n:5.1f}% silenti | Parità con Dense: {parity_n:5.1f}%", flush=True)
print("=" * 95, flush=True)
