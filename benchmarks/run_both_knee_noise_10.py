"""
SynapticFold v2: Head-to-Head Comparison on 10 Random Samples (Seed 999).
Compares:
1. Dense Baseline (100% active)
2. Intrinsic Knee-Point (Parameter-Free Maximal Curvature)
3. Intrinsic Noise Floor (Parameter-Free Donoho-Johnstone Universal Threshold)
4. BOTH Combined (Dual Intersection: Knee & Noise)
5. BOTH Combined (Dual Union: Knee | Noise)
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

from synaptic_fold.intrinsic import IntrinsicModelWrapper

# 1. Load Model into RAM
model_path = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
print("[1/3] Loading model into RAM...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16, device_map="cpu")
model.eval()

# 2. Extract the EXACT same 10 samples (seed 999)
print("[2/3] Extracting the exact same 10 BoolQ samples (Seed 999)...", flush=True)
boolq_ds = load_dataset("google/boolq", split="validation")
random.seed(999)
indices = random.sample(range(len(boolq_ds)), 10)
samples = [boolq_ds[i] for i in indices]

yes_candidates = ["Yes", " Yes", "yes", " True", "True"]
no_candidates = ["No", " No", "no", " False", "False"]
yes_ids = list(set([tokenizer.encode(s, add_special_tokens=False)[0] for s in yes_candidates]))
no_ids = list(set([tokenizer.encode(s, add_special_tokens=False)[0] for s in no_candidates]))

prior_offset = 2.50

# Regimes to evaluate
regimes = [
    ("Dense Baseline", None),
    ("Knee-Point", "knee"),
    ("Noise Floor", "noise"),
    ("BOTH (Intersection)", "both_intersect"),
    ("BOTH (Union)", "both_union"),
]

results = {name: [] for name, _ in regimes}

print("\n[3/3] Evaluating All Regimes Across 10 Samples...", flush=True)
print("=" * 105, flush=True)
header = f"{'Idx':<4} | {'Target':<6} | {'Dense Base':<12} | {'Knee-Point':<16} | {'Noise Floor':<16} | {'BOTH (Intersect)':<18} | {'BOTH (Union)':<18}"
print(header, flush=True)
print("-" * 105, flush=True)

for i, s in enumerate(samples):
    target = bool(s["answer"])
    prompt = f"Passage: {s['passage'].strip()}\nQuestion: {s['question'].strip()}?\nAnswer with Yes or No:\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    row_strs = []

    for name, mode in regimes:
        if mode is None:
            hooks = None
        else:
            hooks = IntrinsicModelWrapper.wrap(model, mode=mode)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]
        elapsed = (time.perf_counter() - t0) * 1000.0

        if hooks is not None:
            sp = IntrinsicModelWrapper.get_mean_sparsity(hooks)
            IntrinsicModelWrapper.unwrap(hooks)
        else:
            sp = 0.0

        yes_s = torch.logsumexp(logits[yes_ids].float(), dim=0).item() + prior_offset
        no_s = torch.logsumexp(logits[no_ids].float(), dim=0).item()
        pred = (yes_s > no_s)
        corr = (pred == target)

        results[name].append({"pred": pred, "corr": corr, "time": elapsed, "silent": sp})

        status = "OK " if corr else "ERR"
        if mode is None:
            row_strs.append(f"{str(pred):<5} [{status}]")
        else:
            row_strs.append(f"{str(pred):<5} [{status}] ({sp:4.1f}%)")

    print(f"#{i+1:02d} | {str(target):<6} | {row_strs[0]:<12} | {row_strs[1]:<16} | {row_strs[2]:<16} | {row_strs[3]:<18} | {row_strs[4]:<18}", flush=True)

print("=" * 105, flush=True)

print("\n📊 RIEPILOGO STATISTICO FINALE (10 CAMPIONI BOOLQ):", flush=True)
print("-" * 105, flush=True)
print(f"{'Regime':<24} | {'Accuratezza':<16} | {'Sinapsi a Riposo (Silenti)':<28} | {'Parità con Dense':<18} | {'Latenza Media'}", flush=True)
print("-" * 105, flush=True)

dense_preds = [r["pred"] for r in results["Dense Baseline"]]

for name, _ in regimes:
    data = results[name]
    correct_count = sum(r["corr"] for r in data)
    acc = correct_count / 10.0 * 100.0
    sil = sum(r["silent"] for r in data) / 10.0
    latency = sum(r["time"] for r in data) / 10.0
    parity = sum(r["pred"] == d for r, d in zip(data, dense_preds)) / 10.0 * 100.0
    print(f"{name:<24} | {acc:5.1f}% ({correct_count:02d}/10)      | {sil:5.1f}% silenti ({4864*(1-sil/100):.0f}/4864 attivi)  | {parity:5.1f}%            | {latency:6.1f} ms", flush=True)

print("=" * 105, flush=True)
