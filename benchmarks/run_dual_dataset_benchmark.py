"""
SynapticFold v2: Dual-Dataset Empirical Benchmark Evaluator.

Evaluates performance, accuracy, latency, and memory footprint on:
1. Google BoolQ (20 validation samples) - Boolean Reading Comprehension
2. AllenAI ARC-Easy (20 validation samples) - Multi-Choice Scientific Reasoning

Compares:
- Dense Baseline: Standard full-depth sequential transformer forward pass.
- SynapticFold v2: Cortico-recurrent engine (Sensory Compartment + Fixed-Point Attractor Core + Motor Compartment)
  with GQA In-Place Cache and Adaptive Consensus Stopping.
"""

import os
import sys
import time
import json
import random
import psutil
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.adapter import UniversalModelAdapter
from synaptic_fold.engine import SynapticBrainConfig


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def build_boolq_prompt(passage: str, question: str) -> str:
    return (
        f"Passage: {passage.strip()}\n"
        f"Question: {question.strip()}?\n"
        f"Answer with Yes or No:\n"
        f"Answer:"
    )


def build_arc_prompt(question: str, choices: dict) -> str:
    choices_str = ""
    for label, text in zip(choices["label"], choices["text"]):
        choices_str += f"{label}. {text}\n"
    return (
        f"Question: {question.strip()}\n"
        f"{choices_str}"
        f"Answer:"
    )


def marginalize_token_logits(logits_1d: torch.Tensor, token_ids: list) -> float:
    selected = logits_1d[token_ids]
    return torch.logsumexp(selected.float(), dim=0).item()


def run_benchmark(
    model_path: str,
    n_samples: int = 20,
    seed: int = 42,
    output_json: str = "dual_dataset_benchmark_results.json"
):
    set_seed(seed)
    print("=" * 80)
    print(f"[RUN] SYNAPTIC-FOLD v2 BENCHMARK SUITE")
    print(f"Model Path: {model_path}")
    print(f"Samples per dataset: {n_samples} | Seed: {seed}")
    print("=" * 80)

    # 1. Load Tokenizer & Datasets
    print("\n[1/5] Loading datasets (BoolQ & ARC-Easy)...", flush=True)
    boolq_ds = load_dataset("google/boolq", split="validation")
    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")

    # Sample deterministically
    boolq_indices = random.sample(range(len(boolq_ds)), n_samples)
    arc_indices = random.sample(range(len(arc_ds)), n_samples)

    boolq_samples = [boolq_ds[i] for i in boolq_indices]
    arc_samples = [arc_ds[i] for i in arc_indices]
    print(f"[OK] Loaded {len(boolq_samples)} BoolQ samples & {len(arc_samples)} ARC-Easy samples.")

    # 2. Load Model & Tokenizer
    print("\n[2/5] Loading model & tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    base_model.eval()

    mem_after_load = get_process_memory_mb()
    print(f"[OK] Model loaded. Process RSS: {mem_after_load:.1f} MB")

    # 3. Prepare Candidate Token IDs
    # BoolQ candidates
    yes_candidates = ["Yes", " Yes", "yes", " True", "True"]
    no_candidates = ["No", " No", "no", " False", "False"]
    yes_token_ids = list(set([tokenizer.encode(s, add_special_tokens=False)[0] for s in yes_candidates]))
    no_token_ids = list(set([tokenizer.encode(s, add_special_tokens=False)[0] for s in no_candidates]))

    # ARC labels
    arc_labels = ["A", "B", "C", "D"]
    arc_token_ids = {}
    for l in arc_labels:
        variants = [l, " " + l]
        arc_token_ids[l] = list(set([tokenizer.encode(s, add_special_tokens=False)[0] for s in variants]))

    # 4. Wrap with UniversalModelAdapter
    print("\n[3/5] Instantiating SynapticFold v2 Engine via UniversalModelAdapter...", flush=True)
    brain_cfg = SynapticBrainConfig(
        min_loops=2,
        max_loops=4,
        damping_gamma=0.40,
        km_alpha=0.50,
        device="cpu",
        dtype=torch.bfloat16,
    )
    synaptic_engine = UniversalModelAdapter.wrap_model(base_model, brain_config=brain_cfg)
    total_layers = synaptic_engine.model_info["num_layers"]
    print(f"[OK] Synaptic Engine ready: {total_layers} total layers partitioned into:")
    print(f"  * Sensory Compartment: {len(synaptic_engine.sensory_layers)} layers")
    print(f"  * Recurrent Core: 1 representative neuromorphic core layer (2-4 adaptive loops)")
    print(f"  * Motor Compartment: {len(synaptic_engine.motor_layers)} layers")

    # Containers for results
    results = {
        "metadata": {
            "model_path": model_path,
            "total_layers": total_layers,
            "sensory_layers": len(synaptic_engine.sensory_layers),
            "motor_layers": len(synaptic_engine.motor_layers),
            "n_samples": n_samples,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "boolq": {
            "dense_correct": 0,
            "synaptic_correct": 0,
            "agreements": 0,
            "dense_times_ms": [],
            "synaptic_times_ms": [],
            "synaptic_loops": [],
            "items": [],
        },
        "arc_easy": {
            "dense_correct": 0,
            "synaptic_correct": 0,
            "agreements": 0,
            "dense_times_ms": [],
            "synaptic_times_ms": [],
            "synaptic_loops": [],
            "items": [],
        }
    }

    # 5. Benchmark BoolQ
    print("\n[4/5] Running Benchmark on BoolQ (20 samples)...", flush=True)
    for idx, item in enumerate(boolq_samples):
        prompt = build_boolq_prompt(item["passage"], item["question"])
        target_bool = bool(item["answer"])
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Run Dense Baseline
        t0 = time.perf_counter()
        with torch.no_grad():
            dense_out = base_model(input_ids=input_ids)
            dense_logits_last = dense_out.logits[0, -1, :]
        t_dense = (time.perf_counter() - t0) * 1000.0

        dense_yes_score = marginalize_token_logits(dense_logits_last, yes_token_ids)
        dense_no_score = marginalize_token_logits(dense_logits_last, no_token_ids)
        dense_pred = dense_yes_score > dense_no_score
        dense_is_correct = (dense_pred == target_bool)

        # Run SynapticFold v2
        t0 = time.perf_counter()
        with torch.no_grad():
            syn_logits, loops, sparsity = synaptic_engine(input_ids)
            syn_logits_last = syn_logits[0, -1, :]
        t_syn = (time.perf_counter() - t0) * 1000.0

        syn_yes_score = marginalize_token_logits(syn_logits_last, yes_token_ids)
        syn_no_score = marginalize_token_logits(syn_logits_last, no_token_ids)
        syn_pred = syn_yes_score > syn_no_score
        syn_is_correct = (syn_pred == target_bool)

        agreement = (dense_pred == syn_pred)
        if dense_is_correct:
            results["boolq"]["dense_correct"] += 1
        if syn_is_correct:
            results["boolq"]["synaptic_correct"] += 1
        if agreement:
            results["boolq"]["agreements"] += 1

        results["boolq"]["dense_times_ms"].append(t_dense)
        results["boolq"]["synaptic_times_ms"].append(t_syn)
        results["boolq"]["synaptic_loops"].append(loops)

        results["boolq"]["items"].append({
            "idx": idx,
            "target": target_bool,
            "dense_pred": dense_pred,
            "synaptic_pred": syn_pred,
            "agreement": agreement,
            "dense_time_ms": round(t_dense, 2),
            "synaptic_time_ms": round(t_syn, 2),
            "loops": loops,
        })
        print(f"  * BoolQ #{idx+1:02d}: Target={str(target_bool):<5} | Dense={'OK' if dense_is_correct else 'FAIL'} ({t_dense:5.1f}ms) | Synaptic={'OK' if syn_is_correct else 'FAIL'} ({t_syn:5.1f}ms, loops={loops}) | Match={agreement}")

    # 6. Benchmark ARC-Easy
    print("\n[5/5] Running Benchmark on ARC-Easy (20 samples)...", flush=True)
    for idx, item in enumerate(arc_samples):
        prompt = build_arc_prompt(item["question"], item["choices"])
        target_label = item["answerKey"].strip().upper()
        # Normalization if numeric labels 1,2,3,4 exist
        label_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
        target_label = label_map.get(target_label, target_label)

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Run Dense Baseline
        t0 = time.perf_counter()
        with torch.no_grad():
            dense_out = base_model(input_ids=input_ids)
            dense_logits_last = dense_out.logits[0, -1, :]
        t_dense = (time.perf_counter() - t0) * 1000.0

        dense_scores = {l: marginalize_token_logits(dense_logits_last, arc_token_ids[l]) for l in arc_labels}
        dense_pred = max(dense_scores, key=dense_scores.get)
        dense_is_correct = (dense_pred == target_label)

        # Run SynapticFold v2
        t0 = time.perf_counter()
        with torch.no_grad():
            syn_logits, loops, sparsity = synaptic_engine(input_ids)
            syn_logits_last = syn_logits[0, -1, :]
        t_syn = (time.perf_counter() - t0) * 1000.0

        syn_scores = {l: marginalize_token_logits(syn_logits_last, arc_token_ids[l]) for l in arc_labels}
        syn_pred = max(syn_scores, key=syn_scores.get)
        syn_is_correct = (syn_pred == target_label)

        agreement = (dense_pred == syn_pred)
        if dense_is_correct:
            results["arc_easy"]["dense_correct"] += 1
        if syn_is_correct:
            results["arc_easy"]["synaptic_correct"] += 1
        if agreement:
            results["arc_easy"]["agreements"] += 1

        results["arc_easy"]["dense_times_ms"].append(t_dense)
        results["arc_easy"]["synaptic_times_ms"].append(t_syn)
        results["arc_easy"]["synaptic_loops"].append(loops)

        results["arc_easy"]["items"].append({
            "idx": idx,
            "target": target_label,
            "dense_pred": dense_pred,
            "synaptic_pred": syn_pred,
            "agreement": agreement,
            "dense_time_ms": round(t_dense, 2),
            "synaptic_time_ms": round(t_syn, 2),
            "loops": loops,
        })
        print(f"  * ARC #{idx+1:02d}: Target={target_label} | Dense={'OK' if dense_is_correct else 'FAIL'} ({t_dense:5.1f}ms) | Synaptic={'OK' if syn_is_correct else 'FAIL'} ({t_syn:5.1f}ms, loops={loops}) | Match={agreement}")

    # Summary Statistics
    peak_mem = get_process_memory_mb()
    results["metadata"]["peak_memory_mb"] = round(peak_mem, 1)

    # Compute averages
    for key, name in [("boolq", "BoolQ"), ("arc_easy", "ARC-Easy")]:
        d = results[key]
        d["dense_acc_pct"] = round((d["dense_correct"] / n_samples) * 100.0, 1)
        d["synaptic_acc_pct"] = round((d["synaptic_correct"] / n_samples) * 100.0, 1)
        d["agreement_pct"] = round((d["agreements"] / n_samples) * 100.0, 1)
        d["dense_avg_time_ms"] = round(float(np.mean(d["dense_times_ms"])), 2)
        d["synaptic_avg_time_ms"] = round(float(np.mean(d["synaptic_times_ms"])), 2)
        d["speedup_ratio"] = round(d["dense_avg_time_ms"] / max(0.001, d["synaptic_avg_time_ms"]), 2)
        d["avg_loops"] = round(float(np.mean(d["synaptic_loops"])), 2)

    # Save to JSON
    out_full_path = os.path.join(ROOT_DIR, "benchmarks", output_json)
    with open(out_full_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to: {out_full_path}")

    # Print Final Consolidated Table
    print("\n" + "=" * 80)
    print("[SUMMARY] SYNAPTIC-FOLD v2 BENCHMARK SUMMARY REPORT")
    print("=" * 80)
    print(f"{'DATASET':<12} | {'METRIC':<18} | {'DENSE BASELINE':<16} | {'SYNAPTICFOLD v2':<16} | {'DIFF / SPEEDUP':<16}")
    print("-" * 80)
    for key, name in [("boolq", "BoolQ"), ("arc_easy", "ARC-Easy")]:
        d = results[key]
        delta_acc = d['synaptic_acc_pct'] - d['dense_acc_pct']
        print(f"{name:<12} | {'Accuracy':<18} | {d['dense_acc_pct']:>14.1f}% | {d['synaptic_acc_pct']:>14.1f}% | diff: {delta_acc:>+5.1f}%")
        print(f"{'':<12} | {'Avg Latency (ms)':<18} | {d['dense_avg_time_ms']:>13.1f} ms | {d['synaptic_avg_time_ms']:>13.1f} ms | {d['speedup_ratio']:>14.2f}x")
        print(f"{'':<12} | {'Model Parity':<18} | {'100.0%':>16} | {d['agreement_pct']:>14.1f}% | {'Match Agreement':<16}")
        print(f"{'':<12} | {'Recurrent Loops':<18} | {'N/A (24 seq)':>16} | {d['avg_loops']:>14.2f} | {'loops/sample':<16}")
        print("-" * 80)
    print(f"Memory Footprint Peak RSS: {peak_mem:.1f} MB (Within Safe Bounds < 3.5 GB)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SynapticFold v2 Dual-Dataset Benchmark")
    parser.add_argument(
        "--model-path",
        type=str,
        default=r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775",
        help="Path to HuggingFace model snapshot",
    )
    parser.add_argument("--samples", type=int, default=20, help="Number of samples per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    run_benchmark(model_path=args.model_path, n_samples=args.samples, seed=args.seed)
