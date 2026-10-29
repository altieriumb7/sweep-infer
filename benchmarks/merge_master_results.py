"""
Merge baseline paper results with extended results into a master academic dataset.
"""

import json
import os

P1_PATH = "C:/Users/Umberto/synaptic-fold-v2/benchmarks/paper_multiaspect_results.json"
P2_PATH = "C:/Users/Umberto/synaptic-fold-v2/benchmarks/paper_extended_results.json"
OUT_PATH = "C:/Users/Umberto/synaptic-fold-v2/benchmarks/master_academic_results.json"

with open(P1_PATH, "r", encoding="utf-8") as f:
    d1 = json.load(f)

with open(P2_PATH, "r", encoding="utf-8") as f:
    d2 = json.load(f)

master = dict(d1)
master["metadata"]["total_benchmarks_evaluated"] = 7
master["metadata"]["benchmarks_list"] = [
    "GSM8K", "HumanEval", "BoolQ", "MBPP", "ARC-Challenge", "MMLU-STEM", "HellaSwag"
]

for model_name in ["Meta-Llama-3-8B-Instruct", "Qwen2.5-7B-Instruct"]:
    if model_name in d2["models"]:
        m2 = d2["models"][model_name]
        master["models"][model_name]["extended_benchmarks"] = {
            "mbpp": m2["benchmark_mbpp"],
            "arc_challenge": m2["benchmark_arc_challenge"],
            "mmlu_stem": m2["benchmark_mmlu_stem"],
            "hellaswag": m2["benchmark_hellaswag"]
        }

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(master, f, indent=2)

print("MASTER DATASET GENERATED AT:", OUT_PATH)
