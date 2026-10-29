"""
SynapticFold: Real 7B End-to-End Generation Benchmark.

Compares full unquantized Qwen2.5-7B-Instruct:
1. Baseline Dense 7B (All 28 layers dense, 18,944 neurons per layer)
2. Neurosynaptic Sparse 7B (All 28 layers sparse, 35% active neurons per token)

Validates reasoning depth, Python coding capability, and Italian fluency on the REAL 7B.
"""

import os
import sys
import time
import json
import psutil
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.converter import NeurosynapticConverter
from synaptic_fold.spatial_router import NeurosynapticSparseMLP

SNAPSHOT_7B = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
OUT_JSON = os.path.join(ROOT_DIR, "synaptic_fold", "data", "real_7b_sparse_vs_dense_benchmark.json")

TEST_PROMPTS = [
    "Ciao se io mi chiamo umberto, tu come ti chiami?",
    "Qual e il tuo ruolo e come puoi aiutarmi a programmare in Python?",
    "Scrivi una funzione Python concisa per calcolare il fattoriale con memoization.",
]


def run_benchmark():
    print("=" * 80, flush=True)
    print("🚀 SYNAPTICFOLD: BENCHMARK UFFICIALE SUL VERO QWEN2.5-7B-INSTRUCT (28 STRATI)", flush=True)
    print("   Confronto Side-by-Side: Modello Denso 7B vs Modello Neurosintaptico Sparso 7B", flush=True)
    print("=" * 80, flush=True)

    print("\n[Caricamento] Inizializzazione Qwen2.5-7B-Instruct non quantizzato (BFloat16)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_7B, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT_7B,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        local_files_only=True,
    )
    model.eval()

    rss = psutil.Process().memory_info().rss / (1024**2)
    print(f"  -> Qwen2.5-7B pronto! RAM Processo: {rss:.1f} MB ({rss/1024:.2f} GB)\n", flush=True)

    # ------------------------------------------------------------------------
    # 1. BASELINE: QWEN2.5-7B DENSO ORIGINALE
    # ------------------------------------------------------------------------
    print("=" * 80, flush=True)
    print("1️⃣ ESECUZIONE BASELINE: VERO QWEN2.5-7B DENSO ORIGINALE (100% Neuroni)", flush=True)
    print("=" * 80, flush=True)

    dense_results = {}
    for p in TEST_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(fmt, return_tensors="pt")

        print(f"👤 [Prompt]: {p}", flush=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                **inp,
                max_new_tokens=60,
                temperature=0.7,
                top_k=20,
                do_sample=True,
            )
        dt = time.perf_counter() - t0
        gen_toks = out_ids[0, inp["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
        speed = len(gen_toks) / max(dt, 1e-4)

        print(f"🤖 [Dense 7B]:\n{text}", flush=True)
        print(f"⚡ [{len(gen_toks)} token in {dt:.2f}s ({speed:.2f} tok/s)]\n", flush=True)
        dense_results[p] = {"text": text, "speed": speed, "time": dt}

    # ------------------------------------------------------------------------
    # 2. CONVERSIONE: TUTTI I 28 STRATI CONVERTITI IN NEUROSINAPTICO SPARSO
    # ------------------------------------------------------------------------
    print("=" * 80, flush=True)
    print("2️⃣ CONVERSIONE IN MODELLO NEUROSINAPTICO SPARSO (35% Neuroni Attivi per Token)", flush=True)
    print("   * Tutti i 28 blocchi MLP (18.944 neuroni ciascuno) convertiti in NeurosynapticSparseMLP.", flush=True)
    print("   * 65% dei neuroni sub-soglia spenti ad ogni token.", flush=True)
    print("=" * 80, flush=True)

    model = NeurosynapticConverter.convert_model_to_sparse(
        model=model,
        active_ratio=0.35, # 35% active, 65% zeroed
        routing_mode="exact_topk",
    )

    rss = psutil.Process().memory_info().rss / (1024**2)
    print(f"  -> Modello Neurosintaptico pronto! RAM Processo: {rss:.1f} MB ({rss/1024:.2f} GB)\n", flush=True)

    # ------------------------------------------------------------------------
    # 3. ESECUZIONE: QWEN2.5-7B NEUROSINAPTICO SPARSO
    # ------------------------------------------------------------------------
    print("=" * 80, flush=True)
    print("3️⃣ ESECUZIONE NEUROSINAPTICA: VERO QWEN2.5-7B SPARSO (35% Neuroni Attivi)", flush=True)
    print("=" * 80, flush=True)

    sparse_results = {}
    for p in TEST_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(fmt, return_tensors="pt")

        print(f"👤 [Prompt]: {p}", flush=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                **inp,
                max_new_tokens=60,
                temperature=0.7,
                top_k=20,
                do_sample=True,
            )
        dt = time.perf_counter() - t0
        gen_toks = out_ids[0, inp["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
        speed = len(gen_toks) / max(dt, 1e-4)

        print(f"🧠 [Neurosynaptic 7B (35% Neuroni)]:\n{text}", flush=True)
        print(f"⚡ [{len(gen_toks)} token in {dt:.2f}s ({speed:.2f} tok/s)]\n", flush=True)
        sparse_results[p] = {"text": text, "speed": speed, "time": dt}

    # ------------------------------------------------------------------------
    # 4. RIEPILOGO FINALE SIDE-BY-SIDE
    # ------------------------------------------------------------------------
    print("=" * 80, flush=True)
    print("🏁 CONFRONTO QUALITATIVO DIRETTO: INTELLIGENZA E RAGIONAMENTO", flush=True)
    print("=" * 80, flush=True)
    for p in TEST_PROMPTS:
        print(f"\n[DOMANDA]: {p}", flush=True)
        print(f"  • VERO 7B DENSO:        {dense_results[p]['text'][:100]}...", flush=True)
        print(f"  • VERO 7B NEUROSINAPTICO: {sparse_results[p]['text'][:100]}...", flush=True)

    output_payload = {
        "dense_7b": dense_results,
        "neurosynaptic_sparse_7b": sparse_results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[OK] Risultati salvati in: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    run_benchmark()
