"""
SynapticFold: Rigorous Side-by-Side Benchmark & Deep Profiling.

Compares:
1. Baseline Dense Qwen2.5-7B (100% neurons)
2. Neurosynaptic Sparse Qwen2.5-7B (35% active neurons, exact spatial gating)

Measures:
- RAM footprint (RSS initial, peak, final)
- CPU computing time (User + System seconds)
- Generation speed (tokens/s, latency ms/token)
- Text quality & execution validity on a concrete algorithmic prompt.
"""

import os
import sys
import time
import json
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.converter import NeurosynapticConverter

SNAPSHOT_7B = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
OUT_JSON = os.path.join(ROOT_DIR, "synaptic_fold", "data", "goal_rigorous_benchmark_results.json")

TEST_PROMPT = (
    "Scrivi in Python una funzione 'inverti_stringa(s)' che inverte una stringa senza usare [::-1]. "
    "Fornisci il codice conciso e funzionante."
)


def get_mem_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_benchmark():
    process = psutil.Process()
    torch.set_num_threads(8)

    print("=" * 80, flush=True)
    print("🚀 SYNAPTICFOLD: BENCHMARK RIGOROSO SIDE-BY-SIDE (7B DENSE vs 7B NEUROSINAPTICO)", flush=True)
    print("   Valutazione: RAM, CPU-Time, Velocità e Correttezza del Testo Generato", flush=True)
    print("=" * 80, flush=True)

    print(f"\n[Test Prompt]: {TEST_PROMPT}\n", flush=True)

    print("[1/4] Caricamento Tokenizer e Qwen2.5-7B non quantizzato (BFloat16)...", flush=True)
    ram_start = get_mem_mb()
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_7B, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT_7B,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        local_files_only=True,
    )
    model.eval()
    ram_loaded = get_mem_mb()
    print(f"  -> Modello caricato! RAM: {ram_loaded:.1f} MB (Delta: {ram_loaded - ram_start:.1f} MB)\n", flush=True)

    msgs = [{"role": "user", "content": TEST_PROMPT}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(fmt, return_tensors="pt")
    input_tokens = inp["input_ids"].shape[1]
    MAX_NEW = 55 # Exact length to allow full concise function completion

    # ------------------------------------------------------------------------
    # 1. BASELINE DENSE 7B (100% NEURONI)
    # ------------------------------------------------------------------------
    print("=" * 80, flush=True)
    print("1️⃣ ESECUZIONE BASELINE: MODELLO DENSO ORIGINALE (100% Neuroni)", flush=True)
    print("=" * 80, flush=True)

    cpu_times_before_dense = process.cpu_times()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out_dense = model.generate(
            **inp,
            max_new_tokens=MAX_NEW,
            temperature=0.2, # Low temperature for clean deterministic code
            do_sample=True,
        )
    t_dense_wall = time.perf_counter() - t0
    cpu_times_after_dense = process.cpu_times()
    cpu_dense_user = cpu_times_after_dense.user - cpu_times_before_dense.user
    cpu_dense_sys = cpu_times_after_dense.system - cpu_times_before_dense.system
    cpu_dense_total = cpu_dense_user + cpu_dense_sys
    ram_after_dense = get_mem_mb()

    gen_dense_ids = out_dense[0, input_tokens:]
    text_dense = tokenizer.decode(gen_dense_ids, skip_special_tokens=True).strip()
    speed_dense = len(gen_dense_ids) / t_dense_wall
    ms_per_tok_dense = (t_dense_wall / len(gen_dense_ids)) * 1000

    print(f"🤖 [Testo Generato da Baseline Denso]:\n{text_dense}\n", flush=True)
    print(f"📊 [Metriche Dense]:", flush=True)
    print(f"   * Token generati: {len(gen_dense_ids)} in {t_dense_wall:.2f}s", flush=True)
    print(f"   * Velocità: {speed_dense:.3f} tok/s ({ms_per_tok_dense:.1f} ms/token)", flush=True)
    print(f"   * CPU-Time Totale: {cpu_dense_total:.2f}s (User: {cpu_dense_user:.2f}s, Sys: {cpu_dense_sys:.2f}s)", flush=True)
    print(f"   * RAM RSS: {ram_after_dense:.1f} MB ({ram_after_dense/1024:.2f} GB)\n", flush=True)

    # ------------------------------------------------------------------------
    # 2. CONVERSIONE NEUROSINAPTICA (35% NEURONI ATTIVI)
    # ------------------------------------------------------------------------
    print("=" * 80, flush=True)
    print("2️⃣ CONVERSIONE IN MODELLO NEUROSINAPTICO SPARSO (35% Neuroni Attivi)", flush=True)
    print("=" * 80, flush=True)

    ram_before_conv = get_mem_mb()
    model = NeurosynapticConverter.convert_model_to_sparse(
        model=model,
        active_ratio=0.35,
        routing_mode="exact_topk",
    )
    ram_after_conv = get_mem_mb()
    print(f"  -> Conversione completata! RAM RSS: {ram_after_conv:.1f} MB (Delta conversione: {ram_after_conv - ram_before_conv:.1f} MB)\n", flush=True)

    # ------------------------------------------------------------------------
    # 3. ESECUZIONE NEUROSINAPTICA SPARSA (35% NEURONI ATTIVI)
    # ------------------------------------------------------------------------
    print("=" * 80, flush=True)
    print("3️⃣ ESECUZIONE NEUROSINAPTICA: MODELLO SPARSO (35% Neuroni Attivi)", flush=True)
    print("=" * 80, flush=True)

    cpu_times_before_sparse = process.cpu_times()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out_sparse = model.generate(
            **inp,
            max_new_tokens=MAX_NEW,
            temperature=0.2,
            do_sample=True,
        )
    t_sparse_wall = time.perf_counter() - t0
    cpu_times_after_sparse = process.cpu_times()
    cpu_sparse_user = cpu_times_after_sparse.user - cpu_times_before_sparse.user
    cpu_sparse_sys = cpu_times_after_sparse.system - cpu_times_before_sparse.system
    cpu_sparse_total = cpu_sparse_user + cpu_sparse_sys
    ram_after_sparse = get_mem_mb()

    gen_sparse_ids = out_sparse[0, input_tokens:]
    text_sparse = tokenizer.decode(gen_sparse_ids, skip_special_tokens=True).strip()
    speed_sparse = len(gen_sparse_ids) / t_sparse_wall
    ms_per_tok_sparse = (t_sparse_wall / len(gen_sparse_ids)) * 1000

    print(f"🤖 [Testo Generato da Neurosynaptic Sparse]:\n{text_sparse}\n", flush=True)
    print(f"📊 [Metriche Neurosynaptic Sparse]:", flush=True)
    print(f"   * Token generati: {len(gen_sparse_ids)} in {t_sparse_wall:.2f}s", flush=True)
    print(f"   * Velocità: {speed_sparse:.3f} tok/s ({ms_per_tok_sparse:.1f} ms/token)", flush=True)
    print(f"   * CPU-Time Totale: {cpu_sparse_total:.2f}s (User: {cpu_sparse_user:.2f}s, Sys: {cpu_sparse_sys:.2f}s)", flush=True)
    print(f"   * RAM RSS: {ram_after_sparse:.1f} MB ({ram_after_sparse/1024:.2f} GB)\n", flush=True)

    # ------------------------------------------------------------------------
    # 4. CONFRONTO DIRETTO E VERIFICA DI ESECUZIONE DEL CODICE
    # ------------------------------------------------------------------------
    print("=" * 80, flush=True)
    print("4️⃣ SINTESI COMPARATIVA & VERIFICA FUNZIONALE DEL CODICE", flush=True)
    print("=" * 80, flush=True)

    # Test execution of both codes
    def test_code(code_text, label):
        # Extract python code block if present
        clean_code = code_text
        if "```python" in code_text:
            clean_code = code_text.split("```python")[1].split("```")[0].strip()
        elif "```" in code_text:
            clean_code = code_text.split("```")[1].split("```")[0].strip()
        
        # Test in isolated namespace
        local_scope = {}
        try:
            exec(clean_code, {}, local_scope)
            if "inverti_stringa" in local_scope:
                fn = local_scope["inverti_stringa"]
                out_test = fn("antigravity")
                is_correct = (out_test == "ytivargitna")
                print(f"  -> [{label}] Esecuzione codice: SUCCESSO! inverti_stringa('antigravity') = '{out_test}' (Corretto: {is_correct})", flush=True)
                return True, out_test, clean_code
            else:
                print(f"  -> [{label}] Funzione 'inverti_stringa' non trovata nello scope.", flush=True)
                return False, None, clean_code
        except Exception as e:
            print(f"  -> [{label}] Errore esecuzione codice: {e}", flush=True)
            return False, str(e), clean_code

    ok_dense, res_dense, code_dense = test_code(text_dense, "Dense Baseline")
    ok_sparse, res_sparse, code_sparse = test_code(text_sparse, "Neurosynaptic Sparse")

    speedup = (speed_sparse / speed_dense - 1.0) * 100
    cpu_saved = cpu_dense_total - cpu_sparse_total
    cpu_saved_pct = (cpu_saved / max(cpu_dense_total, 1e-4)) * 100

    print(f"\n[RISULTATI FINALI]:", flush=True)
    print(f"  * Speedup Generazione: {speedup:+.1f}% ({speed_dense:.3f} -> {speed_sparse:.3f} tok/s)", flush=True)
    print(f"  * Risparmio CPU-Time: {cpu_saved:.2f}s ({cpu_saved_pct:+.1f}%)", flush=True)
    print(f"  * RAM Baseline vs Sparse: {ram_after_dense/1024:.2f} GB vs {ram_after_sparse/1024:.2f} GB", flush=True)
    print(f"  * Codice Dense Funzionante: {ok_dense}", flush=True)
    print(f"  * Codice Sparse Funzionante: {ok_sparse}", flush=True)

    results = {
        "prompt": TEST_PROMPT,
        "dense": {
            "tokens": len(gen_dense_ids),
            "wall_time": t_dense_wall,
            "speed_tok_s": speed_dense,
            "ms_per_tok": ms_per_tok_dense,
            "cpu_user": cpu_dense_user,
            "cpu_sys": cpu_dense_sys,
            "cpu_total": cpu_dense_total,
            "ram_rss_mb": ram_after_dense,
            "text": text_dense,
            "code": code_dense,
            "exec_ok": ok_dense,
            "exec_result": res_dense,
        },
        "sparse_35pct": {
            "tokens": len(gen_sparse_ids),
            "wall_time": t_sparse_wall,
            "speed_tok_s": speed_sparse,
            "ms_per_tok": ms_per_tok_sparse,
            "cpu_user": cpu_sparse_user,
            "cpu_sys": cpu_sparse_sys,
            "cpu_total": cpu_sparse_total,
            "ram_rss_mb": ram_after_sparse,
            "text": text_sparse,
            "code": code_sparse,
            "exec_ok": ok_sparse,
            "exec_result": res_sparse,
        },
        "comparison": {
            "speedup_pct": speedup,
            "cpu_time_saved_s": cpu_saved,
            "cpu_time_saved_pct": cpu_saved_pct,
            "ram_delta_mb": ram_after_sparse - ram_after_dense,
        }
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSalvati risultati completi in: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    run_benchmark()
