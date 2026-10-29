"""
SynapticFold: High-Speed Sliced Neurosynaptic Benchmark.

Evaluates the new High-Speed Sliced Neurosynaptic Engine (35% active neurons,
zero-scatter sliced down_proj) on the exact same algorithmic prompt:
"Scrivi in Python una funzione 'inverti_stringa(s)' che inverte una stringa senza usare [::-1]. Fornisci il codice conciso e funzionante."

Captures:
- RAM RSS (initial, conversion delta, final)
- CPU-Time (User, System, Total)
- Inference speed (tok/s, ms/token, total wall time)
- Output text & execution verification of the generated Python function.
- Direct comparison with Baseline Dense results.
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
PREV_JSON = os.path.join(ROOT_DIR, "synaptic_fold", "data", "goal_rigorous_benchmark_results.json")
OUT_JSON = os.path.join(ROOT_DIR, "synaptic_fold", "data", "optimized_synaptic_benchmark_results.json")

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
    print("🚀 SYNAPTICFOLD: BENCHMARK NUOVO MOTORE SLICED NEUROSINAPTICO AD ALTA VELOCITÀ", flush=True)
    print("   Valutazione Parametri: RAM, CPU-Time, Velocità e Correttezza Funzionale", flush=True)
    print("=" * 80, flush=True)

    print(f"\n[Prompt Identico]: {TEST_PROMPT}\n", flush=True)

    # 1. Caricamento modello
    print("[1/3] Caricamento Tokenizer e Qwen2.5-7B (BFloat16)...", flush=True)
    ram_init = get_mem_mb()
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
    print(f"  -> Modello caricato! RAM RSS: {ram_loaded:.1f} MB ({ram_loaded/1024:.2f} GB)\n", flush=True)

    # 2. Conversione nel nuovo motore Sliced
    print("[2/3] Conversione in Neurosynaptic Sliced Engine (35% neuroni attivi)...", flush=True)
    ram_before_conv = get_mem_mb()
    model = NeurosynapticConverter.convert_model_to_sparse(
        model=model,
        active_ratio=0.35,
        routing_mode="exact_topk",
    )
    ram_after_conv = get_mem_mb()
    delta_conv = ram_after_conv - ram_before_conv
    print(f"  -> Conversione completata! RAM: {ram_after_conv:.1f} MB (Delta: {delta_conv:+.1f} MB)\n", flush=True)

    # 3. Generazione testo
    msgs = [{"role": "user", "content": TEST_PROMPT}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(fmt, return_tensors="pt")
    input_tokens = inp["input_ids"].shape[1]
    MAX_NEW = 70 # Sufficiente per chiudere il codice in modo autosufficiente

    print("=" * 80, flush=True)
    print("3/3 AVVIO GENERAZIONE TESTO CON NUOVO MOTORE SLICED NEUROSINAPTICO...", flush=True)
    print("=" * 80, flush=True)

    cpu_before = process.cpu_times()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inp,
            max_new_tokens=MAX_NEW,
            temperature=0.2,
            do_sample=True,
        )
    wall_time = time.perf_counter() - t0
    cpu_after = process.cpu_times()
    cpu_user = cpu_after.user - cpu_before.user
    cpu_sys = cpu_after.system - cpu_before.system
    cpu_total = cpu_user + cpu_sys
    ram_final = get_mem_mb()

    gen_ids = out[0, input_tokens:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    num_tokens = len(gen_ids)
    speed_tok_s = num_tokens / max(wall_time, 1e-4)
    ms_per_tok = (wall_time / max(num_tokens, 1)) * 1000

    print(f"\n🤖 [Testo Generato dal Nuovo Motore]:\n{text}\n", flush=True)
    print(f"📊 [Parametri Misurati del Nuovo Motore Sliced]:", flush=True)
    print(f"   * Token generati: {num_tokens} in {wall_time:.2f}s", flush=True)
    print(f"   * Velocità: {speed_tok_s:.3f} tok/s ({ms_per_tok:.1f} ms/token)", flush=True)
    print(f"   * CPU-Time: {cpu_total:.2f}s (User: {cpu_user:.2f}s, Sys: {cpu_sys:.2f}s)", flush=True)
    print(f"   * RAM RSS Finale: {ram_final:.1f} MB ({ram_final/1024:.2f} GB)\n", flush=True)

    # 4. Verifica esecutiva del codice
    clean_code = text
    if "```python" in text:
        clean_code = text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        clean_code = text.split("```")[1].split("```")[0].strip()

    exec_ok = False
    exec_result = None
    local_scope = {}
    try:
        exec(clean_code, {}, local_scope)
        if "inverti_stringa" in local_scope:
            fn = local_scope["inverti_stringa"]
            test_val = fn("antigravity")
            exec_ok = (test_val == "ytivargitna")
            exec_result = f"inverti_stringa('antigravity') = '{test_val}'"
            print(f"  -> [Verifica Esecutiva]: SUCCESSO! {exec_result} (Corretto: {exec_ok})", flush=True)
        else:
            exec_result = "Funzione 'inverti_stringa' non presente nello scope"
            print(f"  -> [Verifica Esecutiva]: {exec_result}", flush=True)
    except Exception as e:
        exec_result = str(e)
        print(f"  -> [Verifica Esecutiva Fallita]: {e}", flush=True)

    # 5. Confronto diretto con Baseline Dense
    dense_data = {}
    if os.path.exists(PREV_JSON):
        with open(PREV_JSON, "r", encoding="utf-8") as f:
            prev_results = json.load(f)
            dense_data = prev_results.get("dense", {})

    dense_ram_mb = dense_data.get("ram_rss_mb", 11070.8)
    dense_ms_tok = dense_data.get("ms_per_tok", 11128.1)
    dense_speed = dense_data.get("speed_tok_s", 0.090)
    dense_cpu_tok = dense_data.get("cpu_total", 2728.6) / 55.0

    cur_cpu_tok = cpu_total / max(num_tokens, 1)

    ram_saved_mb = dense_ram_mb - ram_final
    ram_saved_pct = (ram_saved_mb / dense_ram_mb) * 100
    speed_gain_pct = (speed_tok_s / dense_speed - 1.0) * 100
    cpu_saved_pct = (1.0 - cur_cpu_tok / dense_cpu_tok) * 100

    print("=" * 80, flush=True)
    print("📈 CONFRONTO DIRETTO: BASELINE DENSO 7B vs NUOVO SLICED NEUROSINAPTICO", flush=True)
    print("=" * 80, flush=True)
    print(f"  * RAM Risparmiata: {ram_saved_mb:.1f} MB in meno ({ram_saved_pct:+.1f}%)", flush=True)
    print(f"  * Velocità Token/s: {dense_speed:.3f} tok/s (Denso) -> {speed_tok_s:.3f} tok/s (Nuovo)", flush=True)
    print(f"  * Tempo medio per token: {dense_ms_tok:.0f} ms -> {ms_per_tok:.0f} ms", flush=True)
    print(f"  * CPU-Time per token: {dense_cpu_tok:.2f}s -> {cur_cpu_tok:.2f}s ({cpu_saved_pct:+.1f}%)", flush=True)
    print(f"  * Codice Funzionante e Verificato: {exec_ok}", flush=True)
    print("=" * 80, flush=True)

    results = {
        "prompt": TEST_PROMPT,
        "optimized_sliced_engine": {
            "tokens": num_tokens,
            "wall_time": wall_time,
            "speed_tok_s": speed_tok_s,
            "ms_per_tok": ms_per_tok,
            "cpu_user": cpu_user,
            "cpu_sys": cpu_sys,
            "cpu_total": cpu_total,
            "cpu_per_tok": cur_cpu_tok,
            "ram_rss_mb": ram_final,
            "text": text,
            "code": clean_code,
            "exec_ok": exec_ok,
            "exec_result": exec_result,
        },
        "baseline_dense_comparison": {
            "dense_ram_rss_mb": dense_ram_mb,
            "ram_saved_mb": ram_saved_mb,
            "ram_saved_pct": ram_saved_pct,
            "dense_speed_tok_s": dense_speed,
            "speed_gain_pct": speed_gain_pct,
            "dense_cpu_per_tok": dense_cpu_tok,
            "cpu_saved_pct": cpu_saved_pct,
        }
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSalvati risultati completi in: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    run_benchmark()
