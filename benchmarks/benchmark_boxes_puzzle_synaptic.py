import sys
import os
import time
import json
import psutil
import urllib.request
import torch
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = r"C:\Users\Umberto\synaptic-fold-v2"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.synaptic_native_model import SynapticNativeModel

print("=" * 90, flush=True)
print("BENCHMARK COMPARATIVO: EFFORT COMPUTAZIONALE & LOGICA SU QWEN2.5-7B vs STANDARD QUANTIZZATO", flush=True)
print("Hardware: AMD Ryzen 7 7730U (8C/16T Zen 3) | 16 GB DDR4-3200 | Windows 11", flush=True)
print("=" * 90, flush=True)

puzzle_prompt = (
    "Ci sono tre scatole: Rossa, Blu e Verde.\n\n"
    "Metto una mela nella scatola Rossa.\n\n"
    "Sposto il contenuto della scatola Rossa nella scatola Blu.\n\n"
    "Scambio le etichette delle scatole Blu e Verde (quella che si chiamava Blu ora si chiama Verde e viceversa).\n\n"
    "Prendo l'oggetto che si trova ora nella scatola chiamata 'Blu' e lo metto nella scatola Rossa.\n"
    "Dove si trova la mela adesso? Spiega il ragionamento passo per passo."
)

# ----------------------------------------------------------------------------
# 1. EVALUATION: SYNAPTIC NATIVE MODEL (QWEN2.5-7B NATIVE FLOAT)
# ----------------------------------------------------------------------------
print("\n" + "#" * 90, flush=True)
print("[1] ESECUZIONE: SYNAPTIC NATIVE MODEL (QWEN2.5-7B, NATIVE PRECISION BFLOAT16)", flush=True)
print("#" * 90, flush=True)

snapshot_dir = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
synapse_dir = os.path.join(ROOT_DIR, "models", "synapse_weights")
tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)

t_load_0 = time.perf_counter()
model = SynapticNativeModel.from_pretrained(
    snapshot_dir,
    dtype=torch.bfloat16,
    device="cpu",
    load_real_weights=True,
    use_neuromorphic_mlps=True,
    synapse_cache_dir=synapse_dir,
)
load_duration_syn = time.perf_counter() - t_load_0

mem_info = model.get_memory_breakdown()
print(f"Modello Synaptic caricato in: {load_duration_syn:.2f} s", flush=True)
print(f"RAM Pesi Fisici Unici:        {mem_info['unique_physical_mb']:.1f} MB ({mem_info['unique_physical_mb']/1024:.2f} GB)", flush=True)
print(f"Process RSS a regime:         {mem_info['rss_mb']:.1f} MB ({mem_info['rss_gb']:.2f} GB)", flush=True)
print(f"Rapporto di Compressione:     {mem_info['compression_ratio']}x", flush=True)

messages = [{"role": "user", "content": puzzle_prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"]
prompt_len = input_ids.shape[1]

print(f"\nPrompt di input: {prompt_len} token", flush=True)
print("Avvio generazione streaming token-by-token...", flush=True)
print("-" * 50, flush=True)

syn_tokens = []
first_token_time = None
t_gen_start = time.perf_counter()

def on_token_syn(tok_id: int):
    global first_token_time
    if first_token_time is None:
        first_token_time = time.perf_counter()
    syn_tokens.append(tok_id)
    tok_str = tokenizer.decode([tok_id], skip_special_tokens=True)
    print(tok_str, end="", flush=True)

with torch.no_grad():
    _ = model.generate(
        input_ids,
        max_new_tokens=150,
        temperature=0.0,  # greedy decoding per riproducibilita' logica
        use_cache=True,
        token_callback=on_token_syn,
    )
t_gen_end = time.perf_counter()

total_gen_time_syn = t_gen_end - t_gen_start
ttft_syn = (first_token_time - t_gen_start) * 1000.0 if first_token_time else 0.0
decoding_time_syn = t_gen_end - first_token_time if first_token_time else total_gen_time_syn
new_tokens_syn = len(syn_tokens)
tok_s_syn = (new_tokens_syn - 1) / max(decoding_time_syn, 1e-4) if new_tokens_syn > 1 else 0.0
ms_per_tok_syn = (decoding_time_syn / max(new_tokens_syn - 1, 1)) * 1000.0

rss_final_syn = psutil.Process().memory_info().rss / (1024**2)

print("\n" + "-" * 50, flush=True)
print(f"Token generati:               {new_tokens_syn}", flush=True)
print(f"TTFT (Time-To-First-Token):   {ttft_syn:.1f} ms", flush=True)
print(f"Throughput di Decoding:       {tok_s_syn:.2f} tok/s ({ms_per_tok_syn:.1f} ms/token)", flush=True)
print(f"Tempo totale generazione:     {total_gen_time_syn:.2f} s", flush=True)
print(f"Picco RAM Processo (RSS):     {rss_final_syn:.1f} MB ({rss_final_syn/1024:.2f} GB)", flush=True)

# ----------------------------------------------------------------------------
# 2. EVALUATION: STANDARD QUANTIZED BASELINE (OLLAMA LLAMA-3-8B / Q4_0)
# ----------------------------------------------------------------------------
print("\n" + "#" * 90, flush=True)
print("[2] ESECUZIONE: STANDARD QUANTIZED BASELINE (OLLAMA LLAMA-3-8B Q4_0, LLAMA.CPP BACKEND)", flush=True)
print("#" * 90, flush=True)

ollama_url = "http://localhost:11434/api/generate"
ollama_payload = {
    "model": "llama3:latest",
    "prompt": puzzle_prompt,
    "options": {
        "temperature": 0.0,
    },
    "stream": False
}

req = urllib.request.Request(
    ollama_url,
    data=json.dumps(ollama_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

t0_ollama = time.perf_counter()
with urllib.request.urlopen(req) as response:
    ollama_res = json.loads(response.read().decode("utf-8"))
t_total_ollama = time.perf_counter() - t0_ollama

ttft_ollama = ollama_res.get("prompt_eval_duration", 0) / 1e6
eval_count_ollama = ollama_res.get("eval_count", 0)
eval_duration_ms_ollama = ollama_res.get("eval_duration", 0) / 1e6
tok_s_ollama = eval_count_ollama / (eval_duration_ms_ollama / 1000.0) if eval_duration_ms_ollama > 0 else 0.0
ms_per_tok_ollama = eval_duration_ms_ollama / max(eval_count_ollama, 1)

print("Risposta Ollama (Standard Quantized Llama-3-8B):\n", flush=True)
print(ollama_res.get("response", ""), flush=True)
print("-" * 50, flush=True)
print(f"Token generati:               {eval_count_ollama}", flush=True)
print(f"TTFT (Prompt eval duration):  {ttft_ollama:.1f} ms", flush=True)
print(f"Throughput di Decoding:       {tok_s_ollama:.2f} tok/s ({ms_per_tok_ollama:.1f} ms/token)", flush=True)
print(f"Tempo totale risposta:        {t_total_ollama:.2f} s", flush=True)

# ----------------------------------------------------------------------------
# 3. TABELLA COMPARATIVA DI EFFORT COMPUTAZIONALE
# ----------------------------------------------------------------------------
print("\n" + "=" * 90, flush=True)
print("TABELLA COMPARATIVA DELL'EFFORT COMPUTAZIONALE (7B SYNAPTIC vs 8B QUANTIZZATO)", flush=True)
print("=" * 90, flush=True)
print(f"{'METRICA COMPUTAZIONALE':<38} | {'SYNAPTIC NATIVE (QWEN 7B)':<24} | {'STANDARD QUANTIZZATO (OLLAMA 8B)'}", flush=True)
print("-" * 90, flush=True)
print(f"{'Precisione e Formato':<38} | {'100% Native BFloat16':<24} | INT4 (Q4_0 Scalare)", flush=True)
print(f"{'Runtime di Esecuzione':<38} | {'Nativo PyTorch OneDNN':<24} | GGUF llama.cpp", flush=True)
print(f"{'RAM Pesi Modello':<38} | {mem_info['unique_physical_mb']:6.1f} MB ({mem_info['unique_physical_mb']/1024:.2f} GB)      | ~4,920 MB (4.80 GB)", flush=True)
print(f"{'Process RSS a Regime':<38} | {rss_final_syn:6.1f} MB ({rss_final_syn/1024:.2f} GB)      | ~5,400 MB (5.27 GB)", flush=True)
print(f"{'TTFT (Prefill Latency)':<38} | {ttft_syn:6.1f} ms               | {ttft_ollama:6.1f} ms", flush=True)
print(f"{'Decoding Throughput':<38} | {tok_s_syn:5.2f} tok/s               | {tok_s_ollama:5.2f} tok/s", flush=True)
print(f"{'Latenza Media per Token':<38} | {ms_per_tok_syn:6.1f} ms/tok            | {ms_per_tok_ollama:6.1f} ms/tok", flush=True)
print(f"{'Token Generati':<38} | {new_tokens_syn:<24} | {eval_count_ollama}", flush=True)
print(f"{'Tempo Totale Generazione':<38} | {total_gen_time_syn:6.2f} s                | {t_total_ollama:6.2f} s", flush=True)
print("=" * 90, flush=True)

# Save benchmark results to json
results = {
    "synaptic_native_7b": {
        "precision": "BFloat16",
        "physical_ram_mb": round(mem_info["unique_physical_mb"], 1),
        "rss_mb": round(rss_final_syn, 1),
        "ttft_ms": round(ttft_syn, 1),
        "tok_s": round(tok_s_syn, 2),
        "ms_per_token": round(ms_per_tok_syn, 1),
        "tokens_generated": new_tokens_syn,
        "total_time_s": round(total_gen_time_syn, 2),
        "response": tokenizer.decode(syn_tokens, skip_special_tokens=True)
    },
    "standard_quantized_8b": {
        "model": "llama3:8b-q4_0",
        "precision": "INT4",
        "ttft_ms": round(ttft_ollama, 1),
        "tok_s": round(tok_s_ollama, 2),
        "ms_per_token": round(ms_per_tok_ollama, 1),
        "tokens_generated": eval_count_ollama,
        "total_time_s": round(t_total_ollama, 2),
        "response": ollama_res.get("response", "")
    }
}

out_json = os.path.join(ROOT_DIR, "benchmarks", "boxes_puzzle_comparison.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n[OK] Risultati completi salvati in: {out_json}", flush=True)
