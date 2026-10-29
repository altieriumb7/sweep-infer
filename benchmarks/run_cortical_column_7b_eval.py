import os
import sys
import time
import gc
import psutil
import torch
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.synaptic_cortical_column_7b import SynapticCorticalColumn7B, CorticalColumn7BConfig

SNAPSHOT_7B = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"

print("=" * 80)
print("BENCHMARK: SYNAPTIC CORTICAL COLUMN 7B (APPROACH B - 16-BIT CANONICAL CORE)")
print("Hardware: AMD Ryzen 7 7730U (16 Threads) | Precision: Pure Native BFloat16")
print("Target: Resident RAM <= 5.18 GB | Fast AVX-2 Batched Prefill | No Degradation")
print("=" * 80, flush=True)

tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_7B, local_files_only=True)

# 1. Selective Stream Load (Only 5 layers in BFloat16)
print("Loading 5-Layer Canonical Cortical Column Architecture in Pure BFloat16...", flush=True)
t0 = time.perf_counter()
model = SynapticCorticalColumn7B.from_pretrained(SNAPSHOT_7B)
t_load = time.perf_counter() - t0
print(f"Loaded and mapped in {t_load:.2f} s!", flush=True)

mem = model.get_memory_breakdown()
print("\n" + "=" * 50)
print("MEMORY PROFILE:")
print(f"  Model Parameters/Buffers: {mem['model_ram_mb']:.1f} MB ({mem['model_ram_gb']:.2f} GB)")
print(f"  Process Resident RAM:     {mem['rss_mb']:.1f} MB ({mem['rss_gb']:.2f} GB)")
print(f"  RAM Limit Compliance:     {'PASSED (<= 5.18 GB)' if mem['rss_gb'] <= 5.18 else 'WARNING'}")
print("=" * 50 + "\n", flush=True)

# 2. Calibration / Identity Bypass for Manifold Bridge
print("Setting Manifold Bridge to clean Identity Pass-through...", flush=True)
model.bridge.is_calibrated = False # Clean pass-through

# 3. Autonomous Testing Suite
tests = [
    ("TIER 1: SANITY GREETING", "Ciao! Chi sei e qual è il tuo ruolo?", 30),
    ("TIER 2: THE THREE BOXES PUZZLE", 
     "Ci sono tre scatole: Rossa, Blu e Verde.\n"
     "1. Metto una mela nella scatola Rossa.\n"
     "2. Sposto il contenuto della scatola Rossa nella scatola Blu.\n"
     "3. Scambio le etichette delle scatole Blu e Verde (quella che si chiamava Blu ora si chiama Verde e viceversa).\n"
     "4. Prendo l'oggetto che si trova ora nella scatola chiamata 'Blu' e lo metto nella scatola Rossa.\n"
     "Dove si trova la mela adesso? Spiega il ragionamento passo per passo.", 60),
    ("TIER 3: THE TIED WOLF RIDDLE",
     "Un contadino deve trasportare un lupo, una pecora e un cavolo dall'altra parte di un fiume con una barca che può portare solo lui e un elemento alla volta. "
     "Tuttavia, il lupo è legato da una corda corta a un albero fin dall'inizio e non può muoversi né mangiare nulla. Qual è la sequenza ottimale di viaggi?", 60)
]

for title, prompt_text, max_tok in tests:
    print("=" * 70, flush=True)
    print(f">>> {title}", flush=True)
    print("=" * 70, flush=True)
    
    msgs = [{"role": "user", "content": prompt_text}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(fmt, return_tensors="pt")["input_ids"]
    
    print(f"Prompt Length: {input_ids.shape[1]} tokens", flush=True)
    print("Streaming output: ", end="", flush=True)
    
    tokens_out = []
    def on_token(tid: int):
        tokens_out.append(tid)
        txt = tokenizer.decode([tid], skip_special_tokens=True)
        print(txt, end="", flush=True)
        
    t0_gen = time.perf_counter()
    _ = model.generate(
        input_ids,
        max_new_tokens=max_tok,
        temperature=0.0,
        top_k=1,
        token_callback=on_token,
    )
    dt_gen = time.perf_counter() - t0_gen
    speed = len(tokens_out) / max(dt_gen, 1e-4)
    print(f"\n\n[Benchmark: {len(tokens_out)} tokens in {dt_gen:.2f} s ({speed:.2f} tok/s | TTFT + Decode)]\n", flush=True)

print("=" * 80)
print("AUTONOMOUS EVALUATION COMPLETE!")
print("=" * 80, flush=True)
