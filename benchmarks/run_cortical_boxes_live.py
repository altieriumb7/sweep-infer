import os
import sys
import time
import torch
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = r"C:\Users\Umberto\synaptic-fold-v2"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.synaptic_cortical_engine import SynapticCorticalEngine, setup_optimal_cpu_threads

setup_optimal_cpu_threads()

SYNAPSE_FILE = os.path.join(ROOT_DIR, "models", "qwen2.5_7b_cortical.synapse")
SNAPSHOT_DIR = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"

print("=" * 80)
print("TEST LIVE: SYNAPTIC CORTICAL ENGINE 7B (ALL 28 DISTINCT LAYERS)")
print("Caricamento istantaneo da .synapse (Zero-Copy Native Binary)")
print("=" * 80, flush=True)

t0 = time.perf_counter()
model = SynapticCorticalEngine.from_synapse(SYNAPSE_FILE)
print(f"Modello 7B caricato in {time.perf_counter() - t0:.2f} s!", flush=True)

tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_DIR, local_files_only=True)

# Test 1: Sanity
p1 = "<|im_start|>user\nCiao! Chi sei?<|im_end|>\n<|im_start|>assistant\n"
ids1 = tokenizer(p1, return_tensors="pt")["input_ids"]

print("\n--- TEST 1: SALUTO DI IDENTITÀ ---", flush=True)
def print_tok(t):
    print(tokenizer.decode([t], skip_special_tokens=True), end="", flush=True)

t0_g = time.perf_counter()
out1 = model.generate(ids1, max_new_tokens=20, temperature=0.0, top_k=1, token_callback=print_tok)
print(f"\n[Test 1 completato in {time.perf_counter() - t0_g:.2f} s]\n", flush=True)

# Test 2: Boxes Puzzle
p2 = (
    "<|im_start|>user\n"
    "Ci sono tre scatole: Rossa, Blu e Verde.\n"
    "1. Metto una mela nella scatola Rossa.\n"
    "2. Sposto il contenuto della scatola Rossa nella scatola Blu.\n"
    "3. Scambio le etichette delle scatole Blu e Verde (quella che si chiamava Blu ora si chiama Verde e viceversa).\n"
    "4. Prendo l'oggetto che si trova ora nella scatola chiamata 'Blu' e lo metto nella scatola Rossa.\n"
    "Dove si trova la mela adesso? Spiega il ragionamento passo per passo.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
ids2 = tokenizer(p2, return_tensors="pt")["input_ids"]
print(f"--- TEST 2: PUZZLE DELLE 3 SCATOLE (Lunghezza prompt: {ids2.shape[1]} token) ---", flush=True)

t0_g2 = time.perf_counter()
out2 = model.generate(ids2, max_new_tokens=70, temperature=0.0, top_k=1, token_callback=print_tok)
print(f"\n[Test 2 completato in {time.perf_counter() - t0_g2:.2f} s]\n", flush=True)
