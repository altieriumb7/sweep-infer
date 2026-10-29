import os
import sys
import time
import psutil
import torch
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = r"C:\Users\Umberto\synaptic-fold-v2"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.synaptic_full28_model import SynapticFull28Model

snapshot_dir = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)

print("=" * 80)
print("TEST LIVE: SYNAPTIC FULL 28-LAYER 7B SUI TEST DELL'UTENTE")
print("Target: Correttezza logica del ragionamento passo per passo")
print("=" * 80, flush=True)

t0 = time.perf_counter()
model = SynapticFull28Model.from_pretrained(snapshot_dir)
print(f"Modello 7B quantizzato e caricato in {time.perf_counter() - t0:.2f} s", flush=True)

# Test 1: Sanity greeting
p1 = "<|im_start|>user\nCiao! Chi sei?<|im_end|>\n<|im_start|>assistant\n"
ids1 = tokenizer(p1, return_tensors="pt")["input_ids"]

print("\n--- TEST 1: SALUTO (7B SYNAPTIC) ---", flush=True)
def print_tok(t):
    print(tokenizer.decode([t], skip_special_tokens=True), end="", flush=True)

t0_g = time.perf_counter()
out1 = model.generate(ids1, max_new_tokens=20, temperature=0.0, top_k=1, token_callback=print_tok)
print(f"\n[Completato in {time.perf_counter() - t0_g:.2f} s]\n", flush=True)

# Test 2: Boxes Puzzle (Concise prompt for fast prefill)
p2 = (
    "<|im_start|>user\n"
    "Ci sono tre scatole: Rossa, Blu, Verde.\n"
    "1. Metto una mela nella scatola Rossa.\n"
    "2. Sposto la mela dalla scatola Rossa alla scatola Blu.\n"
    "3. Scambio le etichette delle scatole Blu e Verde (quella che si chiamava Blu ora si chiama Verde e viceversa).\n"
    "4. Prendo l'oggetto che si trova ora nella scatola chiamata 'Blu' e lo metto nella scatola Rossa.\n"
    "Dove si trova la mela adesso? Spiega il ragionamento passo per passo.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
ids2 = tokenizer(p2, return_tensors="pt")["input_ids"]
print(f"--- TEST 2: PUZZLE DELLE 3 SCATOLE (Lunghezza prompt: {ids2.shape[1]} token) ---", flush=True)

t0_g2 = time.perf_counter()
out2 = model.generate(ids2, max_new_tokens=70, temperature=0.0, top_k=1, token_callback=print_tok)
print(f"\n[Completato in {time.perf_counter() - t0_g2:.2f} s]\n", flush=True)
