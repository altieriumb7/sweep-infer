"""
Generate complete, untruncated Python code and explanation with Neurosynaptic 7B.
"""

import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.converter import NeurosynapticConverter

SNAPSHOT_7B = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"

def main():
    torch.set_num_threads(8) # 8 physical cores on Ryzen 7 7730U to avoid cache thrashing
    
    print("[1/3] Caricamento Tokenizer e Qwen2.5-7B...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_7B, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT_7B,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        local_files_only=True,
    )
    model.eval()

    print("[2/3] Conversione in NeurosynapticSparseMLP (35% neuroni attivi)...", flush=True)
    model = NeurosynapticConverter.convert_model_to_sparse(model, active_ratio=0.35, routing_mode="exact_topk")

    prompt = "Scrivi la funzione Python completa e funzionante per calcolare il fattoriale con memoization. Fornisci esclusivamente il blocco di codice python."
    msgs = [{"role": "user", "content": prompt}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(fmt, return_tensors="pt")

    print(f"\n[3/3] Generazione Codice Completo (senza troncamento)...", flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(
            **inp,
            max_new_tokens=70,
            temperature=0.3, # lower temperature for deterministic clean code
            do_sample=True,
        )
    dt = time.perf_counter() - t0
    gen_toks = out_ids[0, inp["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()

    print("=" * 70, flush=True)
    print("OUTPUT GENERATO DAL MODELLO NEUROSINAPTICO 7B:", flush=True)
    print("=" * 70, flush=True)
    print(text, flush=True)
    print("=" * 70, flush=True)
    print(f"Token generati: {len(gen_toks)} in {dt:.2f}s ({len(gen_toks)/dt:.2f} tok/s)", flush=True)

    # Save to file
    out_file = os.path.join(ROOT_DIR, "synaptic_fold", "data", "complete_generated_code.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Salvato in: {out_file}", flush=True)


if __name__ == "__main__":
    main()
