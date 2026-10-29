"""
SynapticFold: Test Ufficiale di Generazione con Motore Corticale Continuo (Non-Layer).
Valuta il testo prodotto dal Campo Sinaptico Unificato senza pipeline a strati.
"""

import os
import sys
import time
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.continuous_cortical_engine import ContinuousCorticalFieldEngine

def run_continuous_test():
    torch.set_num_threads(8)
    print("=" * 80, flush=True)
    print("🧠 SYNAPTICFOLD: TEST GENERAZIONE CAMPO CORTICALE CONTINUO (NON-LAYER)", flush=True)
    print("   Zero Strati Discreti - Unico Campo Sinaptico Unificato con Attractor Neural ODE", flush=True)
    print("=" * 80, flush=True)

    model_path = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
    print(f"\n[1/3] Caricamento modello base da: {model_path} ...", flush=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map="cpu",
        local_files_only=True,
    )
    hf_model.eval()

    print("\n[2/3] Fusione nel Campo Sinaptico Continuo (Eliminazione dei Layer)...", flush=True)
    t_conv0 = time.perf_counter()
    brain = ContinuousCorticalFieldEngine.from_huggingface_model(
        hf_model=hf_model,
        thought_steps=3,  # 3 passi di integrazione continua dell'ODE
        ode_dt=0.30,
    )
    brain.eval()
    t_conv = time.perf_counter() - t_conv0
    print(f"  -> Campo Corticale inizializzato in {t_conv:.2f}s! Nessun layer residuo nel forward pass.", flush=True)

    # Libera memoria del modello HF originale
    del hf_model
    import gc
    gc.collect()

    test_prompts = [
        "Ciao se io mi chiamo Umberto, tu come ti chiami?",
        "Qual è la capitale d'Italia e perché è famosa?",
        "Un contadino deve attraversare un fiume con un lupo, una capra e un cavolo. Come fa?"
    ]

    print("\n[3/3] Esecuzione Generazione Testuale nel Tempo Continuo...", flush=True)

    for idx, prompt in enumerate(test_prompts, 1):
        print("\n" + "=" * 80, flush=True)
        print(f"TEST {idx}: {prompt}", flush=True)
        print("-" * 80, flush=True)

        msgs = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(formatted, return_tensors="pt")
        input_ids = inp["input_ids"]

        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = brain.generate(
                input_ids=input_ids,
                max_new_tokens=60,
                eos_token_id=tokenizer.eos_token_id,
            )
        dt = time.perf_counter() - t0

        gen_tokens = out_ids[0][input_ids.shape[1]:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        speed = len(gen_tokens) / max(dt, 1e-4)

        print(f"RISPOSTA DEL CAMPO CONTINUO (NON-LAYER):\n\"{text}\"", flush=True)
        print(f"\n[Statistiche: {len(gen_tokens)} token generati in {dt:.2f}s -> {speed:.2f} tok/s]", flush=True)

if __name__ == "__main__":
    run_continuous_test()
