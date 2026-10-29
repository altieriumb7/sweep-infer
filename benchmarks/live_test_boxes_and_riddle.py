import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = r"C:\Users\Umberto\synaptic-fold-v2"
SNAPSHOT_05B = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"

print("=" * 80)
print("TEST LIVE GENERAZIONE AUTONOMA: QWEN2.5 REFLEX BRAIN (BFLOAT16)")
print("Hardware: AMD Ryzen 7 7730U | RAM: < 700 MB | Zero Quantization Noise")
print("=" * 80, flush=True)

tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_05B, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    SNAPSHOT_05B,
    dtype=torch.bfloat16,
    device_map="cpu",
    local_files_only=True,
)

# Test 1: Sanity
p1 = "Ciao! Chi sei e qual è il tuo ruolo?"
# Test 2: Boxes Puzzle (User Explicit Test)
p2 = (
    "Ci sono tre scatole: Rossa, Blu e Verde.\n"
    "Metto una mela nella scatola Rossa.\n"
    "Sposto il contenuto della scatola Rossa nella scatola Blu.\n"
    "Scambio le etichette delle scatole Blu e Verde (quella che si chiamava Blu ora si chiama Verde e viceversa).\n"
    "Prendo l'oggetto che si trova ora nella scatola chiamata 'Blu' e lo metto nella scatola Rossa.\n"
    "Dove si trova la mela adesso? Spiega il ragionamento passo per passo."
)
# Test 3: River Riddle
p3 = (
    "Un contadino deve attraversare un fiume con un cavolo, una pecora e un lupo. "
    "La sua barca può portare solo lui e un oggetto alla volta. "
    "Se lasciati soli, la pecora mangia il cavolo e il lupo mangia la pecora. "
    "Tuttavia, in questa versione il contadino ha una corda con cui ha legato saldamente il lupo a un albero sulla riva iniziale, "
    "rendendolo inoffensivo finché resta lì. "
    "Descrivi la sequenza di viaggi più rapida per portare tutto dall'altra parte."
)

tests = [
    ("TIER 1: SANITY GREETING", p1, 40),
    ("TIER 2: THE THREE BOXES PUZZLE", p2, 120),
    ("TIER 3: THE TIED WOLF RIDDLE", p3, 120),
]

for title, prompt_text, max_tok in tests:
    print(f"\n{'='*70}\n>>> {title}\n{'='*70}", flush=True)
    msgs = [{"role": "user", "content": prompt_text}]
    fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(fmt, return_tensors="pt")["input_ids"]
    
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_tok,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
        )
    dt = time.perf_counter() - t0
    gen_toks = out[0][input_ids.shape[1]:]
    tok_s = len(gen_toks) / max(dt, 1e-4)
    response_str = tokenizer.decode(gen_toks, skip_special_tokens=True)
    
    print(f"[PREFILL + DECODE: {len(gen_toks)} tokens in {dt:.2f} s ({tok_s:.2f} tok/s)]")
    print("\n--- RISPOSTA GENERATA DAL MODELLO ---")
    print(response_str)
    print("--- FINE RISPOSTA ---\n", flush=True)

