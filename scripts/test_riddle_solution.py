import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
torch.set_num_threads(6)
from transformers import AutoModelForCausalLM, AutoTokenizer
from synaptic_fold.converter import NeurosynapticConverter

def solve_riddles():
    model_path = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
    print(f"Loading model from {model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map="cpu",
        local_files_only=True,
    )
    model.eval()

    # Convert to Neurosynaptic Sparse Architecture
    print("Converting to Neurosynaptic Sparse Architecture (active_ratio=0.35, gateways protected)...", flush=True)
    model = NeurosynapticConverter.convert_model_to_sparse(
        model,
        active_ratio=0.35,
        routing_mode="exact_topk",
        protect_gateways=True,
        num_gateway_layers=2
    )

    riddles = [
        (
            "Indovinello Classico (Contadino, Lupo, Capra, Cavolo)",
            "Un contadino deve attraversare un fiume con un lupo, una capra e un cavolo. La barca può portare solo lui e uno solo dei tre alla volta. Se lasciati soli, il lupo mangia la capra e la capra mangia il cavolo. Come fa il contadino a trasportare tutti sull'altra riva sani e salvi? Descrivi tutti i passaggi."
        ),
        (
            "Indovinello Variante (Lupo legato all'albero)",
            "Un contadino deve attraversare un fiume con un cavolo, una pecora e un lupo. La sua barca può portare solo lui e un oggetto alla volta. Se lasciati soli, la pecora mangia il cavolo e il lupo mangia la pecora. Tuttavia, in questa versione il contadino ha una corda con cui ha legato saldamente il lupo a un albero sulla riva iniziale, rendendolo inoffensivo finché resta lì. Descrivi la sequenza di viaggi più rapida per portare tutto dall'altra parte."
        )
    ]

    for title, prompt in riddles:
        print("\n" + "=" * 80, flush=True)
        print(f"DOMANDA: {title}", flush=True)
        print(f"PROMPT: {prompt}", flush=True)
        print("-" * 80, flush=True)

        msgs = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(formatted, return_tensors="pt")

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=220,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        dt = time.perf_counter() - t0
        gen_tokens = out[0][len(inp["input_ids"][0]):]
        answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        speed = len(gen_tokens) / max(dt, 1e-4)

        print("RISPOSTA DEL MODELLO NEUROSINTAPTICO:", flush=True)
        print(answer, flush=True)
        print(f"\n[Statistiche: {len(gen_tokens)} token in {dt:.2f}s -> {speed:.2f} tok/s]", flush=True)

if __name__ == "__main__":
    solve_riddles()
