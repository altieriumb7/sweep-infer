import os
import sys
import time
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SNAPSHOT_7B = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
SNAPSHOT_05B = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"


def main():
    print("=" * 75)
    print("🧠 SYNAPTICFOLD v2: TERMINAL CHAT & TESTING INTERFACE")
    print("=" * 75)
    print("Caricamento architettura verificata:")
    print("  ⭐ 🧠 Synaptic Dual-Brain (Talamo Riflesso + Neocorteccia) [ATTIVO]")
    print("=" * 75)

    from synaptic_fold.synaptic_dual_brain import SynapticDualBrain
    brain = SynapticDualBrain(thalamus_path=SNAPSHOT_05B, neocortex_path=SNAPSHOT_7B)
    tokenizer = brain.tokenizer
    is_dual = True
    is_hf = False

    rss = psutil.Process().memory_info().rss / (1024**2)
    print(f"\n✅ Modello pronto in memoria! RAM Processo: {rss:.1f} MB ({rss/1024:.2f} GB)")
    print("Digita il tuo messaggio (o 'exit' per uscire, 'preset1', 'preset2', 'preset3'):\n")

    history = []

    while True:
        try:
            user_msg = input("\n👤 Tu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nChiusura sessione.")
            break

        if not user_msg:
            continue
        if user_msg.lower() in ("exit", "quit"):
            print("Arrivederci!")
            break

        if user_msg == "preset1":
            user_msg = "Ciao! Chi sei e qual è il tuo ruolo?"
            print(f"👤 Tu [Preset 1]: {user_msg}")
        elif user_msg == "preset2":
            user_msg = (
                "Ci sono tre scatole: Rossa, Blu e Verde.\n"
                "1. Metto una mela nella scatola Rossa.\n"
                "2. Sposto il contenuto della scatola Rossa nella scatola Blu.\n"
                "3. Scambio le etichette delle scatole Blu e Verde (quella che si chiamava Blu ora si chiama Verde e viceversa).\n"
                "4. Prendo l'oggetto che si trova ora nella scatola chiamata 'Blu' e lo metto nella scatola Rossa.\n"
                "Dove si trova la mela adesso? Spiega il ragionamento passo per passo."
            )
            print(f"👤 Tu [Preset 2]:\n{user_msg}")
        elif user_msg == "preset3":
            user_msg = (
                "Un contadino deve attraversare un fiume con un cavolo, una pecora e un lupo. "
                "La sua barca può portare solo lui e un oggetto alla volta. "
                "Tuttavia, il contadino ha una corda con cui ha legato saldamente il lupo a un albero sulla riva iniziale, "
                "rendendolo inoffensivo finché resta lì. "
                "Descrivi la sequenza di viaggi più rapida per portare tutto dall'altra parte."
            )
            print(f"👤 Tu [Preset 3]:\n{user_msg}")

        history.append({"role": "user", "content": user_msg})
        prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

        print("🤖 Modello: ", end="", flush=True)

        tok_count = [0]
        t0 = time.perf_counter()
        ttft = [None]

        def on_token(tok_id: int):
            if ttft[0] is None:
                ttft[0] = time.perf_counter() - t0
            tok_count[0] += 1
            print(tokenizer.decode([tok_id], skip_special_tokens=True), end="", flush=True)

        try:
            with torch.no_grad():
                if is_dual:
                    for chunk in brain.generate_stream(user_msg, max_new_tokens=100, temperature=0.1, repetition_penalty=1.15):
                        if ttft[0] is None and chunk:
                            ttft[0] = time.perf_counter() - t0
                        tok_count[0] += 1
                        print(chunk, end="", flush=True)
                elif is_hf:
                    out = model.generate(
                        input_ids,
                        max_new_tokens=80,
                        temperature=0.1,
                        do_sample=True,
                        repetition_penalty=1.15,
                    )
                    gen_toks = out[0][input_ids.shape[1]:]
                    tok_count[0] = len(gen_toks)
                    print(tokenizer.decode(gen_toks, skip_special_tokens=True), end="", flush=True)
                else:
                    out = model.generate(
                        input_ids,
                        max_new_tokens=80,
                        temperature=0.0,
                        token_callback=on_token,
                    )
        except Exception as e:
            print(f"\n[Errore generazione: {e}]")

        total_t = time.perf_counter() - t0
        speed = tok_count[0] / max(total_t, 1e-4)
        print(f"\n[⚡ {tok_count[0]} token generati in {total_t:.2f} s ({speed:.2f} tok/s)]")


if __name__ == "__main__":
    main()
