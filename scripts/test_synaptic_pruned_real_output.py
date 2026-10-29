"""
Test di Generazione e Fluenza sul Modello Fisicamente Potato: synaptic-pruned-7b
Verifica velocità (tok/s), memoria e qualità del testo generato dal modello potato al 50%.
"""
import sys
import os
import time
import json
import urllib.request

def run_test(prompt: str, max_tokens: int = 50):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "synaptic-pruned-7b:latest",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.2,
            "num_thread": 8
        }
    }
    
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=50.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        dt = time.perf_counter() - t0
        text = data.get("response", "").strip()
        eval_count = data.get("eval_count", 0)
        eval_dur = data.get("eval_duration", 1) / 1e9
        speed = eval_count / max(eval_dur, 1e-4)
        return text, eval_count, dt, speed, True
    except Exception as e:
        return f"ERRORE: {e}", 0, time.perf_counter() - t0, 0.0, False

def main():
    print("=" * 75)
    print("OUTPUT DEL MODELLO CORTICALE PRUNED: synaptic-pruned-7b (3.1 GB)")
    print("=" * 75)
    
    prompts = [
        ("Ciao! Chi sei?", 30),
        ("Spiega in due frasi chiare la differenza principale tra un neurone biologico e un neurone artificiale.", 55),
        ("Scrivi una funzione Python 'somma(a, b)' che restituisce la somma di a e b.", 40),
    ]
    
    for idx, (p, max_toks) in enumerate(prompts, 1):
        print(f"\n[Test {idx}/3] Prompt: \"{p}\"")
        text, count, dt, speed, ok = run_test(p, max_tokens=max_toks)
        print(f"Risultato: {count} token in {dt:.2f}s | Velocità: {speed:.2f} tok/s")
        print(f"TESTO GENERATO:\n>> \"{text}\"")
        print("-" * 75)

if __name__ == "__main__":
    main()
