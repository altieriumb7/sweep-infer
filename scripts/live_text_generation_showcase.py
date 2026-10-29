import os
import sys
import time
import json
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def query_brain(prompt: str, max_tokens: int = 250):
    url = "http://127.0.0.1:8089/completion"
    payload = {
        "prompt": f"<|im_start|>system\nSei un assistente scientifico e filosofico intelligente, brillante e colto. Rispondi in perfetto italiano in modo approfondito, elegante e chiaro.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "n_predict": max_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "stream": False
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    
    text = result.get("content", "").strip()
    tok_count = result.get("tokens_predicted", 0)
    timings = result.get("timings", {})
    tps = timings.get("predicted_per_second", tok_count / max(dt, 1e-4))
    draft_acc = timings.get("draft_n_accepted", 0) / max(timings.get("draft_n", 1), 1)
    
    return text, tok_count, dt, tps, draft_acc, timings

def run_showcase():
    print("=" * 80, flush=True)
    print("🧠 SYNAPTIC BRAIN: DIMOSTRAZIONE DI GENERAZIONE TESTUALE IN TEMPO REALE", flush=True)
    print("   Architettura: Predictive Coding Talamo-Corticale (Zero Pruning)", flush=True)
    print("   Hardware: CPU AMD Ryzen (8 Thread) | Velocità Target: >10 tok/s", flush=True)
    print("=" * 80, flush=True)
    
    prompts = [
        (
            "Indovinello di Logica",
            "Un contadino deve attraversare un fiume con una barca portando con sé un lupo, una capra e un cavolo. La barca può trasportare solo lui e un solo elemento alla volta. Se lasciati soli, il lupo mangia la capra e la capra mangia il cavolo. Qual è la sequenza esatta di tutti i viaggi per portarli tutti sull'altra riva sani e salvi? Spiega ogni passaggio in modo rigoroso."
        ),
        (
            "Neuroscienze e Intelligenza Artificiale",
            "Spiega in modo chiaro e divulgativo come funziona la memoria di lavoro nel cervello umano e quali sono le differenze rispetto alla memoria delle reti neurali artificiali."
        ),
        (
            "Dialogo Filosofico",
            "Scrivi un breve dialogo vivace e arguto tra due scienziati, Leonardo e Sofia, che discutono se le reti neurali comprendono veramente i concetti o se sono soltanto complesse matrici statistiche."
        )
    ]
    
    results = []
    
    for idx, (title, prompt) in enumerate(prompts, 1):
        print(f"\n" + "#" * 80, flush=True)
        print(f"[{idx}/3] TEST: {title}", flush=True)
        print(f"Prompt: \"{prompt}\"", flush=True)
        print("-" * 80, flush=True)
        
        text, n_tok, dt, tps, d_acc, timings = query_brain(prompt, max_tokens=220)
        
        print("RISPOSTA GENERATA DAL CERVELLO:\n", flush=True)
        print(text, flush=True)
        print("\n" + "-" * 80, flush=True)
        print(f"⚡ METRICHE HARDWARE CPU:", flush=True)
        print(f"   • Velocità di Decodifica: {tps:.2f} token/secondo", flush=True)
        print(f"   • Token Generati:         {n_tok} token", flush=True)
        print(f"   • Latenza Totale:         {dt:.2f} secondi", flush=True)
        print(f"   • Tasso Accettazione:     {d_acc*100:.1f}%", flush=True)
        print("#" * 80, flush=True)
        
        results.append({
            "title": title,
            "prompt": prompt,
            "text": text,
            "tokens": n_tok,
            "time_sec": dt,
            "tokens_per_sec": tps,
            "draft_acceptance": d_acc
        })
    
    out_file = r"C:\Users\Umberto\synaptic-fold-v2\synaptic_fold\data\live_generation_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Risultati completi esportati in: {out_file}", flush=True)

if __name__ == "__main__":
    run_showcase()
