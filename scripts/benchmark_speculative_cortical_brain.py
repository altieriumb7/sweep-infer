import os
import sys
import time
import json
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def query_brain(prompt: str, max_tokens: int = 150):
    url = "http://127.0.0.1:8089/completion"
    payload = {
        "prompt": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nSei un assistente logico, brillante, cordiale ed esperto. Rispondi in perfetto italiano in modo chiaro e conciso.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "n_predict": max_tokens,
        "temperature": 0.1,
        "stop": ["<|eot_id|>", "<|start_header_id|>"],
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
    tok_per_sec = result.get("timings", {}).get("predicted_per_second", tok_count / max(dt, 1e-4))
    draft_stats = result.get("timings", {})
    
    return text, tok_count, dt, tok_per_sec, draft_stats

def main():
    print("=" * 80)
    print("🧠 BENCHMARK ARCHITETTURA CORTICALE SPECULATIVA (8B NEOCORTEX + 1B TALAMO)")
    print("   Zero Pruning | Modello 8B Intatto al 100% | Dinamica Predittiva Multi-Token")
    print("=" * 80)
    
    prompts = [
        "Ciao se io mi chiamo Umberto, tu come ti chiami?",
        "Qual è la capitale d'Italia e perché è celebre nel mondo?",
        "Un contadino deve attraversare un fiume con un cavolo, una capra e un lupo. Ha una barca che può trasportare solo lui e un solo elemento alla volta. Se il lupo resta solo con la capra, la mangia. Se la capra resta sola con il cavolo, lo mangia. Come fa il contadino a portare tutti sull'altra riva sani e salvi? Spiega i passaggi."
    ]
    
    for i, p in enumerate(prompts, 1):
        print(f"\n--- [TEST {i}] Prompt: \"{p[:60]}...\" ---", flush=True)
        text, n_tok, dt, tps, stats = query_brain(p, max_tokens=150)
        print(f"Risposta Generata:\n\"{text}\"", flush=True)
        print(f"⚡ [Velocita: {tps:.2f} tok/s | Token: {n_tok} | Tempo: {dt:.2f}s]", flush=True)
        if "draft_accept_rate" in stats or "predicted_n" in stats:
            print(f"   Timings dettaglio: {stats}", flush=True)

if __name__ == "__main__":
    main()
