"""
Script per testare le configurazioni di inferenza (Thread scaling, GPU offload, Batch size)
Verifica empirica per scoprire se stiamo omettendo parametri di accelerazione hardware.
"""
import sys
import os
import time
import json
import urllib.request

def test_config(name: str, options: dict, prompt: str = "Ciao, dimmi brevemente chi sei.", max_tokens: int = 30):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3:latest",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.2,
            **options
        }
    }
    
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=40.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        dt = time.perf_counter() - t0
        eval_count = data.get("eval_count", 0)
        eval_dur = data.get("eval_duration", 1) / 1e9
        speed = eval_count / max(eval_dur, 1e-4)
        print(f"[{name:22s}] {eval_count} tok in {dt:.2f}s | Speed: {speed:.2f} tok/s (eval_dur: {eval_dur:.2f}s)")
        return speed
    except Exception as e:
        print(f"[{name:22s}] ERRORE: {e}")
        return 0.0

def main():
    print("=" * 70)
    print("BENCHMARK MULTI-CONFIGURAZIONE: SCALING THREAD E PARAMETRI CPU")
    print("=" * 70)
    
    # Warmup
    test_config("Warmup", {"num_thread": 8}, max_tokens=10)
    
    configs = [
        ("4 Thread", {"num_thread": 4}),
        ("6 Thread", {"num_thread": 6}),
        ("8 Thread (Fisici)", {"num_thread": 8}),
        ("12 Thread", {"num_thread": 12}),
        ("16 Thread (Tutti SMT)", {"num_thread": 16}),
        ("Batch 128 + 8 Thread", {"num_thread": 8, "num_batch": 128}),
        ("Context 512 + 8 Thread", {"num_thread": 8, "num_ctx": 512}),
    ]
    
    results = {}
    for name, opt in configs:
        speed = test_config(name, opt, max_tokens=30)
        results[name] = speed
        
    print("\n" + "=" * 70)
    print("RIEPILOGO SPEEDUP THREAD:")
    for name, sp in results.items():
        print(f"  - {name:25s}: {sp:.2f} tok/s")
    print("=" * 70)

if __name__ == "__main__":
    main()
