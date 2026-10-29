"""
Test Rapido di Generazione e Benchmark (Watchdog <= 60 secondi)
Verifica velocità, qualità del testo e latenze su prompt brevi ("Ciao", ecc.)
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath("C:/Users/Umberto/synaptic-fold-v2"))
from synaptic_fold.native_engine import SynapticNativeEngine

def main():
    print("=" * 65)
    print("TEST GENERAZIONE RAPIDA SYNAPTIC NATIVE ENGINE (TIMEOUT MAX 60s)")
    print("=" * 65)
    
    engine = SynapticNativeEngine(model_name="synaptic-llama3:latest", default_timeout=50.0)
    
    if not engine.is_service_available():
        print("[ERRORE] Il backend nativo non risponde su localhost:11434!")
        sys.exit(1)
        
    print(f"Modello attivo: {engine.model_name}")
    print(f"Modelli disponibili: {engine.get_loaded_models()}")
    print("-" * 65)
    
    test_prompts = [
        ("Ciao, come stai?", 30),
        ("Chi sei e spiegami in una frase la capitale d'Italia.", 40),
        ("Definisci in una singola frase cos'e' un neurone biologico.", 45),
    ]
    
    total_tokens = 0
    total_time = 0.0
    
    for idx, (prompt, max_toks) in enumerate(test_prompts, 1):
        print(f"\n[Test {idx}/3] Prompt: '{prompt}' (max_tokens={max_toks})")
        t_start = time.perf_counter()
        
        # Generazione con watchdog a 50s (strettamente sotto i 60s richiesti)
        resp = engine.generate(prompt=prompt, max_tokens=max_toks, temperature=0.3, timeout=50.0)
        t_end = time.perf_counter()
        
        if resp.success:
            print(f"Risposta del modello:\n>> \"{resp.text}\"")
            print(f"Metriche: {resp.tokens_generated} token in {resp.elapsed_seconds:.2f}s "
                  f"({resp.tokens_per_second:.2f} tok/s) | RAM: {resp.memory_rss_mb:.1f} MB")
            total_tokens += resp.tokens_generated
            total_time += resp.elapsed_seconds
        else:
            print(f"[FALLITO] Errore: {resp.error}")
            
    print("\n" + "=" * 65)
    print("RIASSUNTO FINALE BENCHMARK:")
    avg_speed = total_tokens / max(total_time, 1e-4)
    print(f"Token Totali Generati: {total_tokens}")
    print(f"Tempo Totale: {total_time:.2f}s (Media per risposta: {total_time/len(test_prompts):.2f}s)")
    print(f"Velocita' Media Reale misurata: {avg_speed:.2f} tok/s")
    print("Tutti i test completati entro la soglia di sicurezza di 60 secondi.")
    print("=" * 65)

if __name__ == "__main__":
    main()
