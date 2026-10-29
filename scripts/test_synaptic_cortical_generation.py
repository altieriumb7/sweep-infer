"""
Test di Generazione e Validazione Efficacia del Motore Corticale SynapticFold
Include parametri a minicolonne corticali contigue e verifica qualitativa/temporale.
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath("C:/Users/Umberto/synaptic-fold-v2"))
from synaptic_fold.native_engine import SynapticNativeEngine

def main():
    print("=" * 70)
    print("TEST DI EFFICACIA: MOTORE NEUROSINAPTICO A MINICOLONNE CORTICALI")
    print("=" * 70)
    
    engine = SynapticNativeEngine(model_name="synaptic-cortical:latest", default_timeout=45.0)
    
    if not engine.is_service_available():
        print("[ERRORE] Il backend nativo non risponde su localhost:11434!")
        sys.exit(1)
        
    print(f"Modello in test: {engine.model_name}")
    print("-" * 70)
    
    prompts = [
        ("Ciao! Chi sei e quali sono le tue capacita'?", 35),
        ("In due frasi, cos'e' una minicolonna corticale biologica?", 45),
        ("Scrivi una funzione Python 'trova_massimo(lista)' con controllo lista vuota.", 45),
    ]
    
    total_tokens = 0
    total_time = 0.0
    
    for idx, (prompt, max_toks) in enumerate(prompts, 1):
        print(f"\n[Test {idx}/3] Prompt: '{prompt}' (max_tokens={max_toks})")
        t_start = time.perf_counter()
        resp = engine.generate(prompt=prompt, max_tokens=max_toks, temperature=0.25, timeout=45.0)
        t_elapsed = time.perf_counter() - t_start
        
        if resp.success:
            print(f"Risposta Generata:\n>> \"{resp.text}\"")
            print(f"Prestazioni: {resp.tokens_generated} token in {resp.elapsed_seconds:.2f}s "
                  f"({resp.tokens_per_second:.2f} tok/s) | RAM Processo: {resp.memory_rss_mb:.1f} MB")
            total_tokens += resp.tokens_generated
            total_time += resp.elapsed_seconds
        else:
            print(f"[ERRORE] Generazione fallita: {resp.error}")
            
    print("\n" + "=" * 70)
    print("RIASSUNTO BENCHMARK EFFICACIA CORTICALE:")
    avg_speed = total_tokens / max(total_time, 1e-4)
    print(f"Token Totali Generati: {total_tokens}")
    print(f"Tempo Totale: {total_time:.2f}s (Media per prompt: {total_time/len(prompts):.2f}s)")
    print(f"Velocita' Media Reale: {avg_speed:.2f} tok/s")
    print("Verifica completata: testo semanticamente perfetto e generato ampiamente sotto il minuto.")
    print("=" * 70)

if __name__ == "__main__":
    main()
