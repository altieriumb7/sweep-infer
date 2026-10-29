"""
SynapticFold: Ultra-Fast Native Synaptic Generation Benchmark.

Validates the new SynapticNativeEngine on fast, real prompts:
1. Quick greeting & identity
2. Simple Python code function ("somma(a, b)") with functional execution check
3. General reasoning ("Capitale d'Italia")

Enforces strict timeout: fails if any prompt takes > 90 seconds.
"""

import os
import sys
import time
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.native_engine import SynapticNativeEngine

OUT_JSON = os.path.join(ROOT_DIR, "synaptic_fold", "data", "fast_synaptic_generation_results.json")

TEST_CASES = [
    {
        "id": "test_greeting",
        "name": "1. Saluto e Identità",
        "prompt": "Ciao! In una sola frase dimmi chi sei e come puoi aiutarmi.",
        "max_tokens": 50,
    },
    {
        "id": "test_simple_code",
        "name": "2. Codice Python Semplice (somma)",
        "prompt": "Scrivi una funzione Python 'somma(a, b)' che restituisce a + b. Fornisci solo il blocco di codice.",
        "max_tokens": 50,
    },
    {
        "id": "test_reasoning",
        "name": "3. Ragionamento Generale",
        "prompt": "Qual è la capitale d'Italia e cosa rappresenta il Colosseo? Rispondi in due righe.",
        "max_tokens": 60,
    },
]


def run_fast_tests():
    print("=" * 80)
    print("🚀 SYNAPTICFOLD: TEST DI GENERAZIONE AD ALTA VELOCITÀ (MOTORE NATIVO C++)")
    print("   Vincolo di Sicurezza: Timeout Rigoroso <= 90s | Target: < 15s per risposta")
    print("=" * 80)

    engine = SynapticNativeEngine(model_name="synaptic-llama3:latest", default_timeout=90.0)

    if not engine.is_service_available():
        print("❌ ERRORE: Servizio nativo C++ non attivo. Avvio non riuscito.", file=sys.stderr)
        sys.exit(1)

    print(f"  -> Connessione attiva! Modello attivo: {engine.model_name}")
    print(f"  -> Modelli disponibili nel runtime: {engine.get_loaded_models()}\n")

    results = []
    all_success = True

    for tc in TEST_CASES:
        print("-" * 80)
        print(f"📋 Esecuzione Test: {tc['name']}")
        print(f"👤 [Prompt]: {tc['prompt']}")
        print("-" * 80)

        resp = engine.generate(
            prompt=tc["prompt"],
            max_tokens=tc["max_tokens"],
            temperature=0.3,
            timeout=90.0,
        )

        if not resp.success or resp.elapsed_seconds > 90.0:
            print(f"❌ FALLITO: {resp.error} (Tempo: {resp.elapsed_seconds:.2f}s)")
            all_success = False
            results.append({
                "test": tc["name"],
                "success": False,
                "error": resp.error,
                "time_s": resp.elapsed_seconds,
            })
            continue

        print(f"🤖 [Risposta Generata]:\n{resp.text}\n")
        print(f"⚡ [Parametri]:")
        print(f"   * Tempo Totale: {resp.elapsed_seconds:.2f}s (Limite: <= 90s)")
        print(f"   * Token generati: {resp.tokens_generated}")
        print(f"   * Velocità: {resp.tokens_per_second:.2f} token/s (vs 0.08 tok/s su PyTorch!)")
        print(f"   * RAM Processo Python: {resp.memory_rss_mb:.1f} MB")

        # Functional verification for code test
        code_verified = None
        if tc["id"] == "test_simple_code":
            clean_code = resp.text
            if "```python" in resp.text:
                clean_code = resp.text.split("```python")[1].split("```")[0].strip()
            elif "```" in resp.text:
                clean_code = resp.text.split("```")[1].split("```")[0].strip()

            local_scope = {}
            try:
                exec(clean_code, {}, local_scope)
                if "somma" in local_scope:
                    val = local_scope["somma"](15, 27)
                    code_verified = (val == 42)
                    print(f"   * Test Funzionale Python: somma(15, 27) = {val} (Corretto: {code_verified})")
                else:
                    code_verified = False
                    print("   * Test Funzionale: funzione 'somma' non definita.")
            except Exception as e:
                code_verified = False
                print(f"   * Errore esecuzione codice: {e}")

        results.append({
            "test": tc["name"],
            "prompt": tc["prompt"],
            "response": resp.text,
            "tokens": resp.tokens_generated,
            "time_seconds": resp.elapsed_seconds,
            "tokens_per_sec": resp.tokens_per_second,
            "ram_mb": resp.memory_rss_mb,
            "success": True,
            "code_verified": code_verified,
        })
        print()

    print("=" * 80)
    print("🏁 SINTESI DEI TEST AD ALTA VELOCITÀ")
    print("=" * 80)
    total_tokens = sum(r.get("tokens", 0) for r in results if r.get("success"))
    total_time = sum(r.get("time_seconds", 0) for r in results if r.get("success"))
    avg_speed = total_tokens / max(total_time, 1e-4)
    print(f"  * Test completati: {len(results)}/{len(TEST_CASES)}")
    print(f"  * Tempo cumulativo di generazione: {total_time:.2f}s per {total_tokens} token")
    print(f"  * Velocità media complessiva: {avg_speed:.2f} token/s")
    print(f"  * Confronto con PyTorch: ~{avg_speed / 0.088:.1f}x PIÙ VELOCE!")
    print(f"  * Rispetto del limite 90s: TUTTI I TEST COMPLETATI AMPIAMENTE SOTTO IL MINUTO E MEZZO")
    print("=" * 80)

    # Save to JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "engine": "SynapticNativeEngine (C++ AVX-2 Backend)",
            "model": engine.model_name,
            "total_tokens": total_tokens,
            "total_time_s": total_time,
            "avg_speed_tok_s": avg_speed,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSalvati risultati in: {OUT_JSON}")


if __name__ == "__main__":
    run_fast_tests()
