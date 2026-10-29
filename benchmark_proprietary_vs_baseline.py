import subprocess
import time
import json
import urllib.request
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from synaptic_fold.synaptic_wavefront_engine import ProprietaryWavefrontEngine

def check_server_health(port=8089):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return json.loads(resp.read().decode('utf-8')).get('status') == 'ok'
    except Exception:
        return False

def run_tests():
    print("=" * 70)
    print("BENCHMARK PROPRIETARIO: SYNAPTIC RESONANT WAVEFRONT ENGINE (8B PURO)")
    print("Processore: AMD Ryzen 7 7730U (8 Core Zen 3) | Target: LLaMA-3-8B (4.66 GB)")
    print("=" * 70)

    if not check_server_health():
        print("[ERRORE] Il server 8B non e raggiungibile sulla porta 8089.")
        return

    engine = ProprietaryWavefrontEngine(server_url="http://127.0.0.1:8089")

    test_prompts = [
        (
            "Indovinello Logico: Fiume",
            (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "Sei un assistente cognitivo in lingua italiana. Quando ti viene posto un problema di logica, "
                "spiega la sequenza passo dopo passo in modo chiaro ed esaustivo. Nella traversata del fiume, "
                "il contadino ha con se una barca che puo trasportare solo lui e un elemento per volta (lupo, capra o cavolo). "
                "Il lupo non puo rimanere solo con la capra, e la capra non puo rimanere sola con il cavolo.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Come fa il contadino a portare tutti sull'altra riva senza perdite?<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ),
        (
            "Identita e Dialogo",
            (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "Sei un assistente cognitivo italiano che risponde con precisione e calore naturale.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Ciao! Se io mi chiamo Umberto e lavoro a questo progetto, tu chi sei e come puoi aiutarmi oggi?<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        ),
        (
            "Scienza Cognitiva",
            (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "Sei un neuroscienziato computazionale che spiega concetti complessi in modo sintetico e denso.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Spiegami in due punti cardine perche la memoria biologica e basata su attrattori dinamici e non su indirizzamento von Neumann.<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        )
    ]

    results = []

    for name, prompt in test_prompts:
        print(f"\n>>> ESECUZIONE TEST: {name} ...")
        t0 = time.time()
        res = engine.generate(prompt, max_tokens=150)
        dt = time.time() - t0

        tokens = res['tokens_predicted']
        speed = res['effective_speed_tok_s']
        pred_speed = res['predicted_speed_tok_s']
        prompt_speed = res['prompt_speed_tok_s']
        acceptance = res['draft_acceptance'] * 100.0

        print(f"  Token generati: {tokens} in {res['total_time_sec']:.2f} s")
        print(f"  Velocita effettiva: {speed:.2f} tok/s | Velocita fase decodifica: {pred_speed:.2f} tok/s")
        print(f"  Velocita prompt eval (GEMM 32L): {prompt_speed:.2f} tok/s")
        print(f"  Tasso accettazione bozze: {acceptance:.1f}%")
        print(f"  Testo generato:\n  ---\n  {res['text'][:200]}...\n  ---")

        results.append({
            'name': name,
            'tokens': tokens,
            'time_sec': res['total_time_sec'],
            'speed_tok_s': speed,
            'predicted_speed': pred_speed,
            'prompt_speed': prompt_speed,
            'acceptance': acceptance
        })

    print("\n" + "=" * 70)
    print("RIEPILOGO COMPARATIVO SUL MODELLO 8B:")
    print("=" * 70)
    print(f"{'Test':<30} | {'Throughput':<12} | {'Decodifica':<12} | {'Prompt GEMM':<12} | {'Accettaz.'}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<30} | {r['speed_tok_s']:.2f} tok/s    | {r['predicted_speed']:.2f} tok/s    | {r['prompt_speed']:.2f} tok/s    | {r['acceptance']:.1f}%")
    print("=" * 70)

if __name__ == '__main__':
    run_tests()
