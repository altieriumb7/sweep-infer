"""
Full Verification Suite: Proprietary Wavefront Engine on 3 Diverse Domains
Target: Meta-Llama-3-8B-Instruct ONLY (No small models, Zero Pruning)
Evaluates:
  1. River Riddle (Logica e Consequenzialita)
  2. Philosophical Dialog (Coscienza e Semantica)
  3. Neurobiology (Plasticita Sinaptica ed Encefalo)
"""

import urllib.request
import json
import time
import sys
import subprocess
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SERVER_EXE = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"
MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa"
PORT = 8094

DOMAINS = [
    {
        "domain": "1. Logica Formale (Indovinello del Fiume)",
        "prompt": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Sei un assistente logico-cognitivo in lingua italiana di assoluta eccellenza. "
            "Descrivi con chiarezza e precisione la sequenza formale completa per risolvere "
            "l'indovinello del contadino, della capra, del lupo e del cavolo da portare sull'altra riva.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Quali sono tutti i singoli passaggi della traversata senza lasciare soli lupo-capra o capra-cavolo?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "tokens": 150
    },
    {
        "domain": "2. Dialogo Filosofico (Mente e Macchina)",
        "prompt": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Sei un filosofo della mente e linguista teorico in lingua italiana. "
            "Analizza con profondità argomentativa il rapporto tra computazione sinaptica e comprensione semantica.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Un modello linguistico che manipola simboli sintattici possiede una reale comprensione semantica del significato?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "tokens": 150
    },
    {
        "domain": "3. Neuroscienze Cognitive (Plasticità Sinaptica)",
        "prompt": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Sei un neuroscienziato computazionale. Spiega in modo rigoroso, chiaro e scientificamente impeccabile "
            "il principio della plasticità hebbiana e il potenziamento a lungo termine (LTP) nella corteccia cerebrale.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Come avviene il potenziamento a lungo termine a livello dei recettori NMDA e delle sinapsi glutammatergiche?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "tokens": 150
    }
]

def wait_for_server(timeout=25):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
            with urllib.request.urlopen(req, timeout=1) as resp:
                if json.loads(resp.read().decode('utf-8')).get('status') == 'ok':
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def run_suite():
    print("=" * 80)
    print("SUITE COMPLETA DI VERIFICA TESTUALE E VELOCITÀ SU MODELLO 8B (LLAMA-3-8B)")
    print("Target Unico: Meta-Llama-3-8B-Instruct (4.66 GB) su CPU Ryzen 7 7730U")
    print("Wavefront Speculativo: K=4, n=2, m=4, hits=1")
    print("=" * 80)

    cmd = [
        SERVER_EXE,
        "-m", MODEL_BLOB,
        "--threads", "8",
        "--port", str(PORT),
        "-c", "2048",
        "-b", "512",
        "-ub", "512",
        "--spec-type", "ngram-map-k4v",
        "--spec-ngram-map-k4v-size-n", "2",
        "--spec-ngram-map-k4v-size-m", "4",
        "--spec-ngram-map-k4v-min-hits", "1"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_for_server():
            print("[ERRORE] Timeout avvio server.")
            return

        results = []
        for d in DOMAINS:
            print(f"\n--- ESECUZIONE: {d['domain']} ---")
            payload = {
                'prompt': d['prompt'],
                'n_predict': d['tokens'],
                'temperature': 0.0,
                'stream': False
            }

            t0 = time.time()
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion",
                                         data=json.dumps(payload).encode('utf-8'),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as resp:
                res = json.loads(resp.read().decode('utf-8'))
            dt = time.time() - t0

            tokens = res.get('tokens_predicted', 0)
            timings = res.get('timings', {})
            pred_ms = timings.get('predicted_per_token_ms', 0.0)
            pred_speed = timings.get('predicted_per_second', 0.0)
            draft_n = timings.get('draft_n', 0)
            draft_acc = timings.get('draft_n_accepted', 0)
            acc_rate = (draft_acc / draft_n * 100.0) if draft_n > 0 else 0.0
            content = res.get('content', '').strip()

            print(f"\n[TESTO GENERATO DALL'8B]:\n{content}\n")
            print(f"[METRICHE]:")
            print(f"  -> Latenza per token (decodifica): {pred_ms:.1f} ms/token")
            print(f"  -> Velocita pura di generazione: {pred_speed:.2f} token/s")
            print(f"  -> Bozze accettate: {draft_acc}/{draft_n} ({acc_rate:.1f}%)")
            print(f"  -> Tempo totale: {dt:.2f} s per {tokens} token")

            results.append({
                'domain': d['domain'],
                'ms_per_token': pred_ms,
                'tok_per_sec': pred_speed,
                'draft_acc': draft_acc,
                'draft_n': draft_n,
                'acc_rate': acc_rate,
                'text': content
            })

        print("\n" + "=" * 80)
        print("RIASSUNTO SUITE MULTI-DOMINIO (MODELLO 8B):")
        print("=" * 80)
        for r in results:
            print(f"{r['domain']:<42} | {r['ms_per_token']:<6.1f} ms/tok | {r['tok_per_sec']:<6.2f} tok/s | Accettaz: {r['acc_rate']:.1f}%")
        print("=" * 80)

        # Save to benchmark suite file
        suite_path = r"C:\Users\Umberto\synaptic-fold-v2\benchmarks\multidomain_verification_suite.json"
        with open(suite_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[SALVATO] Risultati salvati in: {suite_path}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

if __name__ == '__main__':
    run_suite()
