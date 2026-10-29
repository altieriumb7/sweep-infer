"""
Synaptic Asymmetric Tree & Assembly Optimization Benchmark
Target: Meta-Llama-3-8B-Instruct (4.66 GB, 100% active weights, Zero Pruning)
Hardware: AMD Ryzen 7 7730U (8C/16T Zen 3, DDR4-3200)

Evaluates:
1. Baseline Sequential (Reference)
2. Wavefront K=4 (Linear Sweet Spot)
3. Synaptic Asymmetric Tree (M <= 4, Budget-Constrained, Tree Branching + Syntactic Grammar Context Priming)
"""

import subprocess
import urllib.request
import json
import time
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SERVER_EXE = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"
MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa"
PORT = 8099

PROMPTS = {
    "A_Logica": {
        "title": "Logica e Deduzione (Indovinello dei Tre Scrigni)",
        "prompt": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Sei un assistente logico brillante e rigoroso. Rispondi in perfetto italiano, spiegando passo dopo passo la deduzione.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Davanti a te ci sono tre scrigni: d'Oro, d'Argento e di Piombo. Su ciascuno c'e' un'iscrizione:\n"
            "- Scrigno d'Oro: 'Il tesoro si trova qui dentro.'\n"
            "- Scrigno d'Argento: 'Il tesoro non si trova qui dentro.'\n"
            "- Scrigno di Piombo: 'Il tesoro non si trova nello scrigno d'Oro.'\n"
            "Sapendo che solo una delle tre affermazioni e' vera e le altre due sono false, in quale scrigno si trova il tesoro? "
            "Dimostra formalmente perche' le altre possibilita' sono contraddittorie.<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "n_predict": 220
    },
    "B_Neuroscienze": {
        "title": "Spiegazione Scientifica (Neurodinamica Sinaptica)",
        "prompt": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Sei un neuroscienziato e biofisico computazionale. Scrivi in modo scientificamente accurato, elegante e denso.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Spiega il meccanismo molecolare ed elettrofisiologico del potenziamento a lungo termine (LTP) mediato dai recettori NMDA, "
            "dettagliando il ruolo dell'espulsione del blocco del magnesio (Mg2+) e la cascata del calcio intracellulare.<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "n_predict": 220
    },
    "C_Codice_Sintassi": {
        "title": "Struttura e Codice Tecnico (Algoritmo Matematico)",
        "prompt": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Sei un esperto ingegnere del software e matematico. Rispondi fornendo codice Python pulito, tipizzato e ben documentato.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Implementa una classe Python 'PicardWavefrontContraction' che simula l'iterazione di punto fisso di Picard-Lindelof "
            "per risolvere equazioni differenziali ordinarie vettoriali. Includi type hints, docstring complete e un esempio d'uso.<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "n_predict": 220
    }
}

CONFIGS = [
    {
        "id": "1_Baseline_Sequential",
        "name": "Baseline Sequenziale (No Opt, SMT Jitter, FP16 KV)",
        "args": ["-t", "8", "-c", "2048"]
    },
    {
        "id": "4_Wavefront_Linear_K4",
        "name": "Wavefront Linear K=4 (All Hardware Opts + Core Pinning + Q8 KV)",
        "args": [
            "-t", "8", "-tb", "8", "-C", "0x5555", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "2048",
            "--spec-type", "ngram-map-k4v",
            "--spec-ngram-map-k4v-size-n", "2",
            "--spec-ngram-map-k4v-size-m", "4",
            "--spec-ngram-map-k4v-min-hits", "1"
        ]
    },
    {
        "id": "8_Synaptic_Asymmetric_Tree",
        "name": "Synaptic Asymmetric Tree (Budget-Constrained M<=4, Hedging Split, Grammar-Primed)",
        "args": [
            "-t", "8", "-tb", "8", "-C", "0x5555", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "2048",
            "--spec-type", "ngram-map-k4v",
            "--spec-ngram-map-k4v-size-n", "2",
            "--spec-ngram-map-k4v-size-m", "4",
            "--spec-draft-p-split", "0.35",
            "--spec-draft-p-min", "0.20",
            "--spec-draft-n-max", "4",
            "--spec-ngram-map-k4v-min-hits", "1"
        ]
    }
]

def wait_for_server(timeout=35):
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

def run_benchmark():
    print("=" * 105)
    print("BENCHMARK COMPARATIVO FINALE: BASLEINE vs WAVEFRONT K=4 vs SYNAPTIC ASYMMETRIC TREE")
    print("Modello: Meta-Llama-3-8B-Instruct (4.66 GB, Zero Pruning, 100% Pesi Attivi)")
    print("=" * 105)

    all_results = {}

    for cfg in CONFIGS:
        cid = cfg["id"]
        cname = cfg["name"]
        print(f"\n=========================================================================================")
        print(f">>> TEST CONFIGURAZIONE [{cid}]: {cname}")
        print(f"=========================================================================================")

        cmd = [SERVER_EXE, "-m", MODEL_BLOB, "--port", str(PORT)] + cfg["args"]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        try:
            if not wait_for_server(35):
                print(f"    [ERRORE] Timeout avvio server per {cid}")
                continue

            cfg_res = {}
            for pkey, pinfo in PROMPTS.items():
                p_title = pinfo["title"]
                prompt = pinfo["prompt"]
                n_predict = pinfo["n_predict"]

                payload = {
                    "prompt": prompt,
                    "n_predict": n_predict,
                    "temperature": 0.0,
                    "stream": False
                }
                req_data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=req_data, headers={"Content-Type": "application/json"})

                t0 = time.time()
                try:
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        res = json.loads(resp.read().decode('utf-8'))
                except Exception as e:
                    print(f"    [ERRORE CHIAMATA] {pkey}: {e}")
                    continue
                dt = time.time() - t0

                timings = res.get('timings', {})
                n_tokens = timings.get('predicted_n', 0)
                tok_s = timings.get('predicted_per_second', 0.0)
                ms_tok = timings.get('predicted_per_token_ms', 0.0)
                draft_acc = timings.get('draft_n_accepted', 0)
                draft_n = timings.get('draft_n', 0)
                acc_rate = (draft_acc / draft_n * 100.0) if draft_n > 0 else 0.0

                content = res.get('content', '').strip()
                preview = content.replace('\n', ' ')[:100]

                print(f"  -> Task [{pkey}]: {n_tokens} tok in {dt:.2f}s | Speed: {tok_s:>5.2f} tok/s ({ms_tok:>5.1f} ms/tok) | Draft Acc: {acc_rate:>4.1f}% | Out: {preview}...")

                cfg_res[pkey] = {
                    "title": p_title,
                    "tokens": n_tokens,
                    "time_s": dt,
                    "tok_per_sec": tok_s,
                    "ms_per_tok": ms_tok,
                    "draft_accepted": draft_acc,
                    "draft_total": draft_n,
                    "draft_acc_rate": acc_rate,
                    "text": content
                }

            all_results[cid] = {
                "name": cname,
                "tasks": cfg_res
            }

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            time.sleep(1.0)

    # Print Summary Table
    print("\n" + "=" * 115)
    print("QUADRO DI SINTESI GENERATIVO (META-LLAMA-3-8B-INSTRUCT):")
    print("=" * 115)
    print(f"{'Configurazione':<45} | {'Logica (tok/s)':<15} | {'Neuro (tok/s)':<15} | {'Codice (tok/s)':<15} | {'Media':<12} | {'Draft Acc Med'}")
    print("-" * 115)
    for cid, data in all_results.items():
        cname = data["name"][:43]
        tasks = data["tasks"]
        t_log = tasks.get("A_Logica", {}).get("tok_per_sec", 0.0)
        t_neu = tasks.get("B_Neuroscienze", {}).get("tok_per_sec", 0.0)
        t_cod = tasks.get("C_Codice_Sintassi", {}).get("tok_per_sec", 0.0)
        a_log = tasks.get("A_Logica", {}).get("draft_acc_rate", 0.0)
        a_neu = tasks.get("B_Neuroscienze", {}).get("draft_acc_rate", 0.0)
        a_cod = tasks.get("C_Codice_Sintassi", {}).get("draft_acc_rate", 0.0)
        avg_speed = (t_log + t_neu + t_cod) / 3.0
        avg_acc = (a_log + a_neu + a_cod) / 3.0
        print(f"{cname:<45} | {t_log:>6.2f} tok/s    | {t_neu:>6.2f} tok/s    | {t_cod:>6.2f} tok/s    | {avg_speed:>5.2f} tok/s  | {avg_acc:>5.1f}%")
    print("=" * 115)

    # Save to JSON
    out_file = r"C:\Users\Umberto\synaptic-fold-v2\benchmarks\synaptic_asymmetric_tree_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[SALVATO] Risultati benchmark salvati in: {out_file}")

if __name__ == '__main__':
    run_benchmark()
