"""
Comprehensive Benchmark: All Architectural Optimizations on Actual Text Generation
Model: Meta-Llama-3-8B-Instruct (4.66 GB, Zero Pruning)
Hardware: AMD Ryzen 7 7730U (8 Core Zen 3, 16 GB DDR4-3200)

Evaluates:
1. Core Affinity Pinning (Zen 3 Physical Cores 0x5555)
2. Quantized KV-Cache (Q8_0)
3. Flash Attention on CPU (-fa on)
4. Wavefront Engine K=4
5. Deep Wavefront Engine K=6
6. Ultra Deep Wavefront Engine K=8
7. Tree-Branching Speculation

Across 3 Diverse Text Generation Tasks:
- Task A: Logica e Deduzione (Indovinello dei Tre Scrigni)
- Task B: Spiegazione Scientifica (Neurodinamica Sinaptica)
- Task C: Struttura / Codice e Sintassi (Algoritmo Matematico)
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
PORT = 8098

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
        "id": "2_Core_Affinity_Pinning",
        "name": "+ Core Affinity Pinning (0x5555 Physical Cores Zen 3)",
        "args": ["-t", "8", "-tb", "8", "-C", "0x5555", "-c", "2048"]
    },
    {
        "id": "3_Quantized_KV_FlashAttn",
        "name": "+ KV Cache Q8_0 & Flash Attention (-fa on, -ctk q8_0)",
        "args": ["-t", "8", "-tb", "8", "-C", "0x5555", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "2048"]
    },
    {
        "id": "4_Wavefront_K4",
        "name": "+ Wavefront Engine K=4 (All Prior Opts)",
        "args": [
            "-t", "8", "-tb", "8", "-C", "0x5555", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "2048",
            "--spec-type", "ngram-map-k4v",
            "--spec-ngram-map-k4v-size-n", "2",
            "--spec-ngram-map-k4v-size-m", "4",
            "--spec-ngram-map-k4v-min-hits", "1"
        ]
    },
    {
        "id": "5_Deep_Wavefront_K6",
        "name": "+ Deep Wavefront Engine K=6 (All Prior Opts)",
        "args": [
            "-t", "8", "-tb", "8", "-C", "0x5555", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "2048",
            "--spec-type", "ngram-map-k4v",
            "--spec-ngram-map-k4v-size-n", "2",
            "--spec-ngram-map-k4v-size-m", "6",
            "--spec-ngram-map-k4v-min-hits", "1"
        ]
    },
    {
        "id": "6_Ultra_Wavefront_K8",
        "name": "+ Ultra-Deep Wavefront Engine K=8 (All Prior Opts)",
        "args": [
            "-t", "8", "-tb", "8", "-C", "0x5555", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "2048",
            "--spec-type", "ngram-map-k4v",
            "--spec-ngram-map-k4v-size-n", "2",
            "--spec-ngram-map-k4v-size-m", "8",
            "--spec-ngram-map-k4v-min-hits", "1"
        ]
    },
    {
        "id": "7_Tree_Branching_Wavefront",
        "name": "+ Tree-Branching Speculation (Albero di Ipotesi)",
        "args": [
            "-t", "8", "-tb", "8", "-C", "0x5555", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "2048",
            "--spec-type", "ngram-map-k4v",
            "--spec-ngram-map-k4v-size-n", "2",
            "--spec-ngram-map-k4v-size-m", "4",
            "--spec-draft-p-split", "0.15",
            "--spec-ngram-map-k4v-min-hits", "1"
        ]
    }
]

def wait_for_server(timeout=30):
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
    print("=" * 100)
    print("BENCHMARK COMPLESSIVO: ABLATION E GENERAZIONE TESTUALE CON TUTTE LE OTTIMIZZAZIONI")
    print("Modello: Meta-Llama-3-8B-Instruct (4.66 GB, Zero Pruning) su AMD Ryzen 7 7730U (DDR4-3200)")
    print("=" * 100)

    overall_results = {}

    for cfg in CONFIGS:
        cid = cfg["id"]
        cname = cfg["name"]
        print(f"\n>>> CONFIGURAZIONE [{cid}]: {cname}")

        cmd = [SERVER_EXE, "-m", MODEL_BLOB, "--port", str(PORT)] + cfg["args"]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        try:
            if not wait_for_server(35):
                print(f"    [ERRORE] Server non pronto per {cid}")
                continue

            cfg_results = {}
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

                print(f"    Task [{pkey}]: {n_tokens} tok in {dt:.2f}s | Speed: {tok_s:>5.2f} tok/s ({ms_tok:>5.1f} ms/tok) | Draft Acc: {acc_rate:>4.1f}% | Out: {preview}...")

                cfg_results[pkey] = {
                    "title": p_title,
                    "tokens": n_tokens,
                    "time_s": dt,
                    "tok_per_sec": tok_s,
                    "ms_per_tok": ms_tok,
                    "draft_accepted": draft_acc,
                    "draft_total": draft_n,
                    "draft_acc_rate": acc_rate,
                    "generated_text": content
                }

            overall_results[cid] = {
                "name": cname,
                "tasks": cfg_results
            }

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            time.sleep(1.0)

    # Save to JSON
    out_path = r"C:\Users\Umberto\synaptic-fold-v2\benchmarks\all_optimizations_generation_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2, ensure_ascii=False)
    print(f"\n[SALVATO] Risultati completi salvati in: {out_path}")

    # Print Summary Table
    print("\n" + "=" * 105)
    print("QUADRO COMPARATIVO GENERATIVO DI TUTTE LE OTTIMIZZAZIONI (META-LLAMA-3-8B):")
    print("=" * 105)
    print(f"{'Configurazione':<45} | {'Logica (tok/s)':<15} | {'Neuro (tok/s)':<15} | {'Codice (tok/s)':<15} | {'Media'}")
    print("-" * 105)
    for cid, data in overall_results.items():
        cname = data["name"][:43]
        tasks = data["tasks"]
        t_log = tasks.get("A_Logica", {}).get("tok_per_sec", 0.0)
        t_neu = tasks.get("B_Neuroscienze", {}).get("tok_per_sec", 0.0)
        t_cod = tasks.get("C_Codice_Sintassi", {}).get("tok_per_sec", 0.0)
        avg_speed = (t_log + t_neu + t_cod) / 3.0
        print(f"{cname:<45} | {t_log:>6.2f} tok/s    | {t_neu:>6.2f} tok/s    | {t_cod:>6.2f} tok/s    | {avg_speed:>5.2f} tok/s")
    print("=" * 105)

if __name__ == '__main__':
    run_suite()
