"""
Benchmark BoolQ: Current Solution (Wavefront Engine 8B Zero-Pruning) vs Previous Baseline
Target: 10 Validation Samples from google/boolq [8, 94, 67, 88, 39, 50, 43, 61, 73, 0]
Model: Meta-Llama-3-8B-Instruct (4.66 GB, Q4_0)
"""

import urllib.request
import json
import time
import sys
import os
import subprocess
from datasets import load_dataset

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SERVER_EXE = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"
MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa"
PORT = 8095

# The exact 10 validation indices tested in boolq_results.json
TARGET_INDICES = [8, 94, 67, 88, 39, 50, 43, 61, 73, 0]

# Historical baseline results from C:\Users\Umberto\synaptic-fold\benchmarks\boolq_results.json
HISTORICAL_BASELINE = {
    8:  {"pred": False, "target": True,  "correct": False, "elapsed_s": 60.8},
    94: {"pred": False, "target": True,  "correct": False, "elapsed_s": 68.2},
    67: {"pred": False, "target": False, "correct": True,  "elapsed_s": 77.3},
    88: {"pred": False, "target": True,  "correct": False, "elapsed_s": 76.1},
    39: {"pred": False, "target": False, "correct": True,  "elapsed_s": 78.1},
    50: {"pred": False, "target": True,  "correct": False, "elapsed_s": 77.3},
    43: {"pred": False, "target": False, "correct": True,  "elapsed_s": 92.0},
    61: {"pred": False, "target": False, "correct": True,  "elapsed_s": 79.3},
    73: {"pred": False, "target": True,  "correct": False, "elapsed_s": 93.0},
    0:  {"pred": False, "target": False, "correct": True,  "elapsed_s": 229.3},
}

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

def parse_answer(text: str) -> bool:
    clean = text.strip().lower()
    # Check beginning of response
    if clean.startswith("yes") or clean.startswith("true") or "is correct" in clean:
        return True
    if clean.startswith("no") or clean.startswith("false") or "not" in clean:
        return False
    # If not at start, look for first occurrence
    first_yes = clean.find("yes")
    first_no = clean.find("no")
    if first_yes != -1 and (first_no == -1 or first_yes < first_no):
        return True
    if first_no != -1 and (first_yes == -1 or first_no < first_yes):
        return False
    return False

def run_boolq_benchmark():
    print("=" * 95)
    print("BOOLQ 10-SAMPLE BENCHMARK: SOLUZIONE ATTUALE (8B WAVEFRONT) vs BASELINE PRECEDENTE")
    print("Modello Target: Meta-Llama-3-8B-Instruct (4.66 GB) su CPU Ryzen 7 7730U (Zero Pruning)")
    print("=" * 95)

    print("\n[1] Caricamento del dataset google/boolq (split: validation)...")
    ds = load_dataset('google/boolq', split='validation')
    print(f"  -> Dataset caricato ({len(ds)} campioni totali).")

    print("\n[2] Avvio del server nativo con motore di risonanza speculativa K=4...")
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
            print("[ERRORE FATALE] Impossibile avviare il server 8B.")
            return

        results = []
        total_correct = 0

        print("\n" + "=" * 95)
        print(f"{'#':<3} | {'Idx':<4} | {'Target':<6} | {'Base Pred':<10} | {'Base Time':<10} | {'Attuale':<8} | {'Tempo Attuale':<14} | {'Speedup':<8} | {'Esito'}")
        print("-" * 95)

        for i, idx in enumerate(TARGET_INDICES):
            item = ds[idx]
            passage = item['passage']
            question = item['question']
            target = item['answer']
            base_info = HISTORICAL_BASELINE.get(idx, {})

            # Format strictly with Llama-3 instruction chat template
            prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"You are a precise reading comprehension assistant. Answer the yes/no question based solely on the provided passage. "
                f"Respond with only 'Yes' or 'No', followed by a brief 1-sentence explanation.<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"Passage: {passage}\n"
                f"Question: {question}?<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

            payload = {
                'prompt': prompt,
                'n_predict': 35,
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

            content = res.get('content', '').strip()
            pred = parse_answer(content)
            is_correct = (pred == target)
            if is_correct:
                total_correct += 1

            timings = res.get('timings', {})
            pred_ms = timings.get('predicted_per_token_ms', 0.0)
            draft_acc = timings.get('draft_n_accepted', 0)
            draft_n = timings.get('draft_n', 0)

            base_time = base_info.get('elapsed_s', 80.0)
            speedup = base_time / dt if dt > 0 else 0.0

            base_pred_str = "False (KO)" if not base_info.get('correct') else "True (OK)"
            att_pred_str = f"{'Yes' if pred else 'No':<3} ({'OK' if is_correct else 'KO'})"
            esito_str = "CORRETTO" if is_correct else "ERRATO"

            print(f"{i+1:<3} | {idx:<4} | {str(target):<6} | {base_pred_str:<10} | {base_time:>6.1f} s    | {att_pred_str:<8} | {dt:>6.2f} s ({pred_ms:>4.0f}ms/t) | {speedup:>6.1f}x  | {esito_str}")
            print(f"      [Q]: {question[:75]}...")
            print(f"      [A]: {content[:95]}...\n")

            results.append({
                'idx': idx,
                'question': question,
                'target': target,
                'historical_baseline': base_info,
                'current_solution': {
                    'pred': pred,
                    'is_correct': is_correct,
                    'response_text': content,
                    'elapsed_s': dt,
                    'pred_ms_per_tok': pred_ms,
                    'draft_accepted': draft_acc,
                    'draft_total': draft_n,
                    'speedup_vs_baseline': speedup
                }
            })

        acc_pct = (total_correct / len(TARGET_INDICES)) * 100.0
        base_correct = sum(1 for b in HISTORICAL_BASELINE.values() if b['correct'])
        base_acc = (base_correct / len(TARGET_INDICES)) * 100.0
        avg_dt = sum(r['current_solution']['elapsed_s'] for r in results) / len(results)
        avg_base_t = sum(HISTORICAL_BASELINE[idx]['elapsed_s'] for idx in TARGET_INDICES) / len(TARGET_INDICES)
        overall_speedup = avg_base_t / avg_dt

        print("=" * 95)
        print("RIASSUNTO COMPARATIVO FINALE BOOLQ (10 CAMPIONI):")
        print("=" * 95)
        print(f"  - Accuratezza Baseline Precedente (70% Mute / Swapping) : {base_correct}/10 ({base_acc:.1f}%)")
        print(f"  - Accuratezza Soluzione Attuale (8B Zero Pruning Wavefront): {total_correct}/10 ({acc_pct:.1f}%)")
        print(f"  - Tempo Medio per Campione Baseline                    : {avg_base_t:.1f} s/campione")
        print(f"  - Tempo Medio per Campione Soluzione Attuale           : {avg_dt:.2f} s/campione")
        print(f"  - SPEEDUP MEDIO COMPLESSIVO                            : {overall_speedup:.1f}x PIÙ VELOCE!")
        print("=" * 95)

        # Save to json report
        out_path = r"C:\Users\Umberto\synaptic-fold-v2\benchmarks\boolq_current_vs_baseline.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                'samples_evaluated': len(TARGET_INDICES),
                'baseline_accuracy_pct': base_acc,
                'current_accuracy_pct': acc_pct,
                'baseline_avg_time_s': avg_base_t,
                'current_avg_time_s': avg_dt,
                'overall_speedup': overall_speedup,
                'detailed_results': results
            }, f, indent=2, ensure_ascii=False)
        print(f"\n[REPORT SALVATO] Dati completi salvati in: {out_path}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

if __name__ == '__main__':
    run_boolq_benchmark()
