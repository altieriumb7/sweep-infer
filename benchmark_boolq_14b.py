"""
Benchmark BoolQ on 14B Model (Qwen2.5-14B-Instruct, 14.7B parameters)
Evaluation on the identical 10 validation samples from google/boolq:
[8, 94, 67, 88, 39, 50, 43, 61, 73, 0]
Comparing accuracy and inference speed against 8B Baseline and 8B Wavefront.
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
MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54"
PORT = 8096

TARGET_INDICES = [8, 94, 67, 88, 39, 50, 43, 61, 73, 0]

# Historical results from previous benchmarks
BASELINE_8B = {
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

WAVEFRONT_8B = {
    8:  {"pred": True,  "target": True,  "correct": True,  "elapsed_s": 10.1},
    94: {"pred": True,  "target": True,  "correct": True,  "elapsed_s": 10.2},
    67: {"pred": True,  "target": False, "correct": False, "elapsed_s": 10.9},
    88: {"pred": True,  "target": True,  "correct": True,  "elapsed_s": 10.3},
    39: {"pred": False, "target": False, "correct": True,  "elapsed_s": 10.8},
    50: {"pred": True,  "target": True,  "correct": True,  "elapsed_s": 10.8},
    43: {"pred": False, "target": False, "correct": True,  "elapsed_s": 10.8},
    61: {"pred": True,  "target": False, "correct": False, "elapsed_s": 10.4},
    73: {"pred": True,  "target": True,  "correct": True,  "elapsed_s": 10.4},
    0:  {"pred": False, "target": False, "correct": True,  "elapsed_s": 10.5},
}

def wait_for_server(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
            with urllib.request.urlopen(req, timeout=1) as resp:
                if json.loads(resp.read().decode('utf-8')).get('status') == 'ok':
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False

def parse_answer(text: str) -> bool:
    clean = text.strip().lower()
    if clean.startswith("yes") or clean.startswith("true") or "is correct" in clean:
        return True
    if clean.startswith("no") or clean.startswith("false") or "not" in clean:
        return False
    first_yes = clean.find("yes")
    first_no = clean.find("no")
    if first_yes != -1 and (first_no == -1 or first_yes < first_no):
        return True
    if first_no != -1 and (first_yes == -1 or first_no < first_yes):
        return False
    return False

def run_boolq_14b():
    print("=" * 95)
    print("BOOLQ 10-SAMPLE BENCHMARK: QWEN2.5-14B (14.7B PARAMETRI, 9.0 GB)")
    print("Confronto Head-to-Head con Baseline 8B e Soluzione Wavefront 8B")
    print("=" * 95)

    if not os.path.exists(MODEL_BLOB):
        print(f"[ERRORE] File modello non trovato: {MODEL_BLOB}")
        return

    print("\n[1] Avvio server per Qwen2.5-14B (contesto 2048, 8 thread)...")
    server_cmd = [
        SERVER_EXE,
        "-m", MODEL_BLOB,
        "--port", str(PORT),
        "-c", "2048",
        "-t", "8",
        "-ngl", "0"
    ]
    proc = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        if not wait_for_server(120):
            print("[ERRORE] Il server 14B non ha risposto in tempo utile.")
            return

        print("    Server 14B avviato con successo e reattivo!")

        print("\n[2] Caricamento dataset BoolQ...")
        ds = load_dataset("google/boolq", split="validation")
        print(f"    Dataset caricato. Esecuzione su 10 indici deterministici: {TARGET_INDICES}")

        results_14b = {}
        correct_14b = 0
        total_time_14b = 0.0

        print("\n" + "-" * 95)
        print(f"{'Idx':<5} | {'Domanda':<45} | {'Atteso':<7} | {'Pred 14B':<9} | {'Esito':<6} | {'Tempo (s)':<9}")
        print("-" * 95)

        for idx in TARGET_INDICES:
            sample = ds[idx]
            question = sample["question"]
            passage = sample["passage"]
            target = sample["answer"]

            prompt = (
                f"<|im_start|>system\nYou are a precise reasoning assistant. Answer only 'Yes' or 'No'.<|im_end|>\n"
                f"<|im_start|>user\nContext: {passage}\nQuestion: {question}?\nAnswer with only Yes or No.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            payload = {
                "prompt": prompt,
                "n_predict": 16,
                "temperature": 0.0,
                "top_p": 1.0,
                "stop": ["<|im_end|>", "\n", "."]
            }

            req_data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/completion",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )

            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    resp_json = json.loads(resp.read().decode('utf-8'))
                    ans_text = resp_json.get("content", "").strip()
            except Exception as e:
                ans_text = f"ERROR: {e}"

            elapsed = time.time() - t0
            pred = parse_answer(ans_text)
            is_correct = (pred == target)
            if is_correct:
                correct_14b += 1
            total_time_14b += elapsed

            results_14b[idx] = {
                "question": question,
                "target": target,
                "raw_pred": ans_text,
                "pred": pred,
                "correct": is_correct,
                "elapsed_s": round(elapsed, 2)
            }

            q_short = (question[:42] + '...') if len(question) > 42 else question
            esito_str = "OK" if is_correct else "ERR"
            print(f"{idx:<5} | {q_short:<45} | {str(target):<7} | {str(pred):<9} | {esito_str:<6} | {elapsed:<9.2f}")

        acc_14b = (correct_14b / len(TARGET_INDICES)) * 100.0
        avg_time_14b = total_time_14b / len(TARGET_INDICES)

        print("-" * 95)
        print(f"\n[3] RIEPILOGO STATISTICO E CONFRONTO:")
        print(f"    - Accuratezza Qwen2.5-14B:      {acc_14b:.1f}% ({correct_14b}/{len(TARGET_INDICES)})")
        print(f"    - Tempo medio per campione 14B: {avg_time_14b:.2f} s")
        print(f"    - Tempo totale inferenza 14B:   {total_time_14b:.2f} s")

        print("\n" + "=" * 95)
        print("CONFRONTO TRIANGOLARE SUI 10 CAMPIONI BOOLQ:")
        print("=" * 95)
        print(f"{'Idx':<5} | {'Target':<7} | {'Baseline 8B':<14} | {'Wavefront 8B':<14} | {'Qwen2.5-14B':<14}")
        print("-" * 95)
        for idx in TARGET_INDICES:
            b8 = "OK" if BASELINE_8B[idx]["correct"] else "ERR"
            w8 = "OK" if WAVEFRONT_8B[idx]["correct"] else "ERR"
            q14 = "OK" if results_14b[idx]["correct"] else "ERR"
            t_b8 = BASELINE_8B[idx]["elapsed_s"]
            t_w8 = WAVEFRONT_8B[idx]["elapsed_s"]
            t_q14 = results_14b[idx]["elapsed_s"]
            print(f"{idx:<5} | {str(BASELINE_8B[idx]['target']):<7} | {b8} ({t_b8:.1f}s)      | {w8} ({t_w8:.1f}s)      | {q14} ({t_q14:.1f}s)")

        print("=" * 95)
        print(f"{'Accuratezza':<14} | {'50.0%':<14} | {'80.0%':<14} | {f'{acc_14b:.1f}%':<14}")
        print(f"{'Tempo Medio/Q':<14} | {'93.1 s':<14} | {'10.5 s':<14} | {f'{avg_time_14b:.1f} s':<14}")
        print("=" * 95)

        out_path = r"C:\Users\Umberto\synaptic-fold-v2\benchmarks\boolq_14b_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": "Qwen2.5-14B-Instruct",
                "accuracy": acc_14b,
                "avg_elapsed_s": avg_time_14b,
                "total_elapsed_s": total_time_14b,
                "samples": results_14b
            }, f, indent=2)
        print(f"\n[SALVATO] Risultati salvati in: {out_path}")

    finally:
        print("\nArresto del server 14B...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print("Server 14B arrestato.")

if __name__ == '__main__':
    run_boolq_14b()
