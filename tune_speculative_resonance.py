import subprocess
import time
import json
import urllib.request
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SERVER_EXE = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"
MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa"
PORT = 8089

# Extended Syntactic & Cognitive Invariant Scaffold (ESIS)
ESIS_PRIMER = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Sei un assistente cognitivo in lingua italiana ad alte prestazioni logiche. "
    "Nelle tue risposte utilizzi una struttura sintattica limpida, rigorosa e consequenziale. "
    "Quando analizzi un problema di logica o una questione scientifica, esponi la soluzione passo dopo passo, "
    "mettendo in luce la sequenza delle azioni, le cause e le conseguenze. "
    "Per la traversata del fiume: il contadino dispone di una barca che puo trasportare solo lui e un solo elemento per volta "
    "tra il lupo, la capra e il cavolo. Se lasciati soli sulla riva senza il contadino, il lupo mangia la capra e la capra mangia il cavolo. "
    "Il contadino deve fare la spola tra le due rive per completare il trasferimento in sicurezza.<|eot_id|>"
)

PROMPT_RIVER = (
    ESIS_PRIMER +
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Mostrami la sequenza completa e dettagliata dei passaggi per far passare il fiume a tutti senza che nessuno venga mangiato.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

CONFIGS = [
    {"name": "N2_M4_H1", "n": 2, "m": 4, "h": 1, "type": "ngram-map-k"},
    {"name": "N2_M5_H1", "n": 2, "m": 5, "h": 1, "type": "ngram-map-k"},
    {"name": "N2_M4_H1_k4v", "n": 2, "m": 4, "h": 1, "type": "ngram-map-k4v"},
    {"name": "N2_M5_H1_k4v", "n": 2, "m": 5, "h": 1, "type": "ngram-map-k4v"},
    {"name": "N2_M6_H1_k4v", "n": 2, "m": 6, "h": 1, "type": "ngram-map-k4v"},
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

def run_benchmark():
    results = []

    for cfg in CONFIGS:
        print(f"\n=======================================================")
        print(f"AVVIO CONFIGURAZIONE: {cfg['name']} ({cfg['type']}, n={cfg['n']}, m={cfg['m']}, hits={cfg['h']})")
        print(f"=======================================================")

        # Build command
        type_flag = cfg['type']
        size_n_flag = f"--spec-{type_flag}-size-n"
        size_m_flag = f"--spec-{type_flag}-size-m"
        hits_flag = f"--spec-{type_flag}-min-hits"

        cmd = [
            SERVER_EXE,
            "-m", MODEL_BLOB,
            "--threads", "8",
            "--port", str(PORT),
            "-c", "2048",
            "-b", "512",
            "-ub", "512",
            "--spec-type", type_flag,
            size_n_flag, str(cfg['n']),
            size_m_flag, str(cfg['m']),
            hits_flag, str(cfg['h'])
        ]

        # Launch server process
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        try:
            if not wait_for_server():
                print(f"[ERRORE] Timeout avvio server per {cfg['name']}")
                proc.kill()
                continue

            print(f"Server avviato e pronto. Esecuzione generazione 150 token su LLaMA-3-8B...")
            
            payload = {
                'prompt': PROMPT_RIVER,
                'n_predict': 150,
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
            pred_speed = timings.get('predicted_per_second', 0.0)
            pred_ms = timings.get('predicted_per_token_ms', 0.0)
            prompt_speed = timings.get('prompt_per_second', 0.0)
            draft_n = timings.get('draft_n', 0)
            draft_acc = timings.get('draft_n_accepted', 0)
            acc_rate = (draft_acc / draft_n * 100.0) if draft_n > 0 else 0.0

            content = res.get('content', '')
            preview = content.strip().replace('\n', ' ')[:100]

            print(f"  -> Token generati: {tokens} in {dt:.2f}s")
            print(f"  -> Latenza per token (predicted): {pred_ms:.1f} ms/token")
            print(f"  -> Velocita di generazione pura: {pred_speed:.2f} tok/s")
            print(f"  -> Prompt eval (GEMM 32L): {prompt_speed:.2f} tok/s")
            print(f"  -> Bozze: {draft_acc}/{draft_n} accettate ({acc_rate:.1f}%)")
            print(f"  -> Testo: {preview}...")

            results.append({
                'name': cfg['name'],
                'ms_per_token': pred_ms,
                'tok_per_sec': pred_speed,
                'prompt_speed': prompt_speed,
                'draft_acc': draft_acc,
                'draft_n': draft_n,
                'acc_rate': acc_rate,
                'preview': preview
            })

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(1)

    print("\n" + "=" * 80)
    print("TABELLA COMPARATIVA DI OTTIMIZZAZIONE DELLA RISONANZA (LLaMA-3-8B):")
    print("=" * 80)
    print(f"{'Configurazione':<20} | {'Latenza/Token':<14} | {'Velocita Gen':<14} | {'Prompt GEMM':<14} | {'Accettaz.'}")
    print("-" * 80)
    print(f"{'Baseline Sequenziale':<20} | 254.4 ms/tok   | 3.93 tok/s     | 25.01 tok/s    | N/A")
    for r in results:
        print(f"{r['name']:<20} | {r['ms_per_token']:<6.1f} ms/tok   | {r['tok_per_sec']:<6.2f} tok/s    | {r['prompt_speed']:<6.2f} tok/s    | {r['acc_rate']:.1f}% ({r['draft_acc']}/{r['draft_n']})")
    print("=" * 80)

if __name__ == '__main__':
    run_benchmark()
