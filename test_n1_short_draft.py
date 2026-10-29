import urllib.request
import json
import time
import sys
import subprocess

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SERVER_EXE = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"
MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa"
PORT = 8089

# Dense Syntactic-Semantic Invariant Manifold
SSTM_INVARIANT_PRIMER = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Sei un assistente cognitivo in lingua italiana ad altissima precisione logica e sintattica. "
    "Nella lingua italiana, la costruzione del discorso segue regole di coerenza ferree: "
    "il contadino prende la barca per attraversare il fiume; "
    "sulla riva opposta lascia la capra e torna indietro con la barca vuota; "
    "poi prende il cavolo e lo trasporta sulla riva opposta; "
    "quindi riporta indietro la capra per evitare che mangi il cavolo; "
    "lascia la capra sulla riva iniziale e prende il lupo; "
    "porta il lupo sulla riva opposta con il cavolo, lasciando la capra da sola; "
    "infine torna indietro a prendere la capra e completa la traversata. "
    "In questo modo nessuno degli animali o degli oggetti viene mangiato o perduto. "
    "La soluzione è logica, consequenziale, perfetta e priva di contraddizioni. "
    "Rispondi sempre passo dopo passo con chiarezza, precisione e rigore assoluto.<|eot_id|>"
)

PROMPT_TEST = (
    SSTM_INVARIANT_PRIMER +
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Spiegami la soluzione completa della traversata del fiume con tutti i singoli passaggi necessari.<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

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

def test_config(name, n, m, hits):
    print(f"\n=======================================================")
    print(f"TEST {name}: ngram-map-k (n={n}, m={m}, hits={hits})")
    print(f"=======================================================")

    cmd = [
        SERVER_EXE,
        "-m", MODEL_BLOB,
        "--threads", "8",
        "--port", str(PORT),
        "-c", "2048",
        "-b", "512",
        "-ub", "512",
        "--spec-type", "ngram-map-k",
        "--spec-ngram-map-k-size-n", str(n),
        "--spec-ngram-map-k-size-m", str(m),
        "--spec-ngram-map-k-min-hits", str(hits)
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_for_server():
            print("[ERRORE] Timeout avvio server.")
            return None

        payload = {
            'prompt': PROMPT_TEST,
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
        coverage = (draft_n / tokens * 100.0) if tokens > 0 else 0.0

        print(f"  -> Token generati: {tokens} in {dt:.2f} s")
        print(f"  -> Latenza per token (predicted): {pred_ms:.1f} ms/token")
        print(f"  -> Velocita di generazione pura: {pred_speed:.2f} token/s")
        print(f"  -> Prompt eval (GEMM 32L): {prompt_speed:.2f} token/s")
        print(f"  -> Copertura bozze: {draft_n}/{tokens} ({coverage:.1f}%)")
        print(f"  -> Bozze accettate: {draft_acc}/{draft_n} ({acc_rate:.1f}%)")

        return {
            'name': name,
            'ms_per_token': pred_ms,
            'tok_per_sec': pred_speed,
            'prompt_speed': prompt_speed,
            'coverage': coverage,
            'acc_rate': acc_rate
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(1)

def main():
    r1 = test_config("N1_M2_H1", n=1, m=2, hits=1)
    r2 = test_config("N1_M3_H1", n=1, m=3, hits=1)
    r3 = test_config("N1_M2_H2", n=1, m=2, hits=2)
    
    print("\n" + "=" * 70)
    print("RIEPILOGO CONFIGURAZIONI N=1 (MASSIMA COPERTURA SINTATTICA):")
    print("=" * 70)
    for r in [r1, r2, r3]:
        if r:
            print(f"{r['name']:<15} | {r['ms_per_token']:<6.1f} ms/tok | {r['tok_per_sec']:<6.2f} tok/s | Copertura: {r['coverage']:.1f}% | Accettaz: {r['acc_rate']:.1f}%")
    print("=" * 70)

if __name__ == '__main__':
    main()
