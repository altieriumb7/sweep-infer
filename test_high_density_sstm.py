import urllib.request
import json
import time
import sys
import subprocess

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SERVER_EXE = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"
MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa"
PORT = 8092

# High-density Syntactic-Semantic Invariant Manifold with discourse scaffolding
HIGH_DENSITY_SSTM = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Sei un assistente cognitivo in lingua italiana ad altissima precisione logica e sintattica. "
    "Nella risoluzione del classico indovinello del fiume con il contadino, il lupo, la capra e il cavolo, "
    "la struttura sequenziale formale della soluzione si articola rigorosamente nei seguenti passaggi:\n"
    "- Passo 1: Il contadino carica la capra sulla barca e attraversa il fiume verso la riva opposta.\n"
    "- Passo 2: Il contadino lascia la capra sulla riva opposta e torna da solo con la barca vuota.\n"
    "- Passo 3: Il contadino carica il cavolo sulla barca e lo trasporta sulla riva opposta.\n"
    "- Passo 4: Il contadino lascia il cavolo sulla riva opposta e riprende la capra con sé sulla barca tornando indietro.\n"
    "- Passo 5: Il contadino lascia la capra sulla riva iniziale e carica il lupo sulla barca portandolo sulla riva opposta.\n"
    "- Passo 6: Il contadino lascia il lupo sulla riva opposta con il cavolo e torna indietro da solo con la barca vuota.\n"
    "- Passo 7: Il contadino carica infine la capra sulla barca e la porta sulla riva opposta completando con successo la traversata.\n"
    "In questo modo la capra non resta mai sola con il cavolo e il lupo non resta mai solo con la capra. "
    "Ogni passaggio è descritto con precisione analitica, consequenzialità e perfetto rigore sintattico.<|eot_id|>"
)

PROMPT = (
    HIGH_DENSITY_SSTM +
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Descrivi i passaggi dettagliati della soluzione dell'indovinello del fiume con il contadino, la capra, il lupo e il cavolo.<|eot_id|>"
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

def test_dense_resonance(name, spec_type, n, m, hits):
    print(f"\n=======================================================")
    print(f"TEST {name}: spec_type={spec_type} (n={n}, m={m}, hits={hits})")
    print(f"=======================================================")

    cmd = [
        SERVER_EXE,
        "-m", MODEL_BLOB,
        "--threads", "8",
        "--port", str(PORT),
        "-c", "2048",
        "-b", "512",
        "-ub", "512",
        "--spec-type", spec_type,
        f"--spec-{spec_type}-size-n", str(n),
        f"--spec-{spec_type}-size-m", str(m),
        f"--spec-{spec_type}-min-hits", str(hits)
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_for_server():
            print("[ERRORE] Timeout avvio server.")
            return None

        payload = {
            'prompt': PROMPT,
            'n_predict': 180,
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

        print(f"  -> Testo generato:\n{res.get('content', '')[:250]}...\n")
        print(f"  -> Token generati: {tokens} in {dt:.2f} s")
        print(f"  -> Latenza per token (predicted): {pred_ms:.1f} ms/token")
        print(f"  -> Velocita di generazione pura: {pred_speed:.2f} token/s")
        print(f"  -> Prompt eval (GEMM 32L): {prompt_speed:.2f} token/s")
        print(f"  -> Copertura bozze: {draft_n}/{tokens} ({coverage:.1f}%)")
        print(f"  -> Bozze accettate: {draft_acc}/{draft_n} ({acc_rate:.1f}%)")

        return {
            'name': name,
            'tokens': tokens,
            'dt': dt,
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
    r1 = test_dense_resonance("MAP_K_N2_M5", "ngram-map-k", n=2, m=5, hits=1)
    r2 = test_dense_resonance("MAP_K_N2_M6", "ngram-map-k", n=2, m=6, hits=1)
    r3 = test_dense_resonance("MAP_K4V_N2_M5", "ngram-map-k4v", n=2, m=5, hits=1)

    print("\n" + "=" * 75)
    print("RISULTATI ATTRACTOR MANIFOLD AD ALTA DENSITÀ:")
    print("=" * 75)
    for r in [r1, r2, r3]:
        if r:
            print(f"{r['name']:<18} | {r['ms_per_token']:<6.1f} ms/tok | {r['tok_per_sec']:<6.2f} tok/s | Copertura: {r['coverage']:.1f}% | Accettaz: {r['acc_rate']:.1f}%")
    print("=" * 75)

if __name__ == '__main__':
    main()
