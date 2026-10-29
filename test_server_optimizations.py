import subprocess
import urllib.request
import json
import time
import os

SERVER_EXE = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama\llama-server.exe"
MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa"
PORT = 8097

cmd = [
    SERVER_EXE,
    "-m", MODEL_BLOB,
    "--port", str(PORT),
    "-c", "2048",
    "-t", "8",
    "-tb", "8",
    "-C", "0x5555",
    "-fa", "on",
    "-ctk", "q8_0",
    "-ctv", "q8_0",
    "--spec-type", "ngram-map-k4v",
    "--spec-ngram-map-k4v-size-n", "2",
    "--spec-ngram-map-k4v-size-m", "4",
    "--spec-ngram-map-k4v-min-hits", "1"
]

print("Launching test server with: Core Affinity (0x5555), Flash Attention, Q8_0 KV Cache, Wavefront K=4...")
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')

t0 = time.time()
server_ready = False
while time.time() - t0 < 30:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/health")
        with urllib.request.urlopen(req, timeout=1) as resp:
            if json.loads(resp.read().decode('utf-8')).get('status') == 'ok':
                server_ready = True
                break
    except Exception:
        pass
    time.sleep(0.5)

print(f"Server ready status: {server_ready} in {time.time()-t0:.2f}s")

if server_ready:
    # Test generation
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Sei un assistente logico brillante. Rispondi in italiano in modo chiaro ed esaustivo.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Spiega brevemente come la sincronizzazione di fase e l'eccitazione corticale permettono al cervello di elaborare informazioni in parallelo.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    req_data = json.dumps({
        "prompt": prompt,
        "n_predict": 120,
        "temperature": 0.0,
        "stream": False
    }).encode('utf-8')
    t_gen0 = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=req_data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode('utf-8'))
    dt = time.time() - t_gen0
    timings = res.get('timings', {})
    n_tokens = timings.get('predicted_n', 0)
    tok_s = timings.get('predicted_per_second', 0.0)
    ms_tok = timings.get('predicted_per_token_ms', 0.0)
    draft_acc = timings.get('draft_n_accepted', 0)
    draft_n = timings.get('draft_n', 0)
    print("\n--- RISULTATO GENERAZIONE TEST ---")
    print(f"Token generati: {n_tokens} in {dt:.2f} s")
    print(f"Velocita decodifica: {tok_s:.2f} tok/s ({ms_tok:.1f} ms/token)")
    print(f"Draft accettati: {draft_acc} / {draft_n} ({draft_acc/draft_n*100:.1f}%)" if draft_n > 0 else "Draft: N/A")
    print("\nTesto generato:\n", res.get('content', '').strip()[:300], "...\n")

proc.terminate()
try:
    proc.wait(timeout=5)
except Exception:
    proc.kill()
print("Server arrestato.")
