import urllib.request
import json
import time
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

url = 'http://127.0.0.1:8089/completion'

base_prompt = "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura che la diritta via era smarrita. Ahi quanto a dir qual era e cosa dura esta selva selvaggia e aspra e forte che nel pensier rinova la paura! "

for n in [1, 2, 4, 8, 16]:
    # Warmup / evaluate prompt of length + n
    prompt = base_prompt + (" token" * n)
    payload = {
        'prompt': prompt,
        'n_predict': 1,
        'temperature': 0.0,
        'stream': False
    }
    
    times = []
    for _ in range(3):
        t0 = time.time()
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode('utf-8'))
        times.append(time.time() - t0)
    
    avg_t = sum(times) / len(times)
    timings = res.get('timings', {})
    prompt_ms = timings.get('prompt_ms', 0)
    prompt_n = timings.get('prompt_n', 0)
    pred_ms = timings.get('predicted_ms', 0)
    print(f"Tokens added: {n:2d} | Total request: {avg_t*1000:.1f}ms | prompt_eval: {prompt_ms:.1f}ms for {prompt_n} tok ({timings.get('prompt_per_second', 0):.2f} t/s) | single decode: {pred_ms:.1f}ms")
