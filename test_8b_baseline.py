import urllib.request
import json
import time
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

url = 'http://127.0.0.1:8089/completion'
prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nCiao se io mi chiamo Umberto, tu come ti chiami?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
payload = {
    'prompt': prompt,
    'n_predict': 100,
    'temperature': 0.0,
    'stream': False
}

t0 = time.time()
req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode('utf-8'))
dt = time.time() - t0

tokens = res.get('tokens_predicted', 0)
speed = tokens / dt if dt > 0 else 0
print(f"Generated text: {res.get('content')}")
print(f"Tokens predicted: {tokens}")
print(f"Total time: {dt:.2f} s")
print(f"Generation speed: {speed:.2f} tokens/s")
print(f"Timings: {json.dumps(res.get('timings', {}), indent=2)}")
