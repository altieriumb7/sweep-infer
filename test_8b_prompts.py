import urllib.request
import json
import time
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

url = 'http://127.0.0.1:8089/completion'

prompts = [
    ("Indovinello Fiume", "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nUn contadino deve attraversare un fiume con un lupo, una capra e un cavolo. La barca puo portare solo lui e un solo oggetto/animale. Se lasciati soli, il lupo mangia la capra e la capra mangia il cavolo. Come fa a passare il fiume senza perdite?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    ("Saluto Umberto", "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nCiao se io mi chiamo Umberto, tu come ti chiami?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    ("Concetto Coscienza", "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSpiegami in due frasi chiare ed essenziali cosa differenzia la memoria biologica dalla memoria di un calcolatore.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
]

for name, p in prompts:
    print(f"\n==================== TEST: {name} ====================")
    payload = {
        'prompt': p,
        'n_predict': 150,
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
    print(f"Generated text:\n{res.get('content')}")
    print(f"\nTokens predicted: {tokens}")
    print(f"Total time: {dt:.2f} s")
    print(f"Effective generation speed: {speed:.2f} tokens/s")
    timings = res.get('timings', {})
    print(f"Timings: predicted_per_second={timings.get('predicted_per_second')}, prompt_per_second={timings.get('prompt_per_second')}, draft_n={timings.get('draft_n')}, draft_n_accepted={timings.get('draft_n_accepted')}")
