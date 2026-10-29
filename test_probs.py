import urllib.request
import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

url = 'http://127.0.0.1:8089/completion'
prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nCompleta: Nel mezzo del cammin di nostra<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
payload = {
    'prompt': prompt,
    'n_predict': 5,
    'n_probs': 5,
    'temperature': 0.0
}
req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode('utf-8'))

print('CONTENT:', res.get('content'))
probs = res.get('completion_probabilities', [])
for i, p in enumerate(probs):
    print(f"\nPosition {i}: token='{p.get('token')}' (id={p.get('id')}), logprob={p.get('logprob')}")
    print("  Top alternatives:")
    for top in p.get('top_logprobs', [])[:4]:
        print(f"    - '{top.get('token')}' (id={top.get('id')}): {top.get('logprob'):.4f}")
