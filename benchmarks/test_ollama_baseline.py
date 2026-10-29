import json
import time
import urllib.request

print('=' * 80)
print('BENCHMARK UFFICIALE BASELINE OLLAMA SU NVIDIA GEFORCE RTX 5070 Ti')
print('Modello: Meta-Llama-3-8B-Instruct (Q4_K_M, 100% GPU Offload)')
print('=' * 80)

# Warmup request
print('\n[1] Esecuzione Warmup Ollama...')
payload = {
    'model': 'llama3-local:latest',
    'prompt': 'Hello, count to 10.',
    'stream': False
}
req = urllib.request.Request(
    'http://127.0.0.1:11434/api/generate',
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
with urllib.request.urlopen(req) as resp:
    warmup_res = json.loads(resp.read().decode('utf-8'))
print('Warmup completato.')

# 5 Test Generation Runs with long responses
prompts = [
    'Write a concise 100-word explanation of quantum entanglement.',
    'List 10 key milestones in the history of artificial intelligence with brief descriptions.',
    'Explain how a compiler optimizes abstract syntax trees into machine code.',
    'Write a Python function to solve the knapsack problem using dynamic programming with comments.',
    'Describe the biological and architectural differences between human neurons and artificial transformers.'
]

speeds = []
print('\n[2] Esecuzione dei 5 Test di Generazione End-to-End su Ollama Baseline...\n')

for i, p in enumerate(prompts, 1):
    payload = {
        'model': 'llama3-local:latest',
        'prompt': p,
        'stream': False,
        'options': {
            'num_predict': 150,
            'temperature': 0.0
        }
    }
    req = urllib.request.Request(
        'http://127.0.0.1:11434/api/generate',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    wall_clock = time.perf_counter() - t0

    eval_count = data.get('eval_count', 0)
    eval_duration_ns = data.get('eval_duration', 1)
    prompt_eval_count = data.get('prompt_eval_count', 0)
    prompt_eval_ns = data.get('prompt_eval_duration', 1)

    gen_tok_s = (eval_count / eval_duration_ns) * 1e9
    pp_tok_s = (prompt_eval_count / prompt_eval_ns) * 1e9
    speeds.append(gen_tok_s)

    print(f'Test {i}: {eval_count} token generati in {eval_duration_ns / 1e9:.2f} s | Velocita Generazione: {gen_tok_s:.2f} tok/s (Prompt: {pp_tok_s:.1f} tok/s)')

avg_speed = sum(speeds) / len(speeds)
print('\n' + '=' * 80)
print(f'VELOCITA MEDIA BASELINE OLLAMA SU RTX 5070 Ti: {avg_speed:.2f} tok/s')
print('=' * 80)

# Check GPU utilization in Ollama logs
with open('/tmp/ollama_results.json', 'w') as f:
    json.dump({
        'engine': 'Ollama',
        'model': 'Meta-Llama-3-8B-Instruct (Q4_K_M)',
        'gpu': 'NVIDIA GeForce RTX 5070 Ti (16 GB GDDR7)',
        'avg_tok_s': avg_speed,
        'runs': speeds
    }, f, indent=2)
