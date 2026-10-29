import ctypes
import os
import sys
import time
import json
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = '/workspace/sweep'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LIB_DIR = '/workspace/llama.cpp/build-cuda/bin'
libllama = ctypes.CDLL(os.path.join(LIB_DIR, 'libllama.so'))
libllama.llama_backend_init()

libcudart = ctypes.CDLL('libcudart.so')

# CUDA kernel
CUDA_OPT_LIB_PATH = '/workspace/sweep/synaptic_fold/kernels/synaptic_entropy_topk_cuda_opt.so'
cuda_lib = ctypes.CDLL(CUDA_OPT_LIB_PATH)
cuda_lib.launch_sweep_gpu_entropy_opt.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]
cuda_lib.launch_sweep_gpu_entropy_opt.restype = None

print('=' * 80)
print('BENCHMARK SPETTRO ENTROPICO SU NVIDIA GEFORCE RTX 5070 Ti')
print('Analisi della Generalizzazione di SWEEP su 3 Regimi di Imprevedibilita')
print('Target: Meta-Llama-3-8B-Instruct (4.58 GiB, 100% GPU Offload)')
print('=' * 80)

# Prompts across 3 distinct entropy tiers
regimes = {
    'Tier 1 (Bassa Entropia - Codice e JSON)': [
        'Write a JSON schema for a user profile with id, name, email, and roles.',
        'def fibonacci_iterative(n: int) -> int:\n    """Return the nth Fibonacci number."""\n',
        'Write a standard HTML5 template with header, main content, and footer.',
        'def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n'
    ],
    'Tier 2 (Media Entropia - Matematica e Logica)': [
        'A train leaves town A at 60 mph. Another leaves town B at 80 mph. If they are 280 miles apart, when do they meet?',
        'If all roses are flowers and some flowers fade quickly, does it follow that some roses fade quickly? Explain carefully.',
        'A store offers a 20% discount on a $150 coat, then adds 8% sales tax. What is the final price?'
    ],
    'Tier 3 (Alta Entropia - Scrittura Creativa e Aperta)': [
        'Write a poetic, surreal metaphor describing an ancient clock on a deserted beach.',
        'Imagine a conversation between the ocean and the moon about human dreams.',
        'Compose an abstract story where colors represent emotions that change the weather.'
    ]
}

# Run generation for each prompt using llama-cli and measure tokens/second
results_by_regime = {}

for reg_name, prompt_list in regimes.items():
    print(f'\n--- Esecuzione {reg_name} ---')
    reg_tok_s = []
    for p in prompt_list:
        cmd = f"/workspace/llama.cpp/build-cuda/bin/llama-cli -m /workspace/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf -ngl 99 -p \"{p}\" -n 64 --no-warmup 2>&1 | grep -o 'Generation: [0-9.]*' | awk '{{print $2}}'"
        t0 = time.time()
        out = os.popen(cmd).read().strip()
        try:
            val = float(out.split('\n')[0])
            reg_tok_s.append(val)
            print(f'  Prompt: "{p[:45]}..." -> Baseline = {val:.1f} tok/s')
        except:
            pass
    if reg_tok_s:
        avg_val = sum(reg_tok_s) / len(reg_tok_s)
        results_by_regime[reg_name] = {
            'avg_baseline_tok_s': avg_val,
            'runs': reg_tok_s
        }
        print(f'  --> Media Baseline {reg_name}: {avg_val:.1f} tok/s')

out_path = '/workspace/sweep/benchmarks/entropy_spectrum_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results_by_regime, f, indent=2)

print(f'\n[SALVATO] Risultati salvati in: {out_path}')
