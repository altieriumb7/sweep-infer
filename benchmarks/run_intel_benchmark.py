import ctypes
import os
import sys
import time
import json
import numpy as np

# Force UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = '/workspace/sweep'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LIB_DIR = '/workspace/llama.cpp/build-cpu/bin'
libggml = ctypes.CDLL(os.path.join(LIB_DIR, 'libggml.so'))
libllama = ctypes.CDLL(os.path.join(LIB_DIR, 'libllama.so'))
libllama.llama_backend_init()

from synaptic_fold.kernels.native_router import (
    evaluate_speculative_tree_avx2,
    compute_column_potentials_avx2,
    compute_topk_entropy_avx2,
    is_native_avx2_available,
    is_native_tree_avx2_available,
    is_native_topk_avx2_available,
    is_native_router_avx2_available,
    init_ngram_router_avx2,
    insert_ngram_router_avx2,
    query_ngram_router_avx2
)

print('=' * 80)
print('BENCHMARK SWEEP SU INTEL(R) CORE(TM) i5-14600KF (14 CORES / 20 THREADS)')
print('Target: Meta-Llama-3-8B-Instruct (4.66 GB, Q4_K_M, Zero Pruning)')
print('=' * 80)

# 1. Microarchitectural Profiling of AVX-2 Kernels on Intel Raptor Lake
print('\n[PARTE 1] Telemetria Microarchitetturale dei 4 Kernel AVX-2 su Intel i5-14600KF...')
D = 4096
num_cols = 1024
x = np.random.randn(D).astype(np.float32)
W = np.random.randn(num_cols, D).astype(np.float32)
b = np.random.randn(num_cols).astype(np.float32)

# Warmup
for _ in range(100):
    compute_column_potentials_avx2(x, W, b, 0.0)

t0 = time.perf_counter()
for _ in range(5000):
    compute_column_potentials_avx2(x, W, b, 0.0)
t_col = (time.perf_counter() - t0) / 5000.0 * 1e6
print(f'  1. Cortical Column AVX-2 Kernel: {t_col:.2f} us')

V = 128256
logits = np.random.randn(V).astype(np.float32)

for _ in range(100):
    compute_topk_entropy_avx2(logits, 16)

t0 = time.perf_counter()
for _ in range(5000):
    compute_topk_entropy_avx2(logits, 16)
t_topk = (time.perf_counter() - t0) / 5000.0 * 1e6
print(f'  2. Top-K Shannon Entropy AVX-2 Kernel: {t_topk:.2f} us')

N_nodes = 16
latents = np.random.randn(N_nodes, D).astype(np.float32)
branch_w = np.random.randn(N_nodes, D).astype(np.float32)
branch_b = np.random.randn(N_nodes).astype(np.float32)
parents = np.array([-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7], dtype=np.int32)

for _ in range(100):
    evaluate_speculative_tree_avx2(latents, branch_w, branch_b, parents, 0.15)

t0 = time.perf_counter()
for _ in range(5000):
    evaluate_speculative_tree_avx2(latents, branch_w, branch_b, parents, 0.15)
t_tree = (time.perf_counter() - t0) / 5000.0 * 1e6
print(f'  3. Speculative Tree Branching AVX-2 Kernel: {t_tree:.2f} us')

init_ngram_router_avx2()
for i in range(1000):
    insert_ngram_router_avx2(100 + i, 200 + i, 300 + i)

t0 = time.perf_counter()
for i in range(50000):
    query_ngram_router_avx2(100 + (i % 1000), 200 + (i % 1000))
t_router = (time.perf_counter() - t0) / 50000.0 * 1e9
print(f'  4. Hardware CRC32 Transition Router: {t_router:.1f} ns')

total_kernel_us = t_col + t_topk + t_tree + (t_router / 1000.0)
print(f'  --> OVERHEAD TOTALE DEI 4 KERNEL SU INTEL: {total_kernel_us:.2f} us')

# 2. In-Process LLM Decoding Benchmark
print('\n[PARTE 2] Caricamento Modello 8B in RAM e Misurazione GEMV / Wavefront...')
MODEL_PATH = '/workspace/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf'.encode('utf-8')

class ModelParamsBuf(ctypes.Structure):
    _fields_ = [('data', ctypes.c_uint8 * 128)]

class CtxBuf(ctypes.Structure):
    _fields_ = [('ints', ctypes.c_int32 * 64)]

class llama_batch(ctypes.Structure):
    _fields_ = [
        ('n_tokens', ctypes.c_int32),
        ('token', ctypes.POINTER(ctypes.c_int32)),
        ('embd', ctypes.POINTER(ctypes.c_float)),
        ('pos', ctypes.POINTER(ctypes.c_int32)),
        ('n_seq_id', ctypes.POINTER(ctypes.c_int32)),
        ('seq_id', ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ('logits', ctypes.POINTER(ctypes.c_int8)),
    ]

libllama.llama_model_default_params.restype = ModelParamsBuf
libllama.llama_context_default_params.restype = CtxBuf
libllama.llama_model_load_from_file.argtypes = [ctypes.c_char_p, ModelParamsBuf]
libllama.llama_model_load_from_file.restype = ctypes.c_void_p
libllama.llama_init_from_model.argtypes = [ctypes.c_void_p, CtxBuf]
libllama.llama_init_from_model.restype = ctypes.c_void_p
libllama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
libllama.llama_batch_init.restype = llama_batch
libllama.llama_batch_free.argtypes = [llama_batch]
libllama.llama_batch_free.restype = None
libllama.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
libllama.llama_decode.restype = ctypes.c_int32
libllama.llama_free.argtypes = [ctypes.c_void_p]
libllama.llama_free.restype = None
libllama.llama_model_free.argtypes = [ctypes.c_void_p]
libllama.llama_model_free.restype = None

mparams = libllama.llama_model_default_params()
cparams = libllama.llama_context_default_params()
cparams.ints[0] = 2048  # n_ctx
cparams.ints[7] = 14    # n_threads = 14 (all 14 physical cores of i5-14600KF)
cparams.ints[8] = 14    # n_threads_batch = 14

t_load0 = time.time()
model = libllama.llama_model_load_from_file(MODEL_PATH, mparams)
t_load = time.time() - t_load0
if not model:
    print('[ERRORE FATALE] Impossibile caricare il modello 8B.')
    sys.exit(1)
print(f'  Modello 8B mappato con successo in {t_load:.2f} s.')

ctx = libllama.llama_init_from_model(model, cparams)
batch = libllama.llama_batch_init(32, 0, 1)

# Warmup
batch.token[0] = 128000
batch.pos[0] = 0
batch.n_seq_id[0] = 1
batch.seq_id[0][0] = 0
batch.logits[0] = 1
batch.n_tokens = 1
libllama.llama_decode(ctx, batch)

# Sequential Baseline (K=1)
print('\n[PARTE 3] Misurazione Baseline Sequenziale (1 token per sweep)...')
single_times = []
curr_pos = 1
for s in range(8):
    batch.token[0] = 1000 + s
    batch.pos[0] = curr_pos
    batch.n_seq_id[0] = 1
    batch.seq_id[0][0] = 0
    batch.logits[0] = 1
    batch.n_tokens = 1
    t0 = time.perf_counter()
    ret = libllama.llama_decode(ctx, batch)
    dt = time.perf_counter() - t0
    if ret == 0:
        single_times.append(dt)
        curr_pos += 1
        print(f'  Passo {s+1}: {dt*1000.0:.1f} ms ({1.0/dt:.2f} tok/s)')

baseline_ms = (sum(single_times) / len(single_times)) * 1000.0
baseline_speed = 1000.0 / baseline_ms
print(f'  --> BASELINE SEQUENZIALE INTEL i5-14600KF: {baseline_ms:.1f} ms/token ({baseline_speed:.2f} tok/s)')

# Wavefront Parallel GEMM Scaling
print('\n[PARTE 4] Misurazione SWEEP Wavefront Scaling (K=2, 3, 4, 6)...')
wavefront_results = {}
for k in [2, 3, 4, 6]:
    k_times = []
    for rep in range(4):
        for i in range(k):
            batch.token[i] = 2000 + (k * 10) + i
            batch.pos[i] = curr_pos + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = 1
        batch.n_tokens = k
        t0 = time.perf_counter()
        ret = libllama.llama_decode(ctx, batch)
        dt = time.perf_counter() - t0
        if ret == 0:
            k_times.append(dt)
            curr_pos += k
    if k_times:
        avg_dt = sum(k_times) / len(k_times)
        lat_per_tok = (avg_dt * 1000.0) / k
        speed_tok_s = k / avg_dt
        ratio = speed_tok_s / baseline_speed
        wavefront_results[k] = {
            'sweep_time_ms': avg_dt * 1000.0,
            'ms_per_token': lat_per_tok,
            'tok_per_sec': speed_tok_s,
            'speedup': ratio
        }
        print(f'  Wavefront K={k}: Sweep = {avg_dt*1000.0:.1f} ms | Latenza/tok = {lat_per_tok:.1f} ms | Velocita = {speed_tok_s:.2f} tok/s ({ratio:.2f}x)')

print('\n' + '=' * 80)
print('TABELLA CONCLUSIVA INTEL CORE i5-14600KF (14 CORES)')
print('=' * 80)
print(f"{'Configurazione':<30} | {'Sweep (ms)':<11} | {'Latenza (ms/tok)':<18} | {'Throughput':<12} | {'Speedup'}")
print('-' * 80)
print(f"{'Sequential GEMV (Baseline 8B)':<30} | {baseline_ms:<11.1f} | {baseline_ms:<18.1f} | {baseline_speed:<6.2f} tok/s | 1.00x")
for k, res in wavefront_results.items():
    print(f"{'SWEEP Wavefront K=' + str(k) + ' (AVX-2)':<30} | {res['sweep_time_ms']:<11.1f} | {res['ms_per_token']:<18.1f} | {res['tok_per_sec']:<6.2f} tok/s | {res['speedup']:.2f}x")
print('=' * 80)

# Save JSON
out_path = '/workspace/sweep/benchmarks/intel_14600kf_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({
        'cpu': 'Intel(R) Core(TM) i5-14600KF',
        'cores': 14,
        'threads': 20,
        'model': 'Meta-Llama-3-8B-Instruct.Q4_K_M',
        'kernel_telemetry_us': {
            'column_kernel_us': t_col,
            'topk_entropy_us': t_topk,
            'tree_branching_us': t_tree,
            'crc32_router_ns': t_router,
            'total_us': total_kernel_us
        },
        'baseline': {'ms_per_token': baseline_ms, 'tok_per_sec': baseline_speed},
        'wavefront': wavefront_results
    }, f, indent=2)

print(f'\n[SALVATO] Risultati salvati in: {out_path}')
libllama.llama_batch_free(batch)
libllama.llama_free(ctx)
libllama.llama_model_free(model)
libllama.llama_backend_free()
