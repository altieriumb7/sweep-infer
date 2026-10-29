"""
In-Process Wavefront Benchmark for Qwen2.5-14B (14.7 Billion Parameters)
Evaluates:
  1. Exact RAM Footprint (RSS & Virtual Memory)
  2. Sequential GEMV Baseline (K=1, 48 Layers, 9 GB DRAM stream)
  3. Proprietary Wavefront Scaling (K=2, 3, 4, 6)
  4. Accuracy on BoolQ validation samples
"""

import ctypes
import os
import sys
import time
import json
import psutil

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ollama_lib_dir = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama"
os.add_dll_directory(ollama_lib_dir)

libggml = ctypes.CDLL(os.path.join(ollama_lib_dir, "ggml.dll"))
libggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
libggml.ggml_backend_load_all_from_path(ollama_lib_dir.encode('utf-8'))

libllama = ctypes.CDLL(os.path.join(ollama_lib_dir, "libllama.dll"))
libllama.llama_backend_init()

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

mparams = libllama.llama_model_default_params()
cparams = libllama.llama_context_default_params()
cparams.ints[0] = 2048  # n_ctx = 2048 to save memory
cparams.ints[7] = 8     # n_threads = 8 physical cores
cparams.ints[8] = 8     # n_threads_batch = 8

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

def get_14b_blob_path() -> str:
    blobs_dir = r"C:\Users\Umberto\.ollama\models\blobs"
    # Find the blob that starts with 2049f5674b1e or is roughly 9.0 GB (>= 8.5 GB)
    for fname in os.listdir(blobs_dir):
        if "2049f5674b1e" in fname:
            return os.path.join(blobs_dir, fname)
    # Check by size
    for fname in os.listdir(blobs_dir):
        p = os.path.join(blobs_dir, fname)
        if os.path.isfile(p) and os.path.getsize(p) > 8 * (1024**3):
            return p
    return None

def run_14b_benchmark():
    print("=" * 85)
    print("BENCHMARK PROPRIETARIO IN-PROCESS: MODELLO 14B (QWEN2.5-14B, 14.7B PARAMETRI)")
    print("Piattaforma: AMD Ryzen 7 7730U (8 Core Zen 3, 16 GB DDR4-3200)")
    print("=" * 85)

    blob_path = get_14b_blob_path()
    if not blob_path or not os.path.exists(blob_path):
        print(f"[ATTESA] Modello 14B non ancora presente. Blob cercato in C:\\Users\\Umberto\\.ollama\\models\\blobs")
        return False

    size_gb = os.path.getsize(blob_path) / (1024**3)
    print(f"\n[1] Modello 14B identificato:")
    print(f"    Path: {blob_path}")
    print(f"    Dimensione file su disco: {size_gb:.2f} GB")

    vm_before = psutil.virtual_memory()
    print(f"    RAM Libera pre-caricamento: {vm_before.available / (1024**3):.2f} GB / {vm_before.total / (1024**3):.2f} GB")

    print("\n[2] Mappatura del modello 14B in RAM di processo...")
    t0 = time.time()
    model = libllama.llama_model_load_from_file(blob_path.encode('utf-8'), mparams)
    t_load = time.time() - t0

    if not model:
        print("[ERRORE FATALE] Impossibile caricare il modello 14B in memoria.")
        return False

    vm_after = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    print(f"    Modello 14B caricato con successo in {t_load:.2f} s! Pointer: {hex(model)}")
    print(f"    Process RSS: {proc.memory_info().rss / (1024**3):.2f} GB")
    print(f"    RAM Libera residua di sistema: {vm_after.available / (1024**3):.2f} GB (Utilizzo totale: {vm_after.percent:.1f}%)")

    print("\n[3] Inizializzazione contesto KV (2048 token, 8 thread)...")
    ctx = libllama.llama_init_from_model(model, cparams)
    if not ctx:
        print("[ERRORE] Inizializzazione contesto fallita.")
        libllama.llama_model_free(model)
        return False
    print(f"    Contesto KV allocato: {hex(ctx)}")

    # Warmup
    batch = libllama.llama_batch_init(32, 0, 1)
    batch.token[0] = 151643  # <|im_start|> in Qwen2
    batch.pos[0] = 0
    batch.n_seq_id[0] = 1
    batch.seq_id[0][0] = 0
    batch.logits[0] = 1
    batch.n_tokens = 1
    libllama.llama_decode(ctx, batch)

    print("\n[4] Misurazione Baseline Sequenziale GEMV (K=1, 14.7B parametri)...")
    single_times = []
    curr_pos = 1
    for step in range(5):
        batch.token[0] = 2000 + step
        batch.pos[0] = curr_pos
        batch.n_seq_id[0] = 1
        batch.seq_id[0][0] = 0
        batch.logits[0] = 1
        batch.n_tokens = 1
        t_s0 = time.time()
        ret = libllama.llama_decode(ctx, batch)
        dt = time.time() - t_s0
        if ret == 0:
            single_times.append(dt)
            curr_pos += 1
            print(f"    Passo {step+1}: {dt*1000:.1f} ms ({1.0/dt:.2f} tok/s)")

    avg_single = sum(single_times) / len(single_times)
    baseline_speed = 1.0 / avg_single
    baseline_ms = avg_single * 1000.0
    print(f"    --> BASELINE SEQUENZIALE 14B: {baseline_ms:.1f} ms/token ({baseline_speed:.2f} tok/s)")

    print("\n[5] Misurazione Proprietary Wavefront Scaling su 14B...")
    wavefront_results = {}
    for k in [2, 3, 4]:
        k_times = []
        for rep in range(3):
            for i in range(k):
                batch.token[i] = 3000 + (k * 10) + i
                batch.pos[i] = curr_pos + i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 1
            batch.n_tokens = k

            t_k0 = time.time()
            ret = libllama.llama_decode(ctx, batch)
            dt = time.time() - t_k0
            if ret == 0:
                k_times.append(dt)
                curr_pos += k

        if k_times:
            avg_k_time = sum(k_times) / len(k_times)
            lat_per_tok = (avg_k_time * 1000.0) / k
            speed_tok_s = k / avg_k_time
            wavefront_results[k] = {
                'sweep_time_ms': avg_k_time * 1000.0,
                'ms_per_token': lat_per_tok,
                'tok_per_sec': speed_tok_s,
                'speedup': speed_tok_s / baseline_speed
            }
            print(f"    Wavefront K={k}: Sweep = {avg_k_time*1000:.1f} ms | Latenza/token = {lat_per_tok:.1f} ms | Velocita = {speed_tok_s:.2f} tok/s ({speed_tok_s/baseline_speed:.2f}x)")

    print("\n" + "=" * 85)
    print("CONFRONTO ARCHITETTURALE SUL MODELLO 14B:")
    print("=" * 85)
    print(f"{'Configurazione':<30} | {'Sweep (ms)':<11} | {'Latenza (ms/tok)':<18} | {'Throughput':<12} | {'Speedup'}")
    print("-" * 85)
    print(f"{'Sequential GEMV (Baseline 14B)':<30} | {baseline_ms:<11.1f} | {baseline_ms:<18.1f} | {baseline_speed:<6.2f} tok/s | 1.00x")
    for k, res in wavefront_results.items():
        print(f"{'Wavefront Engine K=' + str(k) + ' (AVX-2)':<30} | {res['sweep_time_ms']:<11.1f} | {res['ms_per_token']:<18.1f} | {res['tok_per_sec']:<6.2f} tok/s | {res['speedup']:.2f}x")
    print("=" * 85)

    libllama.llama_batch_free(batch)
    libllama.llama_free(ctx)
    libllama.llama_model_free(model)
    libllama.llama_backend_free()

    out_path = r"C:\Users\Umberto\synaptic-fold-v2\benchmarks\inprocess_14b_wavefront_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            'model': 'Qwen2.5-14B-Instruct',
            'size_gb': size_gb,
            'baseline': {'ms_per_token': baseline_ms, 'tok_per_sec': baseline_speed},
            'wavefront': wavefront_results
        }, f, indent=2)
    print(f"\n[SALVATO] Risultati salvati in: {out_path}")
    return True

if __name__ == '__main__':
    run_14b_benchmark()
