import ctypes
import os
import sys
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = '/workspace/sweep'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

libcudart = ctypes.CDLL('libcudart.so')

libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
libcudart.cudaMalloc.restype = ctypes.c_int
libcudart.cudaFree.argtypes = [ctypes.c_void_p]
libcudart.cudaFree.restype = ctypes.c_int
libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libcudart.cudaMemcpy.restype = ctypes.c_int
libcudart.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
libcudart.cudaEventCreate.restype = ctypes.c_int
libcudart.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libcudart.cudaEventRecord.restype = ctypes.c_int
libcudart.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
libcudart.cudaEventSynchronize.restype = ctypes.c_int
libcudart.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
libcudart.cudaEventElapsedTime.restype = ctypes.c_int

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

CUDA_OPT_LIB_PATH = '/workspace/sweep/synaptic_fold/kernels/synaptic_entropy_topk_cuda_opt.so'
cuda_lib = ctypes.CDLL(CUDA_OPT_LIB_PATH)

cuda_lib.launch_sweep_gpu_entropy_opt.argtypes = [
    ctypes.c_void_p,             # d_logits
    ctypes.c_int,                # vocab_size
    ctypes.c_int,                # k
    ctypes.c_float,              # entropy_threshold
    ctypes.c_float,              # margin_threshold
    ctypes.c_void_p,             # d_inter_vals
    ctypes.c_void_p,             # d_inter_idxs
    ctypes.c_void_p,             # d_out_indices
    ctypes.c_void_p,             # d_out_probs
    ctypes.c_void_p,             # d_out_entropy
    ctypes.c_void_p,             # d_out_margin
    ctypes.c_void_p,             # d_out_is_confident
    ctypes.c_void_p              # stream
]
cuda_lib.launch_sweep_gpu_entropy_opt.restype = None

print('=' * 80)
print('PROFILING KERNEL CUDA MULTI-SM GRID SU NVIDIA GEFORCE RTX 5070 Ti (BLACKWELL)')
print('Vocabolario Llama-3: V = 128,256 logits | K = 16 | Top-K Softmax + Shannon Entropy')
print('=' * 80)

V = 128256
K = 16
NUM_BLOCKS = 64
entropy_thresh = 0.55
margin_thresh = 0.60

h_logits = np.random.randn(V).astype(np.float32)
h_logits[42] = 18.5
h_logits[1337] = 12.2

h_indices = np.zeros(K, dtype=np.int32)
h_probs = np.zeros(K, dtype=np.float32)
h_entropy = np.zeros(1, dtype=np.float32)
h_margin = np.zeros(1, dtype=np.float32)
h_confident = np.zeros(1, dtype=np.int32)

d_logits = ctypes.c_void_p()
d_inter_vals = ctypes.c_void_p()
d_inter_idxs = ctypes.c_void_p()
d_indices = ctypes.c_void_p()
d_probs = ctypes.c_void_p()
d_entropy = ctypes.c_void_p()
d_margin = ctypes.c_void_p()
d_confident = ctypes.c_void_p()

libcudart.cudaMalloc(ctypes.byref(d_logits), V * 4)
libcudart.cudaMalloc(ctypes.byref(d_inter_vals), NUM_BLOCKS * K * 4)
libcudart.cudaMalloc(ctypes.byref(d_inter_idxs), NUM_BLOCKS * K * 4)
libcudart.cudaMalloc(ctypes.byref(d_indices), K * 4)
libcudart.cudaMalloc(ctypes.byref(d_probs), K * 4)
libcudart.cudaMalloc(ctypes.byref(d_entropy), 4)
libcudart.cudaMalloc(ctypes.byref(d_margin), 4)
libcudart.cudaMalloc(ctypes.byref(d_confident), 4)

libcudart.cudaMemcpy(d_logits, h_logits.ctypes.data, V * 4, cudaMemcpyHostToDevice)

start_event = ctypes.c_void_p()
stop_event = ctypes.c_void_p()
libcudart.cudaEventCreate(ctypes.byref(start_event))
libcudart.cudaEventCreate(ctypes.byref(stop_event))

# Warmup (100 runs)
for _ in range(100):
    cuda_lib.launch_sweep_gpu_entropy_opt(
        d_logits, V, K, entropy_thresh, margin_thresh,
        d_inter_vals, d_inter_idxs,
        d_indices, d_probs, d_entropy, d_margin, d_confident, None
    )
libcudart.cudaEventRecord(start_event, None)
libcudart.cudaEventSynchronize(start_event)

# Benchmark 5000 iterations
N_ITERS = 5000
libcudart.cudaEventRecord(start_event, None)
for _ in range(N_ITERS):
    cuda_lib.launch_sweep_gpu_entropy_opt(
        d_logits, V, K, entropy_thresh, margin_thresh,
        d_inter_vals, d_inter_idxs,
        d_indices, d_probs, d_entropy, d_margin, d_confident, None
    )
libcudart.cudaEventRecord(stop_event, None)
libcudart.cudaEventSynchronize(stop_event)

elapsed_ms = ctypes.c_float()
libcudart.cudaEventElapsedTime(ctypes.byref(elapsed_ms), start_event, stop_event)

avg_latency_us = (elapsed_ms.value / N_ITERS) * 1000.0

libcudart.cudaMemcpy(h_indices.ctypes.data, d_indices, K * 4, cudaMemcpyDeviceToHost)
libcudart.cudaMemcpy(h_probs.ctypes.data, d_probs, K * 4, cudaMemcpyDeviceToHost)
libcudart.cudaMemcpy(h_entropy.ctypes.data, d_entropy, 4, cudaMemcpyDeviceToHost)
libcudart.cudaMemcpy(h_margin.ctypes.data, d_margin, 4, cudaMemcpyDeviceToHost)
libcudart.cudaMemcpy(h_confident.ctypes.data, d_confident, 4, cudaMemcpyDeviceToHost)

print(f'\n[RISULTATI HARDWARE RTX 5070 Ti]')
print(f'  Latenza Media Kernel CUDA In-VRAM (Grid Multi-SM): {avg_latency_us:.3f} us ({avg_latency_us * 1000.0:.1f} ns)')
print(f'  Top-1 Token: {h_indices[0]} (Prob = {h_probs[0]:.4f})')
print(f'  Top-2 Token: {h_indices[1]} (Prob = {h_probs[1]:.4f})')
print(f'  Entropia di Shannon calcolata in SRAM: {h_entropy[0]:.4f} bit')
print(f'  Margine di Confidenza (Delta): {h_margin[0]:.4f}')
print(f'  Gating Speculativo Wavefront: {"ATTIVATO (CONFIDENT)" if h_confident[0] == 1 else "DISATTIVATO"}')

libcudart.cudaFree(d_logits)
libcudart.cudaFree(d_inter_vals)
libcudart.cudaFree(d_inter_idxs)
libcudart.cudaFree(d_indices)
libcudart.cudaFree(d_probs)
libcudart.cudaFree(d_entropy)
libcudart.cudaFree(d_margin)
libcudart.cudaFree(d_confident)
