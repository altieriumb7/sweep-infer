import ctypes
import os
import sys
import time
import numpy as np

DLL_PATH = r"C:\Users\Umberto\synaptic-fold-v2\synaptic_fold\kernels\synaptic_entropy_topk_avx2.dll"
lib = ctypes.CDLL(DLL_PATH)

lib.synaptic_topk_entropy_avx2.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
lib.synaptic_topk_entropy_avx2.restype = None

vocab_size = 128256
k = 5
logits = np.random.randn(vocab_size).astype(np.float32)
p_logits = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

out_indices = (ctypes.c_int32 * k)()
out_probs = (ctypes.c_float * k)()
out_h = ctypes.c_float(0.0)
out_m = ctypes.c_float(0.0)

p_idx = out_indices
p_prb = out_probs
p_h = ctypes.byref(out_h)
p_m = ctypes.byref(out_m)

# Warmup
lib.synaptic_topk_entropy_avx2(p_logits, vocab_size, k, p_idx, p_prb, p_h, p_m)

t0 = time.perf_counter()
N_REPS = 500
for _ in range(N_REPS):
    lib.synaptic_topk_entropy_avx2(p_logits, vocab_size, k, p_idx, p_prb, p_h, p_m)
dt = (time.perf_counter() - t0) / N_REPS * 1e6

print(f"Preallocated Top-K Entropy Kernel Latency: {dt:.2f} us (microsecondi)!")
