import ctypes
import os
import sys
import time
import numpy as np

DLL_PATH = r"C:\Users\Umberto\synaptic-fold-v2\synaptic_fold\kernels\synaptic_tree_avx2.dll"
lib = ctypes.CDLL(DLL_PATH)

lib.synaptic_tree_avx2_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
]
lib.synaptic_tree_avx2_forward.restype = None

# Preallocated buffers
D = 4096
N = 5
node_latents = np.ones((N, D), dtype=np.float32)
branch_weights = np.ones((N, D), dtype=np.float32) / np.sqrt(D)
branch_biases = np.zeros(N, dtype=np.float32)
parent_indices = np.array([-1, 0, 0, 1, 2], dtype=np.int32)

p_lat = node_latents.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
p_w = branch_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
p_b = branch_biases.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
p_pi = parent_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

out_scores = np.zeros(N, dtype=np.float32)
out_selected = np.zeros(4, dtype=np.int32)
out_mask = np.zeros((4, 4), dtype=np.float32)
num_sel = ctypes.c_int(0)

p_scores = out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
p_sel = out_selected.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
p_mask = out_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
p_nsel = ctypes.byref(num_sel)

# Warmup
lib.synaptic_tree_avx2_forward(p_lat, p_w, p_b, p_pi, N, D, ctypes.c_float(0.15), p_scores, p_sel, p_mask, p_nsel)

# Benchmark 500 calls
t0 = time.perf_counter()
N_REPS = 500
for _ in range(N_REPS):
    lib.synaptic_tree_avx2_forward(p_lat, p_w, p_b, p_pi, N, D, ctypes.c_float(0.15), p_scores, p_sel, p_mask, p_nsel)
dt = (time.perf_counter() - t0) / N_REPS * 1e6

print(f"Preallocated Native Call Latency: {dt:.2f} us (microsecondi)!")
