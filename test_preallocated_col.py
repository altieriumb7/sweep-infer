import ctypes
import os
import sys
import time
import numpy as np

DLL_PATH = r"C:\Users\Umberto\synaptic-fold-v2\synaptic_fold\kernels\synaptic_column_avx2.dll"
lib = ctypes.CDLL(DLL_PATH)

lib.synaptic_column_avx2_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
lib.synaptic_column_avx2_forward.restype = None

D = 4096
num_cols = 4
x_vec = np.ones(D, dtype=np.float32)
W_cent = np.ones((num_cols, D), dtype=np.float32) / np.sqrt(D)
b_rest = np.zeros(num_cols, dtype=np.float32)
out = np.zeros(num_cols, dtype=np.float32)

px = x_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
pW = W_cent.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
pb = b_rest.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
pout = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Warmup
lib.synaptic_column_avx2_forward(px, pW, pb, pout, D, num_cols, ctypes.c_float(0.0))

t0 = time.perf_counter()
N_REPS = 500
for _ in range(N_REPS):
    lib.synaptic_column_avx2_forward(px, pW, pb, pout, D, num_cols, ctypes.c_float(0.0))
dt = (time.perf_counter() - t0) / N_REPS * 1e6

print(f"Preallocated Column Kernel Latency: {dt:.2f} us (microsecondi)!")
