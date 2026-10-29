import ctypes
import numpy as np
import time
import os

dir_path = r"C:\Users\Umberto\synaptic-fold-v2\synaptic_fold\kernels"
orig_dll_path = os.path.join(dir_path, "synaptic_column_avx2.dll")
opt_dll_path = os.path.join(dir_path, "synaptic_column_avx2_opt.dll")

orig_dll = ctypes.CDLL(orig_dll_path)
opt_dll = ctypes.CDLL(opt_dll_path)

# ABI: (x, W, b, out, D, num_columns, v_threshold)
for dll in [orig_dll, opt_dll]:
    dll.synaptic_column_avx2_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float
    ]
    dll.synaptic_column_avx2_forward.restype = None

D = 3584          # Feature dimension (Qwen2/Llama)
C = 64            # 64 cortical columns
v_thresh = 0.5    # Membrane threshold

np.random.seed(42)
x = np.random.randn(D).astype(np.float32)
W = np.random.randn(C, D).astype(np.float32)
b = np.random.randn(C).astype(np.float32)

out_orig = np.zeros(C, dtype=np.float32)
out_opt = np.zeros(C, dtype=np.float32)

x_p = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
W_p = W.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
b_p = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
out_orig_p = out_orig.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
out_opt_p = out_opt.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# 1. Correctness check
orig_dll.synaptic_column_avx2_forward(x_p, W_p, b_p, out_orig_p, D, C, ctypes.c_float(v_thresh))
opt_dll.synaptic_column_avx2_forward(x_p, W_p, b_p, out_opt_p, D, C, ctypes.c_float(v_thresh))

# Ground truth numpy
raw_pot = W @ x + b
np_gt = np.where(raw_pot > v_thresh, raw_pot, 0.0).astype(np.float32)

max_diff_orig = np.max(np.abs(out_orig - np_gt))
max_diff_opt = np.max(np.abs(out_opt - np_gt))
max_diff_between = np.max(np.abs(out_orig - out_opt))

print(f"Max Diff Orig vs NumPy: {max_diff_orig:.6e}")
print(f"Max Diff Opt vs NumPy:  {max_diff_opt:.6e}")
print(f"Max Diff Opt vs Orig:   {max_diff_between:.6e}")

# 2. Benchmark latency (1000 warmup, 5000 iterations)
N_ITER = 5000

# Warmup
for _ in range(1000):
    orig_dll.synaptic_column_avx2_forward(x_p, W_p, b_p, out_orig_p, D, C, ctypes.c_float(v_thresh))
    opt_dll.synaptic_column_avx2_forward(x_p, W_p, b_p, out_opt_p, D, C, ctypes.c_float(v_thresh))

# Measure Original
t0 = time.perf_counter()
for _ in range(N_ITER):
    orig_dll.synaptic_column_avx2_forward(x_p, W_p, b_p, out_orig_p, D, C, ctypes.c_float(v_thresh))
t_orig = (time.perf_counter() - t0) / N_ITER * 1e6

# Measure Optimized
t0 = time.perf_counter()
for _ in range(N_ITER):
    opt_dll.synaptic_column_avx2_forward(x_p, W_p, b_p, out_opt_p, D, C, ctypes.c_float(v_thresh))
t_opt = (time.perf_counter() - t0) / N_ITER * 1e6

# Measure NumPy
t0 = time.perf_counter()
for _ in range(N_ITER):
    _ = np.where((W @ x + b) > v_thresh, (W @ x + b), 0.0)
t_np = (time.perf_counter() - t0) / N_ITER * 1e6

print("\n" + "=" * 70)
print(f"BENCHMARK CONFRONTO KERNEL ASSEMBLY AVX-2 ({C} Colonne, D={D}):")
print("=" * 70)
print(f"NumPy C-BLAS:                      {t_np:>6.2f} us (1.00x)")
print(f"Originale (synaptic_column_avx2): {t_orig:>6.2f} us ({t_np/t_orig:>5.2f}x vs NumPy)")
print(f"Ottimizzato 4-Way (avx2_opt):     {t_opt:>6.2f} us ({t_np/t_opt:>5.2f}x vs NumPy, {t_orig/t_opt:>4.2f}x vs Orig!)")
print("=" * 70)
