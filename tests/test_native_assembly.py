"""
Test di Validazione e Benchmark del Kernel Nativo Assembly AVX-2 (synaptic_column_avx2.dll)
Verifica correttezza numerica e velocita' di attivazione delle minicolonne corticali.
"""
import ctypes
import os
import sys
import time
import numpy as np

def test_native_assembly_kernel():
    dll_path = os.path.abspath("C:/Users/Umberto/synaptic-fold-v2/synaptic_fold/kernels/synaptic_column_avx2.dll")
    assert os.path.exists(dll_path), f"DLL non trovata in: {dll_path}"
    
    # Carica la DLL nativa compilata da NASM + Clang
    lib = ctypes.CDLL(dll_path)
    
    # Definizione firma funzione:
    # void synaptic_column_avx2_forward(
    #     const float* x, const float* W, const float* b, float* out,
    #     int D, int num_cols, float v_threshold
    # )
    kernel_fn = lib.synaptic_column_avx2_forward
    kernel_fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    kernel_fn.restype = None
    
    # Parametri test (simulazione dimensione hidden_size 3584, 32 minicolonne)
    D = 3584
    num_columns = 32
    v_thresh = 0.5
    
    np.random.seed(42)
    x = np.random.randn(D).astype(np.float32)
    W = np.random.randn(num_columns, D).astype(np.float32)
    b = np.random.randn(num_columns).astype(np.float32)
    out_native = np.zeros(num_columns, dtype=np.float32)
    
    # 1. Esecuzione Python / NumPy di riferimento
    t0_py = time.perf_counter()
    dot = np.dot(W, x) + b
    out_expected = np.where(dot >= v_thresh, dot, 0.0).astype(np.float32)
    t_py = time.perf_counter() - t0_py
    
    # 2. Esecuzione Assembly AVX-2 Nativo
    x_p = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    W_p = W.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_p = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_p = out_native.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    t0_asm = time.perf_counter()
    kernel_fn(x_p, W_p, b_p, out_p, D, num_columns, ctypes.c_float(v_thresh))
    t_asm = time.perf_counter() - t0_asm
    
    # Verifica errore numerico
    max_err = np.max(np.abs(out_expected - out_native))
    print("=" * 65)
    print("TEST NATIVO ASSEMBLY AVX-2 MINICOLONNE CORTICALI")
    print("=" * 65)
    print(f"Dimensioni: D={D}, Num Minicolonne={num_columns}")
    print(f"Errore massimo rispetto a NumPy: {max_err:.2e}")
    print(f"Tempo NumPy: {t_py*1e6:.2f} us")
    print(f"Tempo Assembly AVX-2: {t_asm*1e6:.2f} us")
    print(f"Attivazioni attive: {np.count_nonzero(out_native)} su {num_columns}")
    print("Output nativo (prime 5 colonne):", out_native[:5])
    print("Output atteso (prime 5 colonne):", out_expected[:5])
    
    assert max_err < 1e-4, f"Errore troppo elevato: {max_err}"
    print("TEST ASSEMBLY AVX-2 NATIVO: SUCCESSO COMPLETO (PASSED)!")
    print("=" * 65)

if __name__ == "__main__":
    test_native_assembly_kernel()
