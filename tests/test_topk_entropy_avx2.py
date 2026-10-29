import ctypes
import os
import sys
import time
import numpy as np

DLL_PATH = r"C:\Users\Umberto\synaptic-fold-v2\synaptic_fold\kernels\synaptic_entropy_topk_avx2.dll"
assert os.path.exists(DLL_PATH), f"DLL not found: {DLL_PATH}"

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

def test_topk_entropy():
    vocab_size = 128256
    k = 5
    np.random.seed(42)
    # Generate realistic logit distribution
    logits = np.random.randn(vocab_size).astype(np.float32)
    logits[1234] = 18.5
    logits[5678] = 16.2
    logits[9999] = 14.8
    logits[42] = 13.9
    logits[100] = 12.5

    # 1. NumPy reference implementation
    t0 = time.perf_counter()
    for _ in range(50):
        top_k_idx = np.argpartition(logits, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(-logits[top_k_idx])]
        top_l = logits[top_k_idx]
        shift_l = top_l - np.max(top_l)
        exp_l = np.exp(shift_l)
        probs_ref = exp_l / np.sum(exp_l)
        H_ref = float(-np.sum(probs_ref * np.log2(probs_ref + 1e-12)))
        margin_ref = float(probs_ref[0] - probs_ref[1])
    t_numpy = (time.perf_counter() - t0) / 50 * 1e6

    # 2. Native AVX-2 implementation
    out_indices = (ctypes.c_int32 * k)()
    out_probs = (ctypes.c_float * k)()
    out_h = ctypes.c_float(0.0)
    out_m = ctypes.c_float(0.0)

    p_logits = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Warmup
    lib.synaptic_topk_entropy_avx2(p_logits, vocab_size, k, out_indices, out_probs, ctypes.byref(out_h), ctypes.byref(out_m))

    t0 = time.perf_counter()
    N_REPS = 200
    for _ in range(N_REPS):
        lib.synaptic_topk_entropy_avx2(p_logits, vocab_size, k, out_indices, out_probs, ctypes.byref(out_h), ctypes.byref(out_m))
    t_avx2 = (time.perf_counter() - t0) / N_REPS * 1e6

    # Check correctness
    indices_native = list(out_indices)
    probs_native = list(out_probs)
    H_native = out_h.value
    margin_native = out_m.value

    print(f"NumPy Reference: Indices={list(top_k_idx)}, Probs={probs_ref[:3]}, H={H_ref:.4f}, Margin={margin_ref:.4f}")
    print(f"AVX-2 Native:    Indices={indices_native}, Probs={probs_native[:3]}, H={H_native:.4f}, Margin={margin_native:.4f}")
    print(f"NumPy Latency:   {t_numpy:.2f} us")
    print(f"AVX-2 Latency:   {t_avx2:.2f} us")
    print(f"Speedup:         {t_numpy / t_avx2:.2f}x faster than NumPy!")

    assert indices_native == list(top_k_idx), "Indices mismatch!"
    assert np.allclose(probs_native, probs_ref, atol=1e-4), "Probs mismatch!"
    assert abs(H_native - H_ref) < 1e-4, "Entropy mismatch!"
    assert abs(margin_native - margin_ref) < 1e-4, "Margin mismatch!"
    print("ALL ASSERTIONS PASSED! PERFECT NUMERICAL FIDELITY!")

if __name__ == '__main__':
    test_topk_entropy()
