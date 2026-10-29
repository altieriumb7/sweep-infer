import ctypes
import os
import sys
import time
import numpy as np

DLL_PATH = r"C:\Users\Umberto\synaptic-fold-v2\synaptic_fold\kernels\synaptic_ngram_router_avx2.dll"
assert os.path.exists(DLL_PATH), f"DLL not found: {DLL_PATH}"

lib = ctypes.CDLL(DLL_PATH)
lib.synaptic_ngram_router_init.argtypes = []
lib.synaptic_ngram_router_init.restype = None

lib.synaptic_ngram_router_insert.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
lib.synaptic_ngram_router_insert.restype = None

lib.synaptic_ngram_router_query.argtypes = [ctypes.c_int32, ctypes.c_int32]
lib.synaptic_ngram_router_query.restype = ctypes.c_int32

lib.synaptic_ngram_router_seed.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
lib.synaptic_ngram_router_seed.restype = None

def test_ngram_router():
    lib.synaptic_ngram_router_init()
    
    # Insert some bigrams
    lib.synaptic_ngram_router_insert(100, 200, 300)
    lib.synaptic_ngram_router_insert(200, 300, 400)
    lib.synaptic_ngram_router_insert(400, 500, 600)

    # Test queries
    q1 = lib.synaptic_ngram_router_query(100, 200)
    q2 = lib.synaptic_ngram_router_query(200, 300)
    q_miss = lib.synaptic_ngram_router_query(999, 888)

    assert q1 == 300, f"Expected 300, got {q1}"
    assert q2 == 400, f"Expected 400, got {q2}"
    assert q_miss == -1, f"Expected -1, got {q_miss}"

    # Benchmark query latency over 1,000,000 queries
    N_QUERIES = 500000
    t0 = time.perf_counter()
    for _ in range(N_QUERIES):
        lib.synaptic_ngram_router_query(100, 200)
    dt_ns = (time.perf_counter() - t0) / N_QUERIES * 1e9

    print(f"CRC32 Hardware Router Latency: {dt_ns:.2f} ns (nanosecondi per query)!")
    print(f"Throughput: {1e9 / dt_ns:,.0f} queries/second!")
    print("ALL ROUTER ASSERTIONS PASSED!")

if __name__ == '__main__':
    test_ngram_router()
