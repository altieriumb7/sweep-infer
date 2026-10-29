"""
Test di Correttezza Numerica e Benchmark di Latenza:
Kernel Assembly AVX-2 Speculativo su Alberi di Ipotesi (synaptic_tree_avx2.dll)

Verifica:
1. Calcolo parallelo dot product e logit di transizione per albero asimmetrico (5 nodi, D=3584/4096).
2. Probabilita cumulative di ramo (Softmax su fratelli, Sigmoid su singole continuazioni).
3. Pruning vettoriale immediato (vcmpps/vandps) con soglia di probabilita.
4. Selezione top-4 nodi asimmetrici.
5. Generazione maschera causale 2D (4x4) per verifica single-pass nel modello 7B/8B.
6. Benchmark di latenza in microsecondi su CPU.
"""

import os
import sys
import time
import ctypes
import numpy as np

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DLL_PATH = os.path.abspath(
    os.path.join(_CURR_DIR, "..", "synaptic_fold", "kernels", "synaptic_tree_avx2.dll")
) if os.path.basename(_CURR_DIR) == "tests" else os.path.abspath(
    os.path.join(_CURR_DIR, "synaptic_fold", "kernels", "synaptic_tree_avx2.dll")
)

def load_assembly_kernel():
    assert os.path.exists(DLL_PATH), f"DLL not found at: {DLL_PATH}"
    lib = ctypes.CDLL(DLL_PATH)
    
    # Signature:
    # void synaptic_tree_avx2_forward(
    #     const float* node_latents,       // RCX
    #     const float* branch_weights,     // RDX
    #     const float* branch_biases,      // R8
    #     const int*   parent_indices,     // R9
    #     int          num_nodes,          // [RBP+48]
    #     int          D,                  // [RBP+56]
    #     float        prob_threshold,     // [RBP+64]
    #     float*       out_path_scores,    // [RBP+72]
    #     int*         out_selected_nodes, // [RBP+80]
    #     float*       out_causal_mask,    // [RBP+88]
    #     int*         out_num_selected    // [RBP+96]
    # );
    fn = lib.synaptic_tree_avx2_forward
    fn.argtypes = [
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
    fn.restype = None
    return fn


def python_reference_tree_eval(
    node_latents: np.ndarray,
    branch_weights: np.ndarray,
    branch_biases: np.ndarray,
    parent_indices: np.ndarray,
    prob_threshold: float,
):
    N, D = node_latents.shape
    logits = np.sum(node_latents * branch_weights, axis=-1) + branch_biases
    scores = np.zeros(N, dtype=np.float32)
    scores[0] = 1.0  # Root hypothesis path probability

    for i in range(1, N):
        P = int(parent_indices[i])
        if P < 0 or scores[P] <= 0.0:
            scores[i] = 0.0
            continue

        siblings = [k for k in range(1, N) if parent_indices[k] == P]
        if len(siblings) > 1:
            sib_logits = logits[siblings]
            max_z = np.max(sib_logits)
            exp_z = np.exp(sib_logits - max_z)
            probs = exp_z / np.sum(exp_z)
            sib_idx = siblings.index(i)
            p_i = probs[sib_idx]
        else:
            p_i = 1.0 / (1.0 + np.exp(-logits[i]))

        scores[i] = scores[P] * p_i

    # Pruning by threshold
    scores = np.where(scores >= prob_threshold, scores, 0.0).astype(np.float32)

    # Cascade downward pruning
    for i in range(1, N):
        P = int(parent_indices[i])
        if P >= 0 and scores[P] == 0.0:
            scores[i] = 0.0

    # Top-4 Selection
    selected = [-1, -1, -1, -1]
    num_selected = 0
    if scores[0] > 0.0:
        selected[0] = 0
        num_selected = 1

    remaining = [(k, scores[k]) for k in range(1, N) if scores[k] > 0.0]
    remaining.sort(key=lambda x: x[1], reverse=True)
    for k, _ in remaining:
        if num_selected >= 4:
            break
        selected[num_selected] = k
        num_selected += 1

    # 4x4 Causal Mask
    mask = np.zeros((4, 4), dtype=np.float32)
    for r in range(num_selected):
        u = selected[r]
        for c in range(num_selected):
            v = selected[c]
            curr = u
            is_anc = False
            while curr >= 0:
                if curr == v:
                    is_anc = True
                    break
                if curr == 0:
                    break
                curr = parent_indices[curr]
            if is_anc:
                mask[r, c] = 1.0

    return scores, selected, mask, num_selected


def test_synaptic_tree_avx2_assembly():
    kernel_fn = load_assembly_kernel()
    print("=" * 78)
    print("TEST AVX-2 NATIVO: PARALLEL TREE BRANCHING SPECULATION KERNEL")
    print("Target Arch: Llama-3-8B / Qwen2.5-7B (D=3584, 5 Nodi Albero Asimmetrico)")
    print("=" * 78)

    # =========================================================================
    # SCENARIO 1: Albero Asimmetrico con Pruning di un Ramo Debole (3 Nodi)
    # =========================================================================
    print("\n--- SCENARIO 1: Pruning di un Ramo Debole (Soglia = 0.15) ---")
    parent_indices = np.array([-1, 0, 0, 1, 2], dtype=np.int32)
    num_nodes = 5
    D = 3584
    prob_threshold = 0.15

    np.random.seed(42)
    node_latents = np.random.randn(num_nodes, D).astype(np.float32)
    node_latents /= np.linalg.norm(node_latents, axis=-1, keepdims=True)

    branch_weights = np.random.randn(num_nodes, D).astype(np.float32)
    branch_weights /= np.linalg.norm(branch_weights, axis=-1, keepdims=True)

    # Biases calibrati per favorire il ramo A e sfavorire il ramo B
    branch_biases = np.array([0.0, 1.8, -0.5, 1.2, -1.0], dtype=np.float32)

    py_scores, py_selected, py_mask, py_num_sel = python_reference_tree_eval(
        node_latents, branch_weights, branch_biases, parent_indices, prob_threshold
    )

    out_scores = np.zeros(num_nodes, dtype=np.float32)
    out_selected = np.zeros(4, dtype=np.int32)
    out_mask = np.zeros((4, 4), dtype=np.float32)
    out_num_sel = ctypes.c_int(0)

    p_latents = node_latents.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    p_weights = branch_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    p_biases = branch_biases.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    p_parents = parent_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    p_out_scores = out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    p_out_selected = out_selected.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    p_out_mask = out_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    kernel_fn(
        p_latents, p_weights, p_biases, p_parents,
        num_nodes, D, ctypes.c_float(prob_threshold),
        p_out_scores, p_out_selected, p_out_mask, ctypes.byref(out_num_sel)
    )

    print(f"Path Scores Assembly AVX-2:  {np.round(out_scores, 4)}")
    print(f"Path Scores NumPy Reference: {np.round(py_scores, 4)}")
    max_err_scores = np.max(np.abs(out_scores - py_scores))
    print(f"Errore Massimo Path Scores:  {max_err_scores:.2e}")
    assert max_err_scores < 2e-3, f"Errore scores troppo alto: {max_err_scores}"

    print(f"Nodi Selezionati Top-4 Assembly: {out_selected.tolist()} (num_sel={out_num_sel.value})")
    print(f"Nodi Selezionati Top-4 NumPy:    {py_selected} (num_sel={py_num_sel})")
    assert out_num_sel.value == py_num_sel
    assert np.array_equal(out_selected[:out_num_sel.value], py_selected[:py_num_sel])

    print("Maschera Causale 2D (4x4) Assembly:")
    print(out_mask)
    assert np.array_equal(out_mask, py_mask)
    print("Scenario 1: VERIFICATO CON SUCCESSO!")

    # =========================================================================
    # SCENARIO 2: Entrambi i Rami Attivi (4 Nodi Sopravvissuti, Maschera Asimmetrica)
    # =========================================================================
    print("\n--- SCENARIO 2: Entrambi i Rami Attivi (Top-4 Asimmetrico, Soglia = 0.05) ---")
    branch_biases_2 = np.array([0.0, 1.2, 0.9, 0.8, -2.5], dtype=np.float32)
    prob_threshold_2 = 0.05

    py_scores_2, py_selected_2, py_mask_2, py_num_sel_2 = python_reference_tree_eval(
        node_latents, branch_weights, branch_biases_2, parent_indices, prob_threshold_2
    )

    p_biases_2 = branch_biases_2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    kernel_fn(
        p_latents, p_weights, p_biases_2, p_parents,
        num_nodes, D, ctypes.c_float(prob_threshold_2),
        p_out_scores, p_out_selected, p_out_mask, ctypes.byref(out_num_sel)
    )

    print(f"Path Scores Assembly AVX-2:  {np.round(out_scores, 4)}")
    print(f"Path Scores NumPy Reference: {np.round(py_scores_2, 4)}")
    max_err_2 = np.max(np.abs(out_scores - py_scores_2))
    print(f"Errore Massimo:              {max_err_2:.2e}")
    assert max_err_2 < 2e-3

    print(f"Nodi Selezionati Top-4 Assembly: {out_selected.tolist()} (num_sel={out_num_sel.value})")
    print(f"Nodi Selezionati Top-4 NumPy:    {py_selected_2} (num_sel={py_num_sel_2})")
    assert out_num_sel.value == 4
    assert np.array_equal(out_selected, py_selected_2)

    print("Maschera Causale 2D (4x4) con Due Rami Concorrenti:")
    print(out_mask)
    assert np.array_equal(out_mask, py_mask_2)
    # Verifica che il nodo 2 (figlio B) non possa guardare il nodo 1 (figlio A)!
    idx_node_1 = list(out_selected).index(1)
    idx_node_2 = list(out_selected).index(2)
    assert out_mask[idx_node_2, idx_node_1] == 0.0, "Violazione causalita tra rami indipendenti!"
    assert out_mask[idx_node_1, idx_node_2] == 0.0, "Violazione causalita tra rami indipendenti!"
    print("Isolamento causale cross-branch confermato: M[child_B, child_A] == 0.0!")
    print("Scenario 2: VERIFICATO CON SUCCESSO!")

    # =========================================================================
    # BENCHMARK AD ALTA PRECISIONE
    # =========================================================================
    print("\n" + "=" * 78)
    print("BENCHMARK DI LATENZA (10.000 iterazioni)")
    print("=" * 78)

    for _ in range(500):
        kernel_fn(
            p_latents, p_weights, p_biases, p_parents,
            num_nodes, D, ctypes.c_float(prob_threshold),
            p_out_scores, p_out_selected, p_out_mask, ctypes.byref(out_num_sel)
        )

    num_iters = 10000
    t0 = time.perf_counter()
    for _ in range(num_iters):
        kernel_fn(
            p_latents, p_weights, p_biases, p_parents,
            num_nodes, D, ctypes.c_float(prob_threshold),
            p_out_scores, p_out_selected, p_out_mask, ctypes.byref(out_num_sel)
        )
    t_total = time.perf_counter() - t0
    lat_us = (t_total / num_iters) * 1e6
    evals_per_sec = num_iters / t_total

    t0_py = time.perf_counter()
    for _ in range(1000):
        python_reference_tree_eval(
            node_latents, branch_weights, branch_biases, parent_indices, prob_threshold
        )
    lat_py_us = ((time.perf_counter() - t0_py) / 1000) * 1e6

    print(f"Latenza Kernel Assembly AVX-2 (D=3584, Qwen2.5-7B): {lat_us:.3f} us / albero")
    print(f"Throughput Assembly AVX-2:                           {evals_per_sec:,.0f} alberi/sec")
    print(f"Latenza Implementazione NumPy:                       {lat_py_us:.3f} us")
    print(f"Speedup Assembly rispetto NumPy:                     {lat_py_us / lat_us:.1f}x")

    # Test con D=4096 (Llama-3-8B standard)
    D_8b = 4096
    latents_8b = np.random.randn(num_nodes, D_8b).astype(np.float32)
    weights_8b = np.random.randn(num_nodes, D_8b).astype(np.float32)
    p_lat8 = latents_8b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    p_w8 = weights_8b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    t0_8b = time.perf_counter()
    for _ in range(num_iters):
        kernel_fn(
            p_lat8, p_w8, p_biases, p_parents,
            num_nodes, D_8b, ctypes.c_float(prob_threshold),
            p_out_scores, p_out_selected, p_out_mask, ctypes.byref(out_num_sel)
        )
    lat_8b_us = ((time.perf_counter() - t0_8b) / num_iters) * 1e6
    print(f"Latenza Kernel Assembly AVX-2 (D=4096, Llama-3-8B):  {lat_8b_us:.3f} us / albero")
    print("=" * 78)
    print(">>> TUTTI I TEST ASSEMBLY AVX-2 COMPLETATI CON SUCCESSO! <<<")

if __name__ == "__main__":
    test_synaptic_tree_avx2_assembly()
