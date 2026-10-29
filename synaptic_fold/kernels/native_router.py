"""
SynapticFold: High-Performance AVX-2 Assembly Router.

Provides Python bindings for compiled x86-64 AVX-2 assembly kernels:
- synaptic_column_avx2.dll (Cortical Minicolumn Activation)
- synaptic_tree_avx2.dll (Parallel Hypothesis Tree Branching Speculation)
"""

import os
import platform
import ctypes
import numpy as np
from typing import Tuple

_DIR = os.path.dirname(__file__)
_EXT = ".dll" if platform.system() == "Windows" else ".so"
_DLL_PATH = os.path.join(_DIR, f"synaptic_column_avx2{_EXT}")
_TREE_DLL_PATH = os.path.join(_DIR, f"synaptic_tree_avx2{_EXT}")
_TOPK_DLL_PATH = os.path.join(_DIR, f"synaptic_entropy_topk_avx2{_EXT}")
_ROUTER_DLL_PATH = os.path.join(_DIR, f"synaptic_ngram_router_avx2{_EXT}")

_KERNEL_LIB = None
_TREE_LIB = None
_TOPK_LIB = None
_ROUTER_LIB = None

if os.path.exists(_DLL_PATH):
    try:
        _KERNEL_LIB = ctypes.CDLL(_DLL_PATH)
        _KERNEL_LIB.synaptic_column_avx2_forward.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        _KERNEL_LIB.synaptic_column_avx2_forward.restype = None
    except Exception as e:
        _KERNEL_LIB = None

if os.path.exists(_TREE_DLL_PATH):
    try:
        _TREE_LIB = ctypes.CDLL(_TREE_DLL_PATH)
        _TREE_LIB.synaptic_tree_avx2_forward.argtypes = [
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
        _TREE_LIB.synaptic_tree_avx2_forward.restype = None
    except Exception as e:
        _TREE_LIB = None

if os.path.exists(_TOPK_DLL_PATH):
    try:
        _TOPK_LIB = ctypes.CDLL(_TOPK_DLL_PATH)
        _TOPK_LIB.synaptic_topk_entropy_avx2.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        _TOPK_LIB.synaptic_topk_entropy_avx2.restype = None
    except Exception as e:
        _TOPK_LIB = None

if os.path.exists(_ROUTER_DLL_PATH):
    try:
        _ROUTER_LIB = ctypes.CDLL(_ROUTER_DLL_PATH)
        _ROUTER_LIB.synaptic_ngram_router_init.argtypes = []
        _ROUTER_LIB.synaptic_ngram_router_init.restype = None
        _ROUTER_LIB.synaptic_ngram_router_insert.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        _ROUTER_LIB.synaptic_ngram_router_insert.restype = None
        _ROUTER_LIB.synaptic_ngram_router_query.argtypes = [ctypes.c_int32, ctypes.c_int32]
        _ROUTER_LIB.synaptic_ngram_router_query.restype = ctypes.c_int32
        _ROUTER_LIB.synaptic_ngram_router_seed.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
        _ROUTER_LIB.synaptic_ngram_router_seed.restype = None
    except Exception as e:
        _ROUTER_LIB = None


def is_native_avx2_available() -> bool:
    """Returns True if the column AVX-2 assembly kernel is loaded."""
    return _KERNEL_LIB is not None


def is_native_tree_avx2_available() -> bool:
    """Returns True if the tree speculation AVX-2 assembly kernel is loaded."""
    return _TREE_LIB is not None


def is_native_topk_avx2_available() -> bool:
    """Returns True if the vectorized top-k entropy AVX-2 kernel is loaded."""
    return _TOPK_LIB is not None


def is_native_router_avx2_available() -> bool:
    """Returns True if the CRC32 associative hardware token router kernel is loaded."""
    return _ROUTER_LIB is not None


def init_ngram_router_avx2():
    """Initializes the fast CRC32 n-gram router hash table."""
    if _ROUTER_LIB is not None:
        _ROUTER_LIB.synaptic_ngram_router_init()


def seed_ngram_router_avx2(tokens: list):
    """Bulk seeds the n-gram router hash table with sequence tokens."""
    if _ROUTER_LIB is not None and len(tokens) >= 3:
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        _ROUTER_LIB.synaptic_ngram_router_seed(arr, len(tokens))


def insert_ngram_router_avx2(t_prev: int, t_curr: int, t_next: int):
    """Inserts a trigram transition into the CRC32 router hash table."""
    if _ROUTER_LIB is not None:
        _ROUTER_LIB.synaptic_ngram_router_insert(t_prev, t_curr, t_next)


def query_ngram_router_avx2(t_prev: int, t_curr: int) -> int:
    """Queries the successor token in < 10 ns via CRC32 hardware instructions."""
    if _ROUTER_LIB is not None:
        return _ROUTER_LIB.synaptic_ngram_router_query(t_prev, t_curr)
    return -1


def compute_column_potentials_avx2(
    x: np.ndarray,
    W_centroids: np.ndarray,
    b_resting: np.ndarray,
    v_threshold: float = 0.0,
) -> np.ndarray:
    """
    Computes cortical minicolumn activation potentials using compiled AVX-2 assembly.
    
    Parameters:
    - x: (D,) float32 vector
    - W_centroids: (num_columns, D) float32 matrix
    - b_resting: (num_columns,) float32 vector
    - v_threshold: float activation threshold
    
    Returns:
    - out_potentials: (num_columns,) float32 vector
    """
    assert x.dtype == np.float32 and W_centroids.dtype == np.float32
    num_columns, D = W_centroids.shape
    
    if _KERNEL_LIB is not None:
        out = np.zeros(num_columns, dtype=np.float32)
        _KERNEL_LIB.synaptic_column_avx2_forward(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            W_centroids.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            b_resting.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            D,
            num_columns,
            ctypes.c_float(v_threshold),
        )
        return out
    else:
        # Fallback to NumPy
        dot = np.dot(W_centroids, x) + b_resting
        return np.where(dot >= v_threshold, dot, 0.0).astype(np.float32)


def evaluate_speculative_tree_avx2(
    node_latents: np.ndarray,
    branch_weights: np.ndarray,
    branch_biases: np.ndarray,
    parent_indices: np.ndarray,
    prob_threshold: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Evaluates speculative hypothesis trees using compiled native AVX-2 assembly.
    
    Parameters:
    - node_latents: (N, D) float32 array
    - branch_weights: (N, D) float32 array
    - branch_biases: (N,) float32 array
    - parent_indices: (N,) int32 array (-1 for root)
    - prob_threshold: pruning threshold for cumulative path probability
    
    Returns:
    - out_path_scores: (N,) float32 cumulative path scores
    - out_selected_nodes: (4,) int32 indices of top-4 selected nodes
    - out_causal_mask: (4, 4) float32 2D causal attention mask
    - num_selected: int count of selected surviving nodes
    """
    assert node_latents.dtype == np.float32 and branch_weights.dtype == np.float32
    num_nodes, D = node_latents.shape
    parent_indices = parent_indices.astype(np.int32)
    branch_biases = branch_biases.astype(np.float32)

    out_scores = np.zeros(num_nodes, dtype=np.float32)
    out_selected = np.full(4, -1, dtype=np.int32)
    out_mask = np.zeros((4, 4), dtype=np.float32)
    out_num_sel = ctypes.c_int(0)

    if _TREE_LIB is not None:
        _TREE_LIB.synaptic_tree_avx2_forward(
            node_latents.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            branch_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            branch_biases.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            parent_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            num_nodes,
            D,
            ctypes.c_float(prob_threshold),
            out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_selected.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(out_num_sel),
        )
        return out_scores, out_selected, out_mask, out_num_sel.value
    else:
        # Fallback to pure Python reference
        logits = np.sum(node_latents * branch_weights, axis=-1) + branch_biases
        scores = np.zeros(num_nodes, dtype=np.float32)
        scores[0] = 1.0

        for i in range(1, num_nodes):
            P = int(parent_indices[i])
            if P < 0 or scores[P] <= 0.0:
                scores[i] = 0.0
                continue
            siblings = [k for k in range(1, num_nodes) if parent_indices[k] == P]
            if len(siblings) > 1:
                sib_logits = logits[siblings]
                max_z = np.max(sib_logits)
                exp_z = np.exp(sib_logits - max_z)
                p_i = exp_z[siblings.index(i)] / np.sum(exp_z)
            else:
                p_i = 1.0 / (1.0 + np.exp(-logits[i]))
            scores[i] = scores[P] * p_i

        scores = np.where(scores >= prob_threshold, scores, 0.0).astype(np.float32)
        for i in range(1, num_nodes):
            P = int(parent_indices[i])
            if P >= 0 and scores[P] == 0.0:
                scores[i] = 0.0

        num_sel = 0
        if scores[0] > 0.0:
            out_selected[0] = 0
            num_sel = 1
        rem = [(k, scores[k]) for k in range(1, num_nodes) if scores[k] > 0.0]
        rem.sort(key=lambda x: x[1], reverse=True)
        for k, _ in rem:
            if num_sel >= 4:
                break
            out_selected[num_sel] = k
            num_sel += 1

        for r in range(num_sel):
            u = out_selected[r]
            for c in range(num_sel):
                v = out_selected[c]
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
                    out_mask[r, c] = 1.0

        return scores, out_selected, out_mask, num_sel


def compute_topk_entropy_avx2(
    logits_ptr_or_arr,
    vocab_size: int = 128256,
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Computes top-k indices, normalized probabilities, Shannon entropy H, and confidence margin
    directly in native AVX-2 assembly.
    
    Parameters:
    - logits_ptr_or_arr: pointer to float (ctypes) or 1D numpy array of float32
    - vocab_size: total vocabulary size (128,256)
    - k: number of top candidates (default 5)
    
    Returns:
    - top_indices: (k,) int32
    - top_probs: (k,) float32
    - entropy_bits: float Shannon entropy in bits
    - confidence_margin: float margin p[0] - p[1]
    """
    out_indices = np.zeros(k, dtype=np.int32)
    out_probs = np.zeros(k, dtype=np.float32)
    out_h = ctypes.c_float(0.0)
    out_m = ctypes.c_float(0.0)

    if _TOPK_LIB is not None:
        if isinstance(logits_ptr_or_arr, np.ndarray):
            p_logits = logits_ptr_or_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        elif isinstance(logits_ptr_or_arr, ctypes.POINTER(ctypes.c_float)):
            p_logits = logits_ptr_or_arr
        elif isinstance(logits_ptr_or_arr, int):
            p_logits = ctypes.cast(logits_ptr_or_arr, ctypes.POINTER(ctypes.c_float))
        else:
            p_logits = ctypes.cast(logits_ptr_or_arr, ctypes.POINTER(ctypes.c_float))

        _TOPK_LIB.synaptic_topk_entropy_avx2(
            p_logits,
            vocab_size,
            k,
            out_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(out_h),
            ctypes.byref(out_m),
        )
        return out_indices, out_probs, float(out_h.value), float(out_m.value)
    else:
        # Fallback to NumPy
        if not isinstance(logits_ptr_or_arr, np.ndarray):
            arr = np.ctypeslib.as_array(logits_ptr_or_arr, shape=(vocab_size,))
        else:
            arr = logits_ptr_or_arr
        top_k_idx = np.argpartition(arr, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(-arr[top_k_idx])]
        top_l = arr[top_k_idx]
        shift_l = top_l - np.max(top_l)
        exp_l = np.exp(shift_l)
        probs = exp_l / np.sum(exp_l)
        H = float(-np.sum(probs * np.log2(probs + 1e-12)))
        margin = float(probs[0] - probs[1]) if k > 1 else 1.0
        return top_k_idx.astype(np.int32), probs.astype(np.float32), H, margin


