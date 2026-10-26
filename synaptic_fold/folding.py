"""
SynapticFold v2: Recurrent Cortical Folding & Chebyshev Depth Trajectory Merging.

Eliminates the "1-Layer Discard Paradox":
- Replaces naive single-middle-layer heuristic with Tri-Archetype Functional Compartmentalization
  (Early Syntactic, Mid Associative/Knowledge, Late Task/Reasoning).
- Implements Orthogonal Procrustes on Grassmannian manifolds.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn


def compute_linear_cka(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """
    Linear Centered Kernel Alignment (CKA) between two representation matrices.
    """
    if features_x.dim() > 2:
        features_x = features_x.reshape(-1, features_x.size(-1))
    if features_y.dim() > 2:
        features_y = features_y.reshape(-1, features_y.size(-1))

    x = features_x - features_x.mean(dim=0, keepdim=True)
    y = features_y - features_y.mean(dim=0, keepdim=True)

    dot_product_sim = torch.norm(torch.matmul(y.t(), x), p="fro") ** 2
    normalization_x = torch.norm(torch.matmul(x.t(), x), p="fro")
    normalization_y = torch.norm(torch.matmul(y.t(), y), p="fro")

    denom = normalization_x * normalization_y
    if denom.item() == 0.0:
        return 0.0
    return (dot_product_sim / denom).item()


def orthogonal_procrustes_alignment(
    source_weight: torch.Tensor,
    target_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Solves the Orthogonal Procrustes problem:
        min_{R in O(d)} || source_weight @ R - target_weight ||_F^2
    """
    correlation_matrix = torch.matmul(source_weight.t(), target_weight)
    u, _, vt = torch.linalg.svd(correlation_matrix.to(torch.float32))
    rotation = torch.matmul(u, vt)
    return rotation.to(source_weight.dtype)


def regmean_merge(
    weights: List[torch.Tensor],
    covariances: List[torch.Tensor],
    damping: float = 1e-4,
    prefer_diagonal: bool = False,
) -> torch.Tensor:
    """
    Closed-form covariance-weighted weight merging (RegMean v2 with trace damping).
    
    For PyTorch Linear weights W of shape (d_out, d_in) and input covariances Gamma of shape (d_in, d_in):
        W* = (sum_l W_l @ Gamma_l) @ (sum_l Gamma_l + lambda * I)^(-1)
    
    Automatically switches to O(d_out * d_in) Diagonal RegMean when d_in > 8192 to prevent
    multi-gigabyte memory allocations and cubic matrix inversion latencies.
    """
    if len(weights) != len(covariances) or len(weights) == 0:
        raise ValueError("Mismatched or empty weight and covariance lists")

    device = weights[0].device
    d_out, d_in = weights[0].shape
    dtype = torch.float32

    # High-dimensional fast path (e.g. SwiGLU down_proj with d_in=18944)
    if d_in > 8192 or prefer_diagonal:
        accum_diag = torch.zeros(d_in, dtype=dtype, device=device)
        accum_action = torch.zeros((d_out, d_in), dtype=dtype, device=device)
        for w, cov in zip(weights, covariances):
            w_f32 = w.to(dtype=dtype, device=device)
            cov_diag = torch.diag(cov).to(dtype=dtype, device=device) if cov.dim() == 2 else cov.to(dtype=dtype, device=device)
            accum_diag += cov_diag
            accum_action += w_f32 * cov_diag.unsqueeze(0)
        reg_val = damping * accum_diag.mean()
        denom = accum_diag.unsqueeze(0) + max(1e-5, reg_val.item())
        merged = accum_action / denom
        return merged.to(weights[0].dtype)

    accumulated_cov = torch.zeros((d_in, d_in), dtype=dtype, device=device)
    accumulated_action = torch.zeros((d_out, d_in), dtype=dtype, device=device)

    for w, cov in zip(weights, covariances):
        cov_f32 = cov.to(dtype=dtype, device=device)
        w_f32 = w.to(dtype=dtype, device=device)
        accumulated_cov += cov_f32
        accumulated_action += torch.matmul(w_f32, cov_f32)

    trace_damp = damping * (torch.trace(accumulated_cov) / max(1, d_in)).item()
    identity = torch.eye(d_in, dtype=dtype, device=device) * max(1e-5, trace_damp)
    regularized_cov = accumulated_cov + identity

    try:
        # Solve: regularized_cov^T @ (W*)^T = accumulated_action^T
        merged_weight = torch.linalg.solve(regularized_cov.t(), accumulated_action.t()).t()
    except RuntimeError:
        merged_weight = torch.matmul(accumulated_action, torch.linalg.pinv(regularized_cov))

    return merged_weight.to(weights[0].dtype)


class RecurrentCorticalFolder:
    """
    Partitions layer sequences into Sensory -> Tri-Archetype Recurrent Core -> Motor.
    """

    def __init__(self, split_start: int, split_end: int):
        if split_start >= split_end:
            raise ValueError("split_start must be strictly less than split_end")
        self.split_start = split_start
        self.split_end = split_end

    def partition_layers(self, total_layers: int) -> Dict[str, List[int]]:
        sensory = list(range(0, self.split_start))
        plateau = list(range(self.split_start, self.split_end))
        motor = list(range(self.split_end, total_layers))

        # Tri-archetype subdivision of plateau
        p_len = len(plateau)
        c1 = plateau[: p_len // 3]
        c2 = plateau[p_len // 3 : 2 * (p_len // 3)]
        c3 = plateau[2 * (p_len // 3) :]

        return {
            "sensory": sensory,
            "plateau": plateau,
            "archetype_early": c1,
            "archetype_mid": c2,
            "archetype_late": c3,
            "motor": motor,
        }
