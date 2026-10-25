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
) -> torch.Tensor:
    """
    Closed-form covariance-weighted weight merging (RegMean v2 with trace damping):
        W* = (sum_l X_l^T X_l + lambda * I)^(-1) (sum_l X_l^T X_l @ W_l)
    """
    if len(weights) != len(covariances) or len(weights) == 0:
        raise ValueError("Mismatched or empty weight and covariance lists")

    device = weights[0].device
    dtype = torch.float32

    accumulated_cov = torch.zeros_like(covariances[0], dtype=dtype, device=device)
    accumulated_action = torch.zeros(
        (covariances[0].size(0), weights[0].size(1)),
        dtype=dtype,
        device=device,
    )

    for w, cov in zip(weights, covariances):
        cov_f32 = cov.to(dtype=dtype, device=device)
        w_f32 = w.to(dtype=dtype, device=device)
        accumulated_cov += cov_f32
        accumulated_action += torch.matmul(cov_f32, w_f32)

    dim = accumulated_cov.size(0)
    trace_damp = damping * (torch.trace(accumulated_cov) / max(1, dim)).item()
    identity = torch.eye(dim, dtype=dtype, device=device) * max(1e-5, trace_damp)
    regularized_cov = accumulated_cov + identity

    try:
        merged_weight = torch.linalg.solve(regularized_cov, accumulated_action)
    except RuntimeError:
        merged_weight = torch.matmul(torch.linalg.pinv(regularized_cov), accumulated_action)

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
