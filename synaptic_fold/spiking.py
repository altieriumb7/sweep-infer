"""
SynapticFold v2: Neuromorphic Spiking Primitives.

Upgraded primitives eliminating:
1. O(1/T) rate-coding bottleneck via FS-Spike (Integer Multi-Spike) & exact residue reset.
2. Bilinear SwiGLU collapse via Dual-Trace Asymmetric Synaptic Gating (AMPA/NMDA).
3. 100x energy drift in WOSS via Dynamic Nucleus & Partition Normalization.
4. Broadcasting bug in fold_rmsnorm_affine.
5. Matrix ill-conditioning in OBS via Cholesky Column-by-Column calibration.
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateHeaviside(torch.autograd.Function):
    """
    FastSigmoid surrogate gradient for non-differentiable spiking threshold:
        S = (u >= V_th)
        dS/du = 1 / (1 + pi * |u - V_th|)^2
    """
    @staticmethod
    def forward(ctx, membrane: torch.Tensor, v_th: float):
        ctx.save_for_backward(membrane)
        ctx.v_th = v_th
        return (membrane >= v_th).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        membrane, = ctx.saved_tensors
        v_th = ctx.v_th
        delta = torch.abs(membrane - v_th)
        surrogate_grad = 1.0 / (1.0 + math.pi * delta) ** 2
        return grad_output * surrogate_grad, None


class SpikeIntegrateAndFire(nn.Module):
    """
    SynapticFold v2 Discrete-time Integrate-and-Fire with exact residue preservation:
        u[t] = beta * (u[t-1] - V_th * S[t-1]) + I[t]
        u[t] = max(V_rest, u[t])
    """

    def __init__(
        self,
        v_th: float = 1.0,
        beta: float = 0.95,
        v_rest: float = 0.0,
        multi_spike: bool = False,
    ):
        super().__init__()
        self.v_th = float(v_th)
        self.beta = float(beta)
        self.v_rest = float(v_rest)
        self.multi_spike = multi_spike

    def forward(
        self,
        current: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        last_spike: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = torch.full_like(current, self.v_rest)
        if last_spike is None:
            last_spike = torch.zeros_like(current)

        # Exact charge conservation: leak is applied to the post-reset residue
        membrane = self.beta * (state - self.v_th * last_spike) + current

        # Resting potential clamp prevents unbounded hyperpolarization / dead-time
        if self.v_rest is not None:
            membrane = torch.clamp(membrane, min=self.v_rest)

        if self.multi_spike:
            # FS-Spike: Integer multi-spike preserves high dynamic range with small T
            spike = torch.floor(torch.clamp(membrane, min=0.0) / self.v_th).to(current.dtype)
        else:
            spike = SurrogateHeaviside.apply(membrane, self.v_th)

        return spike, membrane


class TraceSwiGLU(nn.Module):
    """
    SynapticFold v2 Dual-Trace Asymmetric Synaptic Gating for SwiGLU.
    
    Synthesizes the non-monotonic curvature of SiLU(x @ W_gate) * (x @ W_up)
    using dual-trace fast/slow kinetics (AMPA vs NMDA inspired):
        Tr_fast[t] = lambda_fast * Tr_fast[t-1] + gate_spikes
        Tr_slow[t] = lambda_slow * Tr_slow[t-1] + gate_spikes
        gated_signal = ReLU(Tr_fast[t] - gamma * Tr_slow[t]) * up_spikes
    """

    def __init__(
        self,
        trace_fast: float = 0.70,
        trace_slow: float = 0.25,
        inhibition_gamma: float = 0.35,
    ):
        super().__init__()
        self.trace_fast = trace_fast
        self.trace_slow = trace_slow
        self.inhibition_gamma = inhibition_gamma

    def forward(
        self,
        gate_spikes: torch.Tensor,
        up_spikes: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if state is None:
            tr_fast = torch.zeros_like(gate_spikes)
            tr_slow = torch.zeros_like(gate_spikes)
        else:
            tr_fast, tr_slow = state

        updated_fast = self.trace_fast * tr_fast + gate_spikes
        updated_slow = self.trace_slow * tr_slow + gate_spikes

        # Asymmetric subtraction reconstructs the non-monotonic activation profile
        curvature = F.relu(updated_fast - self.inhibition_gamma * updated_slow)
        gated_signal = curvature * up_spikes

        return gated_signal, (updated_fast, updated_slow)


class WOSSAttention(nn.Module):
    """
    Winner-Oriented Spike Shift (WOSS) Attention v2 with Dynamic Nucleus & Partition Normalization.
    
    Eliminates the 100x vector energy drift by maintaining local partition normalization:
        Delta_tau_ij = - (Z_ij - Z_i_max) * log2(e)
        Alpha_ij = 2^(- Delta_tau_ij) * mask
        y_i = sum_j (Alpha_ij * V_j) / (sum_j Alpha_ij + eps)
    """

    def __init__(self, saliency_threshold: float = 4.6, top_k_nucleus: int = 64):
        super().__init__()
        self.saliency_threshold = saliency_threshold
        self.top_k_nucleus = top_k_nucleus
        self.log2_e = math.log2(math.e)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        scale = 1.0 / math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores + mask

        max_scores, _ = torch.max(scores, dim=-1, keepdim=True)
        delta_scores = scores - max_scores  # <= 0

        # Saliency thresholding in temporal domain
        saliency_mask = delta_scores >= (-self.saliency_threshold)

        # Dynamic Nucleus truncation prevents unbounded accumulation across long contexts
        seq_len = key.size(-2)
        if seq_len > self.top_k_nucleus:
            topk_vals, _ = torch.topk(delta_scores, k=self.top_k_nucleus, dim=-1)
            cutoff = topk_vals[..., -1:]
            saliency_mask = saliency_mask & (delta_scores >= cutoff)

        # Convert to powers-of-two spike weight
        # 2^(delta_scores * log2(e)) = e^(delta_scores)
        spike_weights = torch.exp(delta_scores) * saliency_mask.to(scores.dtype)

        # Partition Normalization preserves energy conservation
        partition_sum = torch.sum(spike_weights, dim=-1, keepdim=True) + 1e-6
        normalized_weights = spike_weights / partition_sum

        output = torch.matmul(normalized_weights, value)

        total_elements = scores.numel()
        active_elements = saliency_mask.sum().item()
        sparsity_ratio = 1.0 - (active_elements / max(1, total_elements))

        return output, sparsity_ratio


def fold_rmsnorm_affine(
    linear_weight: torch.Tensor,
    norm_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Absorbs the affine gain gamma of RMSNorm into an adjacent weight matrix.
    
    For y = (x * norm_weight) @ linear_weight.T, where linear_weight is [out_features, in_features]
    and norm_weight is [in_features], scales input columns:
        W_folded = linear_weight * norm_weight.view(1, -1)
    """
    if linear_weight.dim() != 2:
        raise ValueError("linear_weight must be a 2D matrix")
    in_features = linear_weight.size(1)
    if norm_weight.numel() != in_features:
        raise ValueError(f"norm_weight size {norm_weight.numel()} must match linear in_features {in_features}")
    return linear_weight * norm_weight.view(1, -1)


class OptimalBrainSpiking:
    """
    Closed-form column-by-column Cholesky calibration for zero-shot quantization.
    """

    def __init__(self, damping: float = 1e-4):
        self.damping = damping

    def calibrate_weights(
        self,
        weights: torch.Tensor,
        activations: torch.Tensor,
        grid_step: float,
    ) -> torch.Tensor:
        if activations.dim() > 2:
            activations = activations.reshape(-1, activations.size(-1))

        d_in = activations.size(1)
        cov = torch.matmul(activations.t(), activations) / activations.size(0)

        # Adaptive damping proportional to trace prevents singular matrix inversion
        adaptive_damping = self.damping * (torch.trace(cov) / max(1, d_in)).item()
        cov = cov + max(1e-5, adaptive_damping) * torch.eye(d_in, device=cov.device, dtype=cov.dtype)

        try:
            chol = torch.linalg.cholesky(cov.to(torch.float32))
            h_inv = torch.cholesky_inverse(chol).to(weights.dtype)
        except RuntimeError:
            h_inv = torch.linalg.pinv(cov.to(torch.float32)).to(weights.dtype)

        w_calib = weights.clone()
        d_out, d_in = weights.shape

        # Column-by-column Cholesky update
        for j in range(d_in):
            w_col = w_calib[:, j]
            q_col = torch.round(w_col / grid_step) * grid_step
            err = w_col - q_col
            w_calib[:, j] = q_col

            if j + 1 < d_in:
                h_jj = max(1e-8, h_inv[j, j].item())
                h_slice = h_inv[j, j + 1:].unsqueeze(0)  # [1, d_in - j - 1]
                w_calib[:, j + 1:] -= torch.matmul(err.unsqueeze(1) / h_jj, h_slice)

        return w_calib
