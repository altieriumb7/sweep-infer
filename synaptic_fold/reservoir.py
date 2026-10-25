"""
SynapticFold v2: Layerless Synaptic Reservoir & Zero-Scatter Indexed-GEMM.

Features:
1. Zero-Scatter Indexed-GEMM: delta_h = gathered_act @ (v_down[:, topk_idx])^T.
2. Elastic Synaptic Mobilization (Top-P Energy Spiking): dynamically allocates
   between <1% and 100% active synapses based on input complexity.
3. In-place attractor relaxation preserving representation Dirichlet energy.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReservoirConfig:
    """
    Configuration parameters for SynapticFold v2 Reservoir.
    """
    sparsity_mode: str = "elastic"      # 'elastic' (dynamic Top-P energy) or 'fixed' (Top-K)
    p_energy_mass: float = 0.90         # Fraction of total activation energy to preserve
    active_synapse_ratio: float = 0.30  # Baseline fallback for fixed mode
    spike_threshold: float = 0.0
    attractor_steps: int = 3
    attractor_gamma: float = 0.35
    convergence_tol: float = 1e-3
    lateral_inhibition: bool = True
    anti_collapse_beta: float = 0.05    # Dirichlet energy preservation anchor
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class GlobalSynapticPool(nn.Module):
    """
    Unified Associative Synaptic Memory Substrate with Elastic Mobilization & Zero-Scatter Indexed-GEMM.
    """

    def __init__(
        self,
        k_gate: torch.Tensor,
        u_up: torch.Tensor,
        v_down: torch.Tensor,
        layer_offsets: Optional[List[int]] = None,
        tau_depth: Optional[torch.Tensor] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # K: [N, d], U: [N, d], V: [d, N]
        self.k_gate = nn.Parameter(k_gate.to(device=device, dtype=dtype), requires_grad=False)
        self.u_up = nn.Parameter(u_up.to(device=device, dtype=dtype), requires_grad=False)
        self.v_down = nn.Parameter(v_down.to(device=device, dtype=dtype), requires_grad=False)

        self.total_synapses = self.k_gate.size(0)
        self.hidden_dim = self.k_gate.size(1)
        self.layer_offsets = layer_offsets
        if tau_depth is not None:
            self.register_buffer("tau_depth", tau_depth.to(device=device, dtype=dtype))
        else:
            self.tau_depth = None

    @property
    def memory_footprint_mb(self) -> float:
        bytes_total = (self.k_gate.numel() + self.u_up.numel() + self.v_down.numel()) * self.k_gate.element_size()
        return bytes_total / (1024 ** 2)

    def query_synapses(
        self,
        hidden_states: torch.Tensor,
        sparsity_mode: str = "elastic",
        p_energy_mass: float = 0.90,
        k_ratio: float = 0.30,
        spike_threshold: float = 0.0,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Queries associative memory via Elastic Mobilization and Zero-Scatter Indexed-GEMM.
        """
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            b, s, d = orig_shape
            h_flat = hidden_states.reshape(-1, d)
        else:
            h_flat = hidden_states
            b, s, d = h_flat.size(0), 1, h_flat.size(1)

        # 1. Concept detector activations
        zk = torch.matmul(h_flat, self.k_gate.t())  # [Tokens, N]
        zu = torch.matmul(h_flat, self.u_up.t())    # [Tokens, N]
        act = F.silu(zk) * zu                      # [Tokens, N]

        n_synapses = self.total_synapses
        abs_act = torch.abs(act)

        # 2. Elastic Synaptic Mobilization vs Fixed Top-K
        if sparsity_mode == "elastic":
            # Sort activations to compute cumulative energy spectrum
            sorted_vals, sorted_indices = torch.sort(abs_act, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_vals, dim=-1)
            total_mass = cumsum[:, -1:] + 1e-8
            energy_fraction = cumsum / total_mass

            # Determine minimal active synapses needed to reach p_energy_mass
            meets_mass = (energy_fraction >= p_energy_mass)
            first_idx = torch.argmax(meets_mass.to(torch.int32), dim=-1)
            k_active = int(first_idx.max().item()) + 1
            k_active = max(1, min(n_synapses, k_active))

            topk_idx = sorted_indices[:, :k_active]
            topk_vals = sorted_vals[:, :k_active]
        else:
            k_active = max(1, min(n_synapses, int(n_synapses * k_ratio)))
            topk_vals, topk_idx = torch.topk(abs_act, k=k_active, dim=-1)

        if spike_threshold > 0.0:
            threshold_mask = topk_vals >= spike_threshold
        else:
            threshold_mask = torch.ones_like(topk_vals, dtype=torch.bool)

        gathered_act = torch.gather(act, -1, topk_idx) * threshold_mask.to(act.dtype) # [Tokens, k_active]

        # 3. ZERO-SCATTER INDEXED-GEMM
        num_tokens = h_flat.size(0)
        if num_tokens == 1:
            active_indices = topk_idx[0]
            v_active = self.v_down[:, active_indices]  # [d, k_active]
            delta_h_flat = torch.matmul(gathered_act, v_active.t())  # [1, d]
        else:
            delta_h_flat = torch.zeros((num_tokens, d), dtype=h_flat.dtype, device=h_flat.device)
            for t_i in range(num_tokens):
                act_t = gathered_act[t_i:t_i+1] # [1, k_active]
                idx_t = topk_idx[t_i]          # [k_active]
                v_act = self.v_down[:, idx_t]  # [d, k_active]
                delta_h_flat[t_i:t_i+1] = torch.matmul(act_t, v_act.t())

        sparsity_ratio = 1.0 - (k_active / float(n_synapses))

        if hidden_states.dim() == 3:
            delta_h = delta_h_flat.view(b, s, d)
        else:
            delta_h = delta_h_flat

        return delta_h, sparsity_ratio, topk_idx


class ContinuousAttractorDynamics(nn.Module):
    """
    Continuous Cognitive Attractor Engine v2 with Dirichlet Energy Preservation.
    """

    def __init__(self, config: ReservoirConfig):
        super().__init__()
        self.config = config

    def step(
        self,
        current_state: torch.Tensor,
        relational_delta: torch.Tensor,
        synaptic_delta: torch.Tensor,
        initial_anchor: Optional[torch.Tensor] = None,
        step_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        gamma = step_scale if step_scale is not None else self.config.attractor_gamma
        total_update = relational_delta + synaptic_delta

        stepped_state = current_state + gamma * total_update

        # Anti-collapse anchor: injects conservative sensory residual
        if initial_anchor is not None and self.config.anti_collapse_beta > 0.0:
            stepped_state = stepped_state + self.config.anti_collapse_beta * (initial_anchor - stepped_state)

        diff = stepped_state - current_state
        norm_diff = torch.norm(diff, p="fro").item()
        norm_curr = max(1e-8, torch.norm(current_state, p="fro").item())
        velocity = norm_diff / norm_curr

        return stepped_state, velocity
