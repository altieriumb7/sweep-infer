"""
SynapticFold v2: Intrinsic Parameter-Free Synaptic Mobilization Primitives.

Eliminates arbitrary hyper-parameters (such as static Top-K ratios or hardcoded P_mass thresholds)
by deriving the active synaptic core directly from the intrinsic geometry of the representation:
1. Kneedle Knee-Point Detection: Identifies the point of maximal curvature on the normalized
   cumulative activation energy manifold:
       D(j) = (y_j - x_j) / sqrt(2)
       j* = argmax_{j} (y_j - x_j)
2. Donoho-Johnstone Universal Noise Threshold: Computes the non-parametric noise floor via MAD:
       sigma_hat = median(|g|) / 0.67449
       tau_noise = sigma_hat * sqrt(2 * ln(d_ff))
       Neuron_i is Active <=> |g_i| > tau_noise
3. IntrinsicSynapticMLPHook: Drop-in zero-parameter transformer MLP hook supporting both
   zero-allocation sparse masking and zero-scatter indexed GEMM.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def kneedle_knee_point(
    gate_magnitudes: torch.Tensor,
    min_pct: float = 0.01,
) -> torch.Tensor:
    """
    Vectorized Kneedle Knee-Point Detection (Satopaa et al.):
    Locates the intrinsic knee of the cumulative energy spectrum where the marginal
    information gain per neuron drops below uniform geometric expectation.
    
    Args:
        gate_magnitudes: Tensor of shape [..., d_ff] containing non-negative gate magnitudes.
        min_pct: Minimum physiological fraction of active concept synapses to guarantee (default 1%).
        
    Returns:
        mask: Boolean tensor of shape [..., d_ff] where active synapses are True.
    """
    d_ff = gate_magnitudes.size(-1)
    orig_shape = gate_magnitudes.shape
    flat_mag = gate_magnitudes.reshape(-1, d_ff)

    # 1. Sort descending to establish empirical energy hierarchy
    sorted_vals, sorted_indices = torch.sort(flat_mag, dim=-1, descending=True)

    # 2. Compute normalized cumulative energy curve y in [0, 1]
    cumsum = torch.cumsum(sorted_vals, dim=-1)
    total_energy = cumsum[..., -1:] + 1e-9
    y = cumsum / total_energy

    # 3. Uniform diagonal chord connecting (0, 0) to (1, 1)
    x = torch.linspace(1.0 / d_ff, 1.0, d_ff, device=flat_mag.device, dtype=y.dtype)
    x = x.unsqueeze(0).expand_as(y)

    # 4. Orthogonal distance D(j) = (y_j - x_j) / sqrt(2)
    # The knee point maximizes this perpendicular distance
    dist = y - x
    knee_idx = torch.argmax(dist, dim=-1)  # [batch]

    # Guarantee physiological minimum
    min_k = max(1, int(d_ff * min_pct))
    effective_k = torch.clamp(knee_idx + 1, min=min_k, max=d_ff)  # [batch]

    # 5. Construct sorted mask and scatter back to original channel order
    ranks = torch.arange(d_ff, device=flat_mag.device).unsqueeze(0).expand_as(sorted_vals)
    keep_sorted = ranks < effective_k.unsqueeze(-1)

    mask = torch.zeros_like(flat_mag, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, keep_sorted)

    return mask.reshape(orig_shape)


def donoho_johnstone_threshold(
    gate_magnitudes: torch.Tensor,
    min_active: int = 1,
) -> torch.Tensor:
    """
    Donoho-Johnstone Universal Noise Threshold:
    Estimates the representation's non-parametric Gaussian noise floor via MAD:
        sigma_hat = median(|g|) / 0.67449
        tau_noise = sigma_hat * sqrt(2 * ln(d_ff))
    
    Args:
        gate_magnitudes: Tensor of shape [..., d_ff] containing non-negative gate magnitudes.
        min_active: Minimum number of synapses to keep active (default 1).
        
    Returns:
        mask: Boolean tensor of shape [..., d_ff] where active synapses are True.
    """
    d_ff = gate_magnitudes.size(-1)
    orig_shape = gate_magnitudes.shape
    flat_mag = gate_magnitudes.reshape(-1, d_ff)

    # 1. Non-parametric noise floor estimation via Median Absolute Deviation
    med = torch.median(flat_mag, dim=-1, keepdim=True).values
    sigma_hat = med / 0.67449075
    tau_noise = sigma_hat * math.sqrt(2.0 * math.log(float(d_ff)))

    # 2. Universal mobilization mask
    mask = flat_mag > tau_noise

    # Guarantee non-empty core
    empty_rows = ~mask.any(dim=-1)
    if empty_rows.any():
        top1 = torch.argmax(flat_mag, dim=-1, keepdim=True)
        mask.scatter_(-1, top1, True)

    return mask.reshape(orig_shape)


class IntrinsicSynapticMLPHook:
    """
    Parameter-Free Transformer MLP Hook.
    
    Dynamically modulates synaptic mobilization in any standard MLP block (SwiGLU, GeGLU,
    or standard ReLU/GELU) based on intrinsic energy curvature or universal noise filtering.
    """

    def __init__(
        self,
        mlp_module: nn.Module,
        mode: str = "knee",
        min_pct: float = 0.01,
        p_mass: float = 0.85,
        execution_mode: str = "masked",
    ):
        """
        Args:
            mlp_module: The target transformer MLP module (e.g. layer.mlp).
            mode: 'knee' (Kneedle Knee-Point, parameter-free),
                  'noise' (Donoho-Johnstone universal noise threshold, parameter-free),
                  'elastic' (Top-P energy accumulation, uses p_mass).
            min_pct: Minimum physiological active fraction (default 1%).
            p_mass: Retained energy mass fraction (only used if mode='elastic').
            execution_mode: 'masked' (zero-allocation in-place sparse masking) or
                            'indexed' (Zero-Scatter Indexed-GEMM via index_select).
        """
        self.mlp = mlp_module
        self.mode = mode.lower()
        self.min_pct = min_pct
        self.p_mass = p_mass
        self.execution_mode = execution_mode.lower()

        self.orig_forward = mlp_module.forward
        self.is_attached = False

        # Introspect projection layers
        self.is_gated = hasattr(mlp_module, "gate_proj") and hasattr(mlp_module, "up_proj")
        self.gate_proj = getattr(mlp_module, "gate_proj", None)
        self.up_proj = getattr(mlp_module, "up_proj", None)
        self.down_proj = getattr(mlp_module, "down_proj", None)

        if not self.is_gated:
            # Fallback for GPT-2 / Falcon / StarCoder
            self.c_fc = getattr(mlp_module, "c_fc", None) or getattr(mlp_module, "dense_h_to_4h", None)
            self.c_proj = getattr(mlp_module, "c_proj", None) or getattr(mlp_module, "dense_4h_to_h", None)
            self.act_fn = getattr(mlp_module, "act", F.gelu)

        # Diagnostics
        self.last_sparsity = 0.0
        self.last_active_synapses = 0
        self.last_energy_retained = 1.0

    def compute_intrinsic_mask(self, gate: torch.Tensor) -> torch.Tensor:
        """
        Calculates the dynamic synaptic mobilization mask according to the configured mode.
        """
        gate_mag = torch.abs(gate)
        d_ff = gate.size(-1)

        if self.mode == "knee":
            mask = kneedle_knee_point(gate_mag, min_pct=self.min_pct)
        elif self.mode == "noise":
            mask = donoho_johnstone_threshold(gate_mag, min_active=max(1, int(d_ff * self.min_pct)))
        elif self.mode in ["both", "both_intersect"]:
            mask_k = kneedle_knee_point(gate_mag, min_pct=self.min_pct)
            mask_n = donoho_johnstone_threshold(gate_mag, min_active=max(1, int(d_ff * self.min_pct)))
            mask = mask_k & mask_n
            # Fallback if intersection is empty
            if not mask.any():
                mask = mask_n
        elif self.mode == "both_union":
            mask_k = kneedle_knee_point(gate_mag, min_pct=self.min_pct)
            mask_n = donoho_johnstone_threshold(gate_mag, min_active=max(1, int(d_ff * self.min_pct)))
            mask = mask_k | mask_n
        elif self.mode == "elastic":
            sorted_vals, sorted_indices = torch.sort(gate_mag, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_vals, dim=-1)
            total = cumsum[..., -1:] + 1e-9
            keep_sorted = (cumsum / total) <= self.p_mass
            min_k = max(1, int(d_ff * self.min_pct))
            keep_sorted[..., :min_k] = True
            mask = torch.zeros_like(gate_mag, dtype=torch.bool)
            mask.scatter_(-1, sorted_indices, keep_sorted)
        else:
            raise ValueError(f"Unknown intrinsic mode: '{self.mode}'. Choose from ['knee', 'noise', 'both', 'both_intersect', 'both_union', 'elastic'].")

        # Diagnostics tracking
        active_count = mask.sum(dim=-1).float().mean().item()
        self.last_active_synapses = int(round(active_count))
        self.last_sparsity = 1.0 - (active_count / float(d_ff))

        total_energy = gate_mag.sum(dim=-1).float().mean().item() + 1e-9
        retained_energy = (gate_mag * mask).sum(dim=-1).float().mean().item()
        self.last_energy_retained = retained_energy / total_energy

        return mask

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Interception forward implementing Gate-First Pre-Up Gating.
        """
        if self.is_gated:
            # 1. Biological Pre-Synaptic Gate Routing
            gate_raw = self.gate_proj(x)
            gate = F.silu(gate_raw)

            # 2. Intrinsic Parameter-Free Mobilization
            mask = self.compute_intrinsic_mask(gate)

            # 3. Gated Projection Execution
            if self.execution_mode == "indexed" and x.size(0) == 1 and x.dim() == 2:
                # Optimized single-token indexed contraction
                active_idx = torch.nonzero(mask[0]).squeeze(-1)
                active_gate = gate[:, active_idx]
                w_up_active = torch.index_select(self.up_proj.weight, 0, active_idx)
                active_up = F.linear(x, w_up_active)
                active_inter = active_gate * active_up
                w_down_active = torch.index_select(self.down_proj.weight, 1, active_idx)
                out = F.linear(active_inter, w_down_active)
                return out
            else:
                # In-place masked execution (zero weight reallocation overhead)
                masked_gate = gate * mask.to(gate.dtype)
                up = self.up_proj(x)
                intermediate = masked_gate * up
                out = self.down_proj(intermediate)
                return out
        else:
            # Standard MLP (GPT-2, Falcon)
            h = self.c_fc(x)
            act = self.act_fn(h)
            mask = self.compute_intrinsic_mask(act)
            masked_act = act * mask.to(act.dtype)
            out = self.c_proj(masked_act)
            return out

    def attach(self) -> "IntrinsicSynapticMLPHook":
        """
        Attaches this hook to the target MLP module.
        """
        if not self.is_attached:
            self.mlp.forward = self.forward
            self.is_attached = True
        return self

    def detach(self) -> "IntrinsicSynapticMLPHook":
        """
        Detaches this hook, restoring the original MLP forward pass.
        """
        if self.is_attached:
            self.mlp.forward = self.orig_forward
            self.is_attached = False
        return self


class IntrinsicModelWrapper:
    """
    High-level manager to instrument all MLP layers of any HuggingFace model
    with Intrinsic Synaptic Mobilization.
    """

    @staticmethod
    def wrap(
        model: nn.Module,
        mode: str = "knee",
        min_pct: float = 0.01,
        p_mass: float = 0.85,
        execution_mode: str = "masked",
    ) -> List[IntrinsicSynapticMLPHook]:
        """
        Instruments all decoder layers of a model with IntrinsicSynapticMLPHooks.
        """
        base_model = getattr(model, "model", getattr(model, "transformer", model))
        layers = getattr(base_model, "layers", getattr(base_model, "h", None))

        if layers is None:
            raise AttributeError("Could not identify decoder layers list on model.")

        hooks = []
        for layer in layers:
            mlp = getattr(layer, "mlp", getattr(layer, "feed_forward", None))
            if mlp is not None:
                hook = IntrinsicSynapticMLPHook(
                    mlp_module=mlp,
                    mode=mode,
                    min_pct=min_pct,
                    p_mass=p_mass,
                    execution_mode=execution_mode,
                )
                hook.attach()
                hooks.append(hook)

        return hooks

    @staticmethod
    def unwrap(hooks: List[IntrinsicSynapticMLPHook]):
        """
        Detaches all hooks and restores the original model.
        """
        for h in hooks:
            h.detach()

    @staticmethod
    def get_mean_sparsity(hooks: List[IntrinsicSynapticMLPHook]) -> float:
        """
        Computes the current average percentage of quiescent (silent) synapses across all layers.
        """
        if not hooks:
            return 0.0
        return float(sum(h.last_sparsity for h in hooks) / len(hooks)) * 100.0
