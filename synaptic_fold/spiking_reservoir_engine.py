"""
SynapticFold: Event-Driven Spiking Reservoir with Temporal Delta-Gating (TMC-Event).

Bypasses the autoregressive token-by-token compute wall by tracking semantic continuity
in the residual stream. If the relative temporal delta ||x_t - x_{t-1}|| is below
the event threshold, the router search is skipped and the active minicolumn state
reverberates in an in-place leaky reservoir ODE.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikingReservoirMLP(nn.Module):
    """
    Cortical Manifold with Temporal Event Sparsity.
    Skips router re-evaluations on continuous token sequences.
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        intermediate_size: int = 14336,
        num_packets: int = 64,
        active_k: int = 8,
        delta_threshold: float = 0.25,
        leaky_gamma: float = 0.15,
        temperature: float = 0.5,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_packets = num_packets
        self.active_k = active_k
        self.delta_threshold = delta_threshold
        self.leaky_gamma = leaky_gamma
        self.temperature = temperature
        self.dtype = dtype
        self.device = device
        self.packet_size = intermediate_size // num_packets

        # Centroids
        self.register_buffer("centroids", torch.empty(num_packets, hidden_size, dtype=dtype, device=device))

        # Packets
        self.register_buffer("gate_packets", torch.empty(num_packets, self.packet_size, hidden_size, dtype=dtype, device=device))
        self.register_buffer("up_packets", torch.empty(num_packets, self.packet_size, hidden_size, dtype=dtype, device=device))
        self.register_buffer("down_packets", torch.empty(num_packets, hidden_size, self.packet_size, dtype=dtype, device=device))

        theoretical_scale = math.sqrt(num_packets * active_k)
        self.register_buffer("manifold_scale", torch.tensor(theoretical_scale, dtype=dtype, device=device))

        # Temporal Event Cache (for autoregressive single-token decoding)
        self.register_buffer("last_x", torch.zeros(1, hidden_size, dtype=dtype, device=device))
        self.register_buffer("last_y", torch.zeros(1, hidden_size, dtype=dtype, device=device))
        self.cached_topk_indices: Optional[torch.Tensor] = None
        self.cached_gate_weights: Optional[torch.Tensor] = None
        self.has_cache = False

        # Metrics
        self.total_tokens_evaluated = 0
        self.event_skips_count = 0

    def reset_reservoir(self):
        """Resets the temporal memory state between generation prompts."""
        self.last_x.zero_()
        self.last_y.zero_()
        self.cached_topk_indices = None
        self.cached_gate_weights = None
        self.has_cache = False

    @classmethod
    def from_dense_swiglu(
        cls,
        gate_proj_w: torch.Tensor,
        up_proj_w: torch.Tensor,
        down_proj_w: torch.Tensor,
        num_packets: int = 64,
        active_k: int = 8,
        delta_threshold: float = 0.25,
        leaky_gamma: float = 0.15,
        temperature: float = 0.5,
    ) -> "SpikingReservoirMLP":
        inter_size, hid_size = gate_proj_w.shape
        engine = cls(
            hidden_size=hid_size,
            intermediate_size=inter_size,
            num_packets=num_packets,
            active_k=active_k,
            delta_threshold=delta_threshold,
            leaky_gamma=leaky_gamma,
            temperature=temperature,
            dtype=gate_proj_w.dtype,
            device=str(gate_proj_w.device),
        )
        packet_size = engine.packet_size
        for p in range(num_packets):
            s = p * packet_size
            e = s + packet_size if p < num_packets - 1 else inter_size
            cur_size = e - s

            engine.gate_packets.data[p, :cur_size].copy_(gate_proj_w[s:e])
            engine.up_packets.data[p, :cur_size].copy_(up_proj_w[s:e])
            engine.down_packets.data[p, :, :cur_size].copy_(down_proj_w[:, s:e])

            c = gate_proj_w[s:e].float().mean(dim=0)
            c = F.normalize(c, p=2, dim=-1)
            engine.centroids.data[p].copy_(c)

        return engine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        B = x_flat.shape[0]

        # Multi-token prompt (Prefill phase): run standard TMC without temporal skip
        if B > 1 or not self.has_cache:
            x_norm = F.normalize(x_flat.float(), p=2, dim=-1)
            scores = torch.matmul(x_norm, self.centroids.t())
            topk_scores, topk_indices = torch.topk(scores, k=self.active_k, dim=-1)
            gate_weights = F.softmax(topk_scores / self.temperature, dim=-1).to(self.dtype)

            # Record state of the last token in prefill
            if B == 1:
                self.cached_topk_indices = topk_indices[0].clone()
                self.cached_gate_weights = gate_weights[0].clone()
                self.last_x.copy_(x_flat[0:1])
                self.has_cache = True

            out = self._compute_packets(x_flat, topk_indices, gate_weights)
            if B == 1:
                self.last_y.copy_(out)
            return out.reshape(*orig_shape)

        # Single-token Autoregressive Decoding with Event Sparsity
        self.total_tokens_evaluated += 1
        x_curr = x_flat[0:1]

        # Calculate directional cosine delta from previous token
        cos_sim = F.cosine_similarity(x_curr.float(), self.last_x.float(), dim=-1).item()
        directional_delta = 1.0 - cos_sim

        if directional_delta < self.delta_threshold:
            # Event Skip Triggered! Reuse active minicolumn set
            self.event_skips_count += 1
            topk_indices = self.cached_topk_indices.unsqueeze(0)
            gate_weights = self.cached_gate_weights.unsqueeze(0)
        else:
            # Full Spiking Event: Re-evaluate router
            x_norm = F.normalize(x_curr.float(), p=2, dim=-1)
            scores = torch.matmul(x_norm, self.centroids.t())
            topk_scores, topk_idx = torch.topk(scores, k=self.active_k, dim=-1)
            gate_weights = F.softmax(topk_scores / self.temperature, dim=-1).to(self.dtype)
            topk_indices = topk_idx

            self.cached_topk_indices = topk_indices[0].clone()
            self.cached_gate_weights = gate_weights[0].clone()

        # Compute output on targeted packets
        out_curr = self._compute_packets(x_curr, topk_indices, gate_weights)

        # Leaky reverberation filter
        if self.leaky_gamma > 0.0:
            out_final = (1.0 - self.leaky_gamma) * out_curr + self.leaky_gamma * self.last_y
        else:
            out_final = out_curr

        self.last_x.copy_(x_curr)
        self.last_y.copy_(out_final)

        return out_final.reshape(*orig_shape)

    def _compute_packets(self, x_flat: torch.Tensor, topk_indices: torch.Tensor, gate_weights: torch.Tensor) -> torch.Tensor:
        B = x_flat.shape[0]
        out = torch.zeros_like(x_flat)

        for b in range(B):
            indices = topk_indices[b]
            alphas = gate_weights[b].view(self.active_k, 1, 1)

            Wg = self.gate_packets[indices]
            Wu = self.up_packets[indices]
            Wd = self.down_packets[indices]

            x_exp = x_flat[b].unsqueeze(1).unsqueeze(0).expand(self.active_k, -1, -1)
            g = torch.bmm(Wg, x_exp)
            u = torch.bmm(Wu, x_exp)
            act = F.silu(g.float()) * u.float()

            y_p = torch.bmm(Wd, act.to(self.dtype))
            out[b : b + 1] = torch.sum(y_p * alphas, dim=0).squeeze(-1).unsqueeze(0) * self.manifold_scale.to(self.dtype)

        return out
