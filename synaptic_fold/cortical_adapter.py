"""
SynapticFold: Non-Linear Cortical Manifold Adapter for Approach B.

Bridges the latent Riemannian geometry between the Canonical Cortical Column (L12)
and the Motor Output Gateway (L26) using gated non-linear SwiGLU projections,
RMS normalization, and residual stabilization.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class NativeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.float()
        var = x_f32.pow(2).mean(-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(var + self.eps)
        return (normed * self.weight.float()).to(orig_dtype)


class CorticalManifoldAdapter(nn.Module):
    """
    Non-Linear Cortical Manifold Adapter.
    Maps ODE recurrent attractor representations h_12 to the motor manifold h_26.
    """
    def __init__(
        self,
        hidden_size: int = 3584,
        bottleneck_dim: int = 1024,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.dtype = dtype

        self.norm = NativeRMSNorm(hidden_size, eps=1e-6, dtype=dtype)
        self.gate_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, bottleneck_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(bottleneck_dim, hidden_size, bias=False, dtype=dtype)
        self.act_fn = nn.SiLU()

        # Learnable residual gate scalar
        self.res_gate = nn.Parameter(torch.tensor(0.10, dtype=dtype))

        # Initialize projection weights with small variance for smooth initial drift
        with torch.no_grad():
            nn.init.normal_(self.gate_proj.weight, std=0.01)
            nn.init.normal_(self.up_proj.weight, std=0.01)
            nn.init.normal_(self.down_proj.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        target_dtype = self.gate_proj.weight.dtype
        x_cast = x.to(target_dtype)
        h = self.norm(x_cast)
        delta = self.down_proj(self.act_fn(self.gate_proj(h)) * self.up_proj(h))
        out = x_cast + self.res_gate.to(target_dtype) * delta
        return out.to(orig_dtype)

    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Adapter weights not found at: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        self.to(self.dtype)
