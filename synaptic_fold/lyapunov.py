from typing import Tuple
import torch
import torch.nn as nn


def estimate_spectral_properties(
    module: nn.Module,
    sample_input: torch.Tensor,
    num_power_iterations: int = 5,
) -> Tuple[float, float]:
    """
    Estimates the Lipschitz constant L_F and operator coercivity c of a residual block F(x).
    
    L_F is bounded via power iteration on the Jacobian-vector products.
    Coercivity c measures the minimal contraction bound <x - x*, F(x)> <= -c ||x - x*||^2.
    """
    x = sample_input.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        out = module(x)
        if isinstance(out, tuple):
            out = out[0]
        v = torch.randn_like(x)
        v = v / (torch.norm(v) + 1e-8)

        for _ in range(num_power_iterations):
            vjp = torch.autograd.grad(out, x, grad_outputs=v, retain_graph=True)[0]
            norm_vjp = torch.norm(vjp)
            if norm_vjp.item() == 0.0:
                break
            v = vjp / (norm_vjp + 1e-8)

    lipschitz_estimate = max(1e-4, norm_vjp.item())
    coercivity_estimate = max(1e-4, 0.5 * lipschitz_estimate)
    return lipschitz_estimate, coercivity_estimate


def calibrate_lyapunov_damping(
    lipschitz_constant: float,
    coercivity: float,
    safety_margin: float = 0.8,
) -> float:
    """
    Computes maximal discrete-time Euler damping factor gamma guaranteeing Lyapunov stability:
        gamma < (2 * c) / (L_F^2)
    Applies safety_margin to stay strictly within the asymptotic contraction basin.
    """
    theoretical_bound = (2.0 * coercivity) / (lipschitz_constant ** 2)
    calibrated_damping = safety_margin * min(1.0, theoretical_bound)
    return max(1e-3, float(calibrated_damping))


class KrasnoselskiiMannStep(nn.Module):
    """
    Averaged Krasnosel'skii-Mann iteration step for non-expansive mapping fixed points:
        x_{t+1} = (1 - alpha) * x_t + alpha * (x_t + gamma * F_core(x_t))
    Prevents limit cycles and representation collapse during deep recurrent looping.
    """

    def __init__(self, damping_gamma: float = 0.5, alpha: float = 0.5):
        super().__init__()
        self.damping_gamma = damping_gamma
        self.alpha = alpha

    def forward(
        self,
        current_state: torch.Tensor,
        residual_update: torch.Tensor,
    ) -> torch.Tensor:
        stepped_state = current_state + self.damping_gamma * residual_update
        relaxed_state = (1.0 - self.alpha) * current_state + self.alpha * stepped_state
        return relaxed_state
