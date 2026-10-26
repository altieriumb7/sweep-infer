from typing import Tuple
import torch
import torch.nn as nn


def estimate_spectral_properties(
    module: nn.Module,
    sample_input: torch.Tensor,
    num_iterations: int = 5,
    epsilon: float = 1e-3,
) -> Tuple[float, float]:
    """
    Estimates the Lipschitz constant L_F and operator coercivity c of a residual block F(x)
    using Finite-Difference Directional Contraction (FDDC).
    
    Zero autograd computation graphs, zero backward memory retention, and fully compatible
    with non-differentiable spiking operators.
    
    Lipschitz constant: L_F approx max_v ||F(x + eps*v) - F(x)|| / (eps * ||v||)
    Coercivity: c approx <F(x + eps*v) - F(x), eps*v> / (eps * ||v||)^2
    """
    x = sample_input.detach().clone()
    was_training = module.training
    module.eval()

    with torch.no_grad():
        f_x = module(x)
        if isinstance(f_x, tuple):
            f_x = f_x[0]

        max_lipschitz = 1e-4
        min_coercivity = float("inf")

        # Directional power exploration
        v = torch.randn_like(x)
        v = v / (torch.norm(v) + 1e-8)

        for _ in range(num_iterations):
            pert = epsilon * v
            x_pert = x + pert
            f_pert = module(x_pert)
            if isinstance(f_pert, tuple):
                f_pert = f_pert[0]

            diff = f_pert - f_x
            norm_diff = torch.norm(diff).item()
            norm_pert = torch.norm(pert).item()

            l_iter = norm_diff / max(1e-8, norm_pert)
            c_iter = torch.sum(diff * pert).item() / max(1e-8, norm_pert ** 2)

            max_lipschitz = max(max_lipschitz, l_iter)
            min_coercivity = min(min_coercivity, c_iter)

            # Update perturbation direction along maximal residual divergence
            if norm_diff > 1e-8:
                v = diff / norm_diff

    if was_training:
        module.train()

    lipschitz_estimate = max(1e-4, max_lipschitz)
    coercivity_estimate = min_coercivity if min_coercivity != float("inf") else 0.0
    return lipschitz_estimate, coercivity_estimate


def calibrate_lyapunov_damping(
    lipschitz_constant: float,
    coercivity: float,
    safety_margin: float = 0.8,
) -> float:
    """
    Computes maximal discrete-time Euler damping factor gamma guaranteeing Lyapunov stability.
    
    If coercivity > 0: gamma < (2 * c) / (L_F^2)
    If coercivity <= 0 (expansive or non-contractive): conservative contraction gamma <= safety / max(1.0, L_F)
    """
    if coercivity > 0.0:
        theoretical_bound = (2.0 * coercivity) / (lipschitz_constant ** 2)
        calibrated = safety_margin * min(1.0, theoretical_bound)
    else:
        # Conservative stabilization for non-strictly-contractive basins
        calibrated = safety_margin / max(1.0, lipschitz_constant)

    return max(1e-3, min(1.0, float(calibrated)))


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
