"""
SynapticFold v2: Cognitive Consensus & 2nd-Order Curvature Halting.

Eliminates the "100% Premature Halting at Step 1" bug:
- Replaces instantaneous 1st-order velocity checks with 2nd-order trajectory acceleration.
- Enforces an adaptive patience barrier (patience >= 2 steps).
- Integrates predictive Shannon entropy stabilization: halts only when uncertainty resolves.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConsensusMetrics:
    cauchy_velocity: float
    trajectory_acceleration: float
    directional_stagnation: float
    predictive_entropy: float
    entropy_delta: float
    step_count: int
    converged: bool


class AdaptiveConsensusCheck(nn.Module):
    """
    Second-Order Adaptive Consensus Checker v2.
    """

    def __init__(
        self,
        unembed_head: Optional[nn.Linear] = None,
        norm_layer: Optional[nn.Module] = None,
        epsilon_velocity: float = 5e-3,
        epsilon_acceleration: float = 2e-3,
        epsilon_angle: float = 2e-4,
        entropy_stability_tau: float = 5e-3,
        min_patience_steps: int = 2,
    ):
        super().__init__()
        self.unembed_head = unembed_head
        self.norm_layer = norm_layer
        self.epsilon_velocity = epsilon_velocity
        self.epsilon_acceleration = epsilon_acceleration
        self.epsilon_angle = epsilon_angle
        self.entropy_stability_tau = entropy_stability_tau
        self.min_patience_steps = min_patience_steps

        # Internal state tracking across relaxation steps
        self.prev_diff: Optional[torch.Tensor] = None
        self.prev_entropy: Optional[float] = None
        self.consecutive_stable_steps: int = 0

    def reset(self):
        self.prev_diff = None
        self.prev_entropy = None
        self.consecutive_stable_steps = 0

    def forward(
        self,
        state_prev: torch.Tensor,
        state_curr: torch.Tensor,
        current_step: int,
    ) -> ConsensusMetrics:
        # 1. State difference vector
        diff_curr = state_curr - state_prev
        norm_diff = torch.norm(diff_curr, p="fro").item()
        norm_prev = max(1e-8, torch.norm(state_prev, p="fro").item())
        cauchy_velocity = norm_diff / norm_prev

        # 2. Second-order trajectory acceleration (curvature)
        if self.prev_diff is not None:
            acc_diff = diff_curr - self.prev_diff
            norm_acc = torch.norm(acc_diff, p="fro").item()
            norm_prev_diff = max(1e-8, torch.norm(self.prev_diff, p="fro").item())
            trajectory_acceleration = norm_acc / norm_prev_diff
        else:
            trajectory_acceleration = 1.0  # High initial acceleration on step 1

        self.prev_diff = diff_curr.detach().clone()

        # 3. Directional stagnation
        v_curr = state_curr.reshape(-1)
        v_prev = state_prev.reshape(-1)
        cosine_sim = F.cosine_similarity(v_curr.unsqueeze(0), v_prev.unsqueeze(0)).item()
        directional_stagnation = max(0.0, 1.0 - cosine_sim)

        # 4. Kinematic Stability Evaluation (Zero-Cost Latent Geometry)
        is_kinematically_stable = (
            cauchy_velocity < self.epsilon_velocity
            and trajectory_acceleration < self.epsilon_acceleration
            and directional_stagnation < self.epsilon_angle
        )

        # 5. Deferred Predictive Shannon Entropy Evaluation
        predictive_entropy = 0.0
        entropy_delta = 1.0

        if self.unembed_head is not None:
            # Deferred Evaluation: only project through the 2.18 GB lm_head when latent
            # trajectory is approaching kinematic consensus or at patience threshold
            should_evaluate_head = is_kinematically_stable or (current_step >= self.min_patience_steps)
            if should_evaluate_head:
                with torch.no_grad():
                    h_last = state_curr[:, -1:, :]
                    if self.norm_layer is not None:
                        h_last = self.norm_layer(h_last)
                    logits = self.unembed_head(h_last)[0, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    predictive_entropy = -torch.sum(probs * log_probs).item()

                    if self.prev_entropy is not None:
                        entropy_delta = abs(predictive_entropy - self.prev_entropy)
                    self.prev_entropy = predictive_entropy
            else:
                entropy_delta = 1.0  # Keep patience active while trajectory is in high acceleration

        is_cognitively_stable = (
            self.unembed_head is None or entropy_delta < self.entropy_stability_tau
        )

        if is_kinematically_stable and is_cognitively_stable:
            self.consecutive_stable_steps += 1
        else:
            self.consecutive_stable_steps = 0

        # Guarantees multi-hop reflection by enforcing min_patience_steps
        converged = (
            current_step >= self.min_patience_steps
            and self.consecutive_stable_steps >= 2
        )

        return ConsensusMetrics(
            cauchy_velocity=cauchy_velocity,
            trajectory_acceleration=trajectory_acceleration,
            directional_stagnation=directional_stagnation,
            predictive_entropy=predictive_entropy,
            entropy_delta=entropy_delta,
            step_count=current_step,
            converged=converged,
        )
