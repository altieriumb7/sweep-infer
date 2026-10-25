from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PlateauCalibrationResult:
    split_start: int
    split_end: int
    core_layer_idx: int
    total_layers: int
    folded_layer_count: int
    compression_ratio: float
    boundary_cka: float
    min_internal_cka: float
    max_internal_drift: float
    motor_cushion_layers: int
    sensory_layers: int
    cka_matrix: torch.Tensor
    layer_drifts: torch.Tensor
    score: float


class RepresentationPlateauProfiler:
    """
    Analytical module for topological layer profiling and automatic folding plateau detection
    in Large Language Models (Qwen2.5, Llama-3, Mistral, etc.).

    Uses Vectorized Linear Centered Kernel Alignment (CKA) and normalized Lipschitz layer drift
    to identify the Gromov representation plateau without trial-and-error heuristics.
    """

    def __init__(
        self,
        tau_cka: float = 0.90,
        delta_crit: float = 0.22,
        min_sensory_layers: int = 3,
        min_motor_layers: int = 4,
        alpha_drift_penalty: float = 2.0,
    ):
        self.tau_cka = tau_cka
        self.delta_crit = delta_crit
        self.min_sensory_layers = min_sensory_layers
        self.min_motor_layers = min_motor_layers
        self.alpha_drift_penalty = alpha_drift_penalty

    @staticmethod
    def compute_cka_matrix(hidden_states_list: List[torch.Tensor], max_tokens: int = 1024) -> torch.Tensor:
        """
        Computes the complete L x L pairwise Linear CKA similarity matrix across all layers
        using vectorized Gram matrix projection.

        Complexity: O(L * N^2 + L^2 * N^2) where N is number of tokens (downsampled to <= max_tokens).
        """
        num_layers = len(hidden_states_list)
        if num_layers == 0:
            raise ValueError("Empty hidden states list provided.")

        flat_layers = []
        for h in hidden_states_list:
            if h.dim() > 2:
                h_flat = h.reshape(-1, h.size(-1))
            else:
                h_flat = h
            if h_flat.size(0) > max_tokens:
                step = max(1, h_flat.size(0) // max_tokens)
                h_flat = h_flat[::step][:max_tokens]
            flat_layers.append(h_flat.to(torch.float32))

        vectors = []
        for x in flat_layers:
            x_centered = x - x.mean(dim=0, keepdim=True)
            k = torch.matmul(x_centered, x_centered.t()).view(-1)
            norm_k = torch.norm(k, p=2)
            if norm_k.item() > 1e-9:
                vectors.append(k / norm_k)
            else:
                vectors.append(torch.zeros_like(k))

        v_matrix = torch.stack(vectors, dim=1)  # (N*N, L)
        cka_matrix = torch.matmul(v_matrix.t(), v_matrix)
        cka_matrix = torch.clamp(cka_matrix, 0.0, 1.0)
        return cka_matrix

    @staticmethod
    def compute_layer_drifts(hidden_states_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the relative residual drift between consecutive layers:
            delta_l = || h_{l+1} - h_l ||_F / (|| h_l ||_F + eps)
        serving as a normalized proxy for the local Lipschitz perturbation rate.
        """
        num_layers = len(hidden_states_list)
        if num_layers < 2:
            return torch.zeros(0)

        drifts = []
        eps = 1e-9
        for l in range(num_layers - 1):
            h_curr = hidden_states_list[l].to(torch.float32)
            h_next = hidden_states_list[l + 1].to(torch.float32)
            diff_norm = torch.norm(h_next - h_curr, p="fro").item()
            curr_norm = torch.norm(h_curr, p="fro").item() + eps
            drifts.append(diff_norm / curr_norm)

        return torch.tensor(drifts, dtype=torch.float32)

    def find_optimal_plateau(
        self,
        cka_matrix: torch.Tensor,
        layer_drifts: torch.Tensor,
    ) -> PlateauCalibrationResult:
        """
        Searches for the optimal contiguous interval [split_start, split_end] that:
          1. Retains at least min_sensory_layers at the beginning and min_motor_layers at the end.
          2. Satisfies CKA(i, j) >= tau_cka for all layers in [split_start, split_end].
          3. Satisfies delta_l <= delta_crit for all layers in [split_start, split_end - 1].
          4. Maximizes the Pareto objective: (folded_layers) * boundary_cka - alpha * max_drift.
          5. Computes the optimal Medoid Core layer c* maximizing alignment with all folded layers.
        """
        total_layers = cka_matrix.size(0)
        max_start = total_layers - self.min_motor_layers - 1
        min_start = self.min_sensory_layers

        best_score = -float("inf")
        best_candidate: Optional[Tuple[int, int]] = None

        for s in range(min_start, max_start):
            for e in range(s + 2, total_layers - self.min_motor_layers + 1):
                sub_cka = cka_matrix[s:e, s:e]
                min_internal_cka = sub_cka.min().item()
                if min_internal_cka < self.tau_cka:
                    continue

                internal_drifts = layer_drifts[s : e - 1]
                max_drift = internal_drifts.max().item() if len(internal_drifts) > 0 else 0.0
                if max_drift > self.delta_crit:
                    continue

                folded_count = e - s
                boundary_cka = cka_matrix[s, e - 1].item()
                score = (folded_count * boundary_cka) - (self.alpha_drift_penalty * max_drift)

                if score > best_score:
                    best_score = score
                    best_candidate = (s, e)

        # Fallback if strict criteria yield no interval: relax tau_cka slightly or select top-1
        if best_candidate is None:
            relaxed_tau = max(0.80, self.tau_cka - 0.08)
            for s in range(min_start, max_start):
                for e in range(s + 2, total_layers - self.min_motor_layers + 1):
                    sub_cka = cka_matrix[s:e, s:e]
                    min_internal_cka = sub_cka.min().item()
                    if min_internal_cka >= relaxed_tau:
                        internal_drifts = layer_drifts[s : e - 1]
                        max_drift = internal_drifts.max().item() if len(internal_drifts) > 0 else 0.0
                        score = (e - s) * cka_matrix[s, e - 1].item() - (self.alpha_drift_penalty * max_drift)
                        if score > best_score:
                            best_score = score
                            best_candidate = (s, e)

        if best_candidate is None:
            # Ultimate conservative fallback: middle 2 layers
            mid = total_layers // 2
            best_candidate = (mid - 1, mid + 1)
            best_score = 0.0

        split_start, split_end = best_candidate
        plateau_sub_cka = cka_matrix[split_start:split_end, split_start:split_end]
        sub_drifts = layer_drifts[split_start : split_end - 1]

        # Calculate Core Medoid: argmax_k sum_j CKA(k, j) within [split_start, split_end)
        medoid_local_idx = plateau_sub_cka.sum(dim=1).argmax().item()
        core_layer_idx = split_start + medoid_local_idx

        folded_count = split_end - split_start
        compression_ratio = (folded_count - 1) / total_layers

        return PlateauCalibrationResult(
            split_start=split_start,
            split_end=split_end,
            core_layer_idx=core_layer_idx,
            total_layers=total_layers,
            folded_layer_count=folded_count,
            compression_ratio=compression_ratio,
            boundary_cka=cka_matrix[split_start, split_end - 1].item(),
            min_internal_cka=plateau_sub_cka.min().item(),
            max_internal_drift=sub_drifts.max().item() if len(sub_drifts) > 0 else 0.0,
            motor_cushion_layers=total_layers - split_end,
            sensory_layers=split_start,
            cka_matrix=cka_matrix,
            layer_drifts=layer_drifts,
            score=best_score,
        )

    @torch.no_grad()
    def calibrate_from_model(
        self,
        model: nn.Module,
        calibration_input_ids: torch.Tensor,
    ) -> PlateauCalibrationResult:
        """
        Profiles a HuggingFace CausalLM model by capturing intermediate representations
        across calibration inputs and computing the optimal folding plateau.
        """
        model.eval()
        device = next(model.parameters()).device
        calibration_input_ids = calibration_input_ids.to(device)

        base_model = getattr(model, "model", model)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise AttributeError("Target model does not possess a recognizable 'layers' ModuleList.")

        hidden_states_per_layer: List[torch.Tensor] = []
        hooks = []

        def get_hook(idx: int):
            def hook(module, input_val, output_val):
                out = output_val[0] if isinstance(output_val, tuple) else output_val
                hidden_states_per_layer.append(out.detach().cpu())
            return hook

        for idx, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(get_hook(idx)))

        try:
            model(calibration_input_ids)
        finally:
            for h in hooks:
                h.remove()

        cka_matrix = self.compute_cka_matrix(hidden_states_per_layer)
        layer_drifts = self.compute_layer_drifts(hidden_states_per_layer)

        return self.find_optimal_plateau(cka_matrix, layer_drifts)
