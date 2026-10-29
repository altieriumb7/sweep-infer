import os
import sys
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.cortical_column_engine import CorticalMinicolumnMLP

def test_cortical_column_engine():
    torch.set_num_threads(8)
    hidden_size = 3584
    intermediate_size = 18944
    num_columns = 32
    base_active_columns = 11

    print("--- TESTING CORTICAL MINICOLUMN ENGINE ---")
    
    # Mock dense MLP
    class MockDenseMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.bfloat16)
            self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.bfloat16)
            self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False, dtype=torch.bfloat16)

        def forward(self, x):
            act = F.silu(self.gate_proj(x).float()) * self.up_proj(x).float()
            return self.down_proj(act.to(torch.bfloat16))

    dense_mlp = MockDenseMLP()

    # Convert to CorticalMinicolumnMLP
    col_mlp = CorticalMinicolumnMLP.from_dense_with_permutation(
        dense_mlp=dense_mlp,
        num_columns=num_columns,
        base_active_columns=base_active_columns,
    )

    print(f"Col size: {col_mlp.col_size} neurons per minicolumn across {num_columns} minicolumns.")

    # Calibration inputs
    calib_x = torch.randn(50, hidden_size, dtype=torch.bfloat16)
    col_mlp.calibrate_dendritic_router(calib_x)
    col_mlp.calibrate_homeostatic_resting_potential(calib_x)

    print(f"Router Calibrated: {col_mlp.is_calibrated}")
    print(f"Resting Compensation: {col_mlp.has_resting_compensation}")

    # Test single-token forward (autoregressive step B=1)
    test_x = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        y_dense = dense_mlp(test_x)
        y_col = col_mlp(test_x)

    cos_sim = F.cosine_similarity(y_dense.float().flatten(), y_col.float().flatten(), dim=0).item()
    print(f"Single-token Cosine Similarity: {cos_sim:.5f}")
    assert y_col.shape == test_x.shape, f"Shape mismatch: {y_col.shape} vs {test_x.shape}"
    print("Cortical Minicolumn Engine Unit Test: PASSED!")

if __name__ == "__main__":
    test_cortical_column_engine()
