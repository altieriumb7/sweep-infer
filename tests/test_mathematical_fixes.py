import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import torch.nn as nn
from synaptic_fold.folding import regmean_merge
from synaptic_fold.lyapunov import estimate_spectral_properties, calibrate_lyapunov_damping

def test_regmean_non_square_and_high_dim():
    # 1. Non-square Linear Layer (e.g. Q/K/V projection: d_out=3584, d_in=1024)
    d_out, d_in = 3584, 1024
    w1 = torch.randn(d_out, d_in)
    w2 = torch.randn(d_out, d_in)
    cov1 = torch.eye(d_in) + 0.1 * torch.randn(d_in, d_in)
    cov1 = cov1 @ cov1.t()
    cov2 = torch.eye(d_in) + 0.1 * torch.randn(d_in, d_in)
    cov2 = cov2 @ cov2.t()

    merged = regmean_merge([w1, w2], [cov1, cov2])
    assert merged.shape == (d_out, d_in)

    # 2. High-dimensional SwiGLU down_proj (d_out=3584, d_in=18944)
    d_out_ffn, d_in_ffn = 3584, 18944
    w_down1 = torch.randn(d_out_ffn, d_in_ffn)
    w_down2 = torch.randn(d_out_ffn, d_in_ffn)
    cov_diag1 = torch.rand(d_in_ffn) + 0.5
    cov_diag2 = torch.rand(d_in_ffn) + 0.5

    merged_high_dim = regmean_merge([w_down1, w_down2], [cov_diag1, cov_diag2])
    assert merged_high_dim.shape == (d_out_ffn, d_in_ffn)
    print("test_regmean_non_square_and_high_dim PASSED")


def test_fddc_spectral_properties():
    layer = nn.Sequential(
        nn.Linear(64, 64),
        nn.LayerNorm(64),
        nn.SiLU(),
    )
    x = torch.randn(4, 64)

    l_est, c_est = estimate_spectral_properties(layer, x, num_iterations=5)
    assert l_est > 0.0, f"Lipschitz estimate must be positive: {l_est}"
    print(f"FDDC Spectral: Lipschitz = {l_est:.4f}, Coercivity = {c_est:.4f}")

    damping = calibrate_lyapunov_damping(l_est, c_est)
    assert 0.0 < damping <= 1.0, f"Damping out of bounds: {damping}"
    print(f"Calibrated Damping: {damping:.4f}")


def test_woss_boolean_and_causal_mask():
    from synaptic_fold.spiking import WOSSAttention

    woss = WOSSAttention(saliency_threshold=4.6, top_k_nucleus=16)
    q = torch.randn(2, 4, 8, 32)
    k = torch.randn(2, 4, 8, 32)
    v = torch.randn(2, 4, 8, 32)

    # 1. 2D Boolean mask
    bool_mask = torch.tensor([[True, True, True, False, False, False, False, False],
                              [True, True, True, True, True, True, False, False]])
    # Broadcast to attention matrix shape
    mask_4d = bool_mask.view(2, 1, 1, 8)
    out1, sp1 = woss(q, k, v, mask=mask_4d)
    assert not torch.isnan(out1).any(), "NaN found in boolean masked WOSS attention!"
    assert out1.shape == (2, 4, 8, 32)

    # 2. Causal float mask with -inf
    causal_mask = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1)
    out2, sp2 = woss(q, k, v, mask=causal_mask)
    assert not torch.isnan(out2).any(), "NaN found in causal masked WOSS attention!"
    assert out2.shape == (2, 4, 8, 32)
    print("test_woss_boolean_and_causal_mask PASSED")


if __name__ == "__main__":
    test_regmean_non_square_and_high_dim()
    test_fddc_spectral_properties()
    test_woss_boolean_and_causal_mask()
