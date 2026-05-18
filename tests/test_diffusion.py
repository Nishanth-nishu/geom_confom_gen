#!/usr/bin/env python3
"""
test_diffusion.py — Unit tests for ConformerDiffusion.

Tests:
    1. Forward process q_sample: SNR decreases monotonically with t.
    2. q_sample output stays in CoM-free subspace.
    3. compute_loss returns finite non-negative loss with correct keys.
    4. DDIM sample: output shape correct, no NaN, CoM is ~0.
    5. Model parameter count is in expected range for EDM dims.
"""

import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geomodel.models.diffusion import ConformerDiffusion

# Tiny model config for fast tests
TINY_CFG = {
    "hidden_dim": 64,
    "num_layers": 2,
    "num_rbf": 20,
    "d_min": 0.5,
    "d_max": 6.0,
    "time_dim": 64,
    "use_attention": True,
    "dropout": 0.0,
}

# Small test molecule: ethanol without H → C, C, O (3 heavy atoms)
def make_test_batch(device):
    atom_types = torch.tensor([6, 6, 8], dtype=torch.long, device=device)
    edge_index  = torch.tensor([[0,1,1,2],[1,0,2,1]], dtype=torch.long, device=device)
    bond_types  = torch.tensor([1,1,1,1], dtype=torch.long, device=device)
    batch_idx   = torch.tensor([0,0,0], dtype=torch.long, device=device)
    x0 = torch.tensor([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0], [1.4, 0.9, 0.0]],
                      dtype=torch.float32, device=device)
    x0 = x0 - x0.mean(0)
    return x0, atom_types, edge_index, bond_types, batch_idx


@pytest.fixture
def model():
    return ConformerDiffusion(model_cfg=TINY_CFG, num_timesteps=100)


def test_snr_decreases(model):
    """SNR should be monotonically decreasing (high at t=0, near 0 at t=T-1)."""
    snr = model.snr
    assert (snr[:-1] > snr[1:]).all(), "SNR is not monotonically decreasing"
    assert snr[0] > 1.0, f"SNR at t=0 should be >1, got {snr[0]:.4f}"
    assert snr[-1] < 0.1, f"SNR at t=T should be <0.1, got {snr[-1]:.4f}"


def test_q_sample_com_free(model):
    """q_sample output must have near-zero CoM per molecule."""
    device = torch.device("cpu")
    x0, at, ei, bt, bi = make_test_batch(device)
    t = torch.tensor([10], dtype=torch.long)
    x_t, noise = model.q_sample(x0, t, bi)
    com = x_t.mean(0)
    assert com.abs().max() < 1e-5, f"CoM not zero: {com.tolist()}"


def test_compute_loss_keys(model):
    """compute_loss returns dict with 'total', 'mse', 'geo' keys, all finite."""
    device = torch.device("cpu")
    x0, at, ei, bt, bi = make_test_batch(device)
    loss_dict = model.compute_loss(x0, at, ei, bt, bi)

    for key in ("total", "mse", "geo"):
        assert key in loss_dict, f"Missing key: '{key}'"
        val = loss_dict[key].item()
        assert not (val != val), f"NaN in '{key}' loss"    # NaN check
        assert val >= 0.0, f"Negative '{key}' loss: {val}"


def test_sample_shape_and_no_nan(model):
    """DDIM sample: output shape (N,3), no NaN, CoM ~0."""
    device = torch.device("cpu")
    _, at, ei, bt, bi = make_test_batch(device)
    model.eval()

    with torch.no_grad():
        x_gen = model.sample(at, ei, bt, bi, num_steps=5)

    assert x_gen.shape == (3, 3), f"Wrong shape: {x_gen.shape}"
    assert not torch.isnan(x_gen).any(), "NaN in generated coordinates"
    com = x_gen.mean(0)
    assert com.abs().max() < 0.5, f"CoM too large: {com.tolist()}"


def test_parameter_count():
    """EDM-aligned model (256 hidden, 9 layers) should have ~6–10M parameters."""
    EDM_CFG = {
        "hidden_dim": 256, "num_layers": 9, "num_rbf": 50,
        "time_dim": 256, "use_attention": True, "dropout": 0.1,
    }
    model = ConformerDiffusion(model_cfg=EDM_CFG, num_timesteps=1000)
    n_params = model.num_parameters
    assert 3_000_000 < n_params < 20_000_000, (
        f"Parameter count {n_params:,} is outside expected range [3M, 20M]"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
