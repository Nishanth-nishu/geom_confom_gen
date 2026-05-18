#!/usr/bin/env python3
"""
test_egnn.py — Unit tests for EquivariantLayer.

Tests:
    1. Output shape correctness.
    2. E(3)-equivariance: rotating input coords by R should rotate output by R.
    3. Translation invariance: translating all atoms should not change output coords
       (only distances are used in messages).
    4. No NaN/Inf values in output.
"""

import math
import torch
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geomodel.models.egnn import EquivariantLayer


def random_rotation_matrix(device: torch.device) -> torch.Tensor:
    """Generate a random orthogonal 3×3 rotation matrix via QR decomposition."""
    A = torch.randn(3, 3, device=device)
    Q, R = torch.linalg.qr(A)
    # Correct for sign convention (ensure det = +1)
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


@pytest.fixture
def simple_graph():
    """A small 5-atom graph for testing."""
    device = torch.device("cpu")
    N, E, H = 5, 8, 64

    h = torch.randn(N, H)
    x = torch.randn(N, 3)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ], dtype=torch.long)
    bond_feat = torch.randn(E, H)

    return h, x, edge_index, bond_feat, device


def test_output_shapes(simple_graph):
    """Output shapes must match input shapes."""
    h, x, edge_index, bond_feat, device = simple_graph
    layer = EquivariantLayer(hidden_dim=64, num_rbf=20).to(device)

    h_new, x_new = layer(h, x, edge_index, bond_feat)

    assert h_new.shape == h.shape, f"Node shape mismatch: {h_new.shape} vs {h.shape}"
    assert x_new.shape == x.shape, f"Coord shape mismatch: {x_new.shape} vs {x.shape}"


def test_no_nan(simple_graph):
    """No NaN or Inf in outputs."""
    h, x, edge_index, bond_feat, device = simple_graph
    layer = EquivariantLayer(hidden_dim=64, num_rbf=20).to(device)
    h_new, x_new = layer(h, x, edge_index, bond_feat)

    assert not torch.isnan(h_new).any(), "NaN in node features"
    assert not torch.isnan(x_new).any(), "NaN in coordinates"
    assert not torch.isinf(h_new).any(), "Inf in node features"
    assert not torch.isinf(x_new).any(), "Inf in coordinates"


def test_equivariance(simple_graph):
    """
    E(3)-equivariance test.

    If input coords are rotated by R, output coords should also be rotated by R.
    Node features h should be invariant (same before and after rotation).
    """
    h, x, edge_index, bond_feat, device = simple_graph
    layer = EquivariantLayer(hidden_dim=64, num_rbf=20).to(device)
    layer.eval()

    R = random_rotation_matrix(device)

    with torch.no_grad():
        # Forward on original
        h1, x1 = layer(h, x, edge_index, bond_feat)

        # Forward on rotated input
        x_rot = x @ R.T
        h2, x2 = layer(h, x_rot, edge_index, bond_feat)

    # Rotated output should equal rotation of original output
    x1_rot = x1 @ R.T
    assert torch.allclose(x1_rot, x2, atol=1e-4), (
        f"Equivariance violated. Max diff: {(x1_rot - x2).abs().max().item():.6f}"
    )

    # Node features should be invariant
    assert torch.allclose(h1, h2, atol=1e-4), (
        f"Invariance of h violated. Max diff: {(h1 - h2).abs().max().item():.6f}"
    )


def test_translation_invariance(simple_graph):
    """
    Translation invariance test.

    Translating all coordinates by a constant vector t should not change the
    relative positions, so outputs should differ by at most the same translation
    in coords (and node features must be identical).
    """
    h, x, edge_index, bond_feat, device = simple_graph
    layer = EquivariantLayer(hidden_dim=64, num_rbf=20).to(device)
    layer.eval()

    t = torch.tensor([5.0, -3.0, 2.5]).to(device)

    with torch.no_grad():
        h1, x1 = layer(h, x, edge_index, bond_feat)
        h2, x2 = layer(h, x + t, edge_index, bond_feat)

    # Node features must be exactly the same (only distances used)
    assert torch.allclose(h1, h2, atol=1e-4), (
        f"Node features changed under translation. Max diff: {(h1 - h2).abs().max():.6f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
