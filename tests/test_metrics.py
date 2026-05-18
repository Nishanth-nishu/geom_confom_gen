#!/usr/bin/env python3
"""
test_metrics.py — Unit tests for Kabsch RMSD and evaluation metrics.

Tests:
    1. kabsch_rmsd(P, P) == 0 (identical structures).
    2. kabsch_rmsd is symmetric: kabsch_rmsd(P, Q) == kabsch_rmsd(Q, P).
    3. kabsch_rmsd is invariant to rotation: kabsch_rmsd(R@P, Q) == kabsch_rmsd(P, Q).
    4. kabsch_rmsd is invariant to translation.
    5. COV-R == 1.0 when identical structures (RMSD=0 < any threshold).
    6. MAT-R == 0.0 when identical structures.
    7. Diversity == 0.0 for a single generated conformer.
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from geomodel.evaluation.kabsch import kabsch_rmsd
from geomodel.evaluation.metrics import molecule_metrics


def random_rotation():
    A = np.random.randn(3, 3)
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_identical_rmsd():
    """RMSD of a structure with itself must be 0."""
    P = np.random.randn(10, 3)
    assert kabsch_rmsd(P, P) < 1e-8, "RMSD(P, P) must be 0"


def test_symmetry():
    """RMSD(P, Q) must equal RMSD(Q, P)."""
    P = np.random.randn(8, 3)
    Q = np.random.randn(8, 3)
    assert abs(kabsch_rmsd(P, Q) - kabsch_rmsd(Q, P)) < 1e-6


def test_rotation_invariance():
    """RMSD must be invariant to rotation of either structure."""
    P = np.random.randn(12, 3)
    Q = np.random.randn(12, 3)
    R = random_rotation()
    rmsd_orig = kabsch_rmsd(P, Q)
    rmsd_rot  = kabsch_rmsd(P @ R.T, Q)
    assert abs(rmsd_orig - rmsd_rot) < 1e-5, (
        f"Rotation changed RMSD: {rmsd_orig:.6f} vs {rmsd_rot:.6f}"
    )


def test_translation_invariance():
    """RMSD must be invariant to translation."""
    P = np.random.randn(10, 3)
    Q = np.random.randn(10, 3)
    t = np.array([5.0, -3.0, 2.5])
    rmsd_orig = kabsch_rmsd(P, Q)
    rmsd_trans = kabsch_rmsd(P + t, Q)
    assert abs(rmsd_orig - rmsd_trans) < 1e-5


def test_known_rmsd():
    """Known RMSD: two structures differing by a fixed displacement."""
    N = 5
    P = np.zeros((N, 3))
    Q = np.zeros((N, 3))
    Q[:, 0] = 1.0          # all atoms shifted 1Å in x-direction
    # After centering, both have mean 0 — so RMSD = 0 (same centered structure)
    # (a pure translation has zero RMSD after Kabsch centering)
    assert kabsch_rmsd(P, Q) < 1e-6


def test_cov_r_perfect():
    """COV-R must be 1.0 when generated = reference (RMSD=0 < threshold)."""
    P = np.random.randn(8, 3)
    metrics = molecule_metrics(ref_set=[P], gen_set=[P], threshold=0.5)
    assert metrics["cov_r"] == 1.0, f"Expected COV-R=1.0, got {metrics['cov_r']}"
    assert metrics["mat_r"] < 1e-6, f"Expected MAT-R≈0, got {metrics['mat_r']}"


def test_diversity_single_gen():
    """Diversity must be 0 when only one conformer is generated."""
    P = np.random.randn(8, 3)
    Q = np.random.randn(8, 3)
    metrics = molecule_metrics(ref_set=[Q], gen_set=[P], threshold=0.5)
    assert metrics["diversity"] == 0.0, (
        f"Single conformer diversity must be 0, got {metrics['diversity']}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
