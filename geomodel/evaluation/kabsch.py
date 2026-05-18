"""
kabsch.py — Kabsch RMSD algorithm for optimal rigid-body alignment.

References:
    - Kabsch, W. (1976). "A solution for the best rotation to relate two sets
      of vectors." Acta Crystallographica A, 32(5), 922–923.
    - Implementation follows: https://en.wikipedia.org/wiki/Kabsch_algorithm

The Kabsch algorithm finds the rotation matrix R that minimizes the RMSD
between two centered point clouds P and Q:

    RMSD = √(mean_i ‖R·p_i − q_i‖²)

This is used to compute the minimum possible RMSD between a generated
conformer and a reference conformer after optimal superimposition.

Note on reflection handling:
    SVD can produce det(VUᵀ) = -1 (improper rotation = reflection).
    We correct this via the diagonal matrix D = diag(1, 1, sign(det(VUᵀ))).
"""

import numpy as np


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute minimum RMSD between point clouds P and Q via Kabsch alignment.

    Steps:
        1. Center P and Q (remove mean).
        2. Compute covariance matrix H = Pᵀ Q.
        3. SVD: H = U Σ Vᵀ.
        4. Correct for reflection: D = diag(1, 1, sign(det(V Uᵀ))).
        5. Optimal rotation: R = V D Uᵀ.
        6. RMSD = √(mean ‖R·P − Q‖²).

    Args:
        P : Generated conformer coordinates, shape (N, 3). Float64.
        Q : Reference conformer coordinates, shape (N, 3). Float64.

    Returns:
        Minimum RMSD in Angstroms (float).

    Raises:
        ValueError: If P and Q have different shapes.
    """
    P = np.array(P, dtype=np.float64)
    Q = np.array(Q, dtype=np.float64)

    if P.shape != Q.shape:
        raise ValueError(
            f"P and Q must have the same shape. Got {P.shape} vs {Q.shape}."
        )

    # Step 1: Center both structures
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)

    # Step 2: Covariance matrix
    H = P.T @ Q                              # (3, 3)

    # Step 3: SVD
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Step 4: Correct for reflections (det = -1 means improper rotation)
    d = np.sign(np.linalg.det(V @ U.T))
    D = np.diag([1.0, 1.0, d])

    # Step 5: Optimal rotation matrix
    R = V @ D @ U.T                          # (3, 3)

    # Step 6: Apply rotation and compute RMSD
    P_rot = P @ R.T                          # (N, 3)
    diff = P_rot - Q                         # (N, 3)
    rmsd = float(np.sqrt((diff ** 2).sum(axis=-1).mean()))

    return rmsd


def batch_min_rmsd(gen_set: list, ref_set: list) -> np.ndarray:
    """
    Compute the minimum RMSD from each reference to the closest generated conformer.

    Args:
        gen_set : List of generated conformers, each shape (N, 3).
        ref_set : List of reference conformers, each shape (N, 3).

    Returns:
        Array of shape (len(ref_set),) with min RMSD per reference.
    """
    min_rmsds = np.zeros(len(ref_set))
    for r_idx, ref in enumerate(ref_set):
        rmsds = [kabsch_rmsd(gen, ref) for gen in gen_set]
        min_rmsds[r_idx] = min(rmsds)
    return min_rmsds


def batch_min_rmsd_gen_to_ref(gen_set: list, ref_set: list) -> np.ndarray:
    """
    Compute the minimum RMSD from each generated conformer to the closest reference.

    Used for COV-P and MAT-P (precision-side metrics).

    Returns:
        Array of shape (len(gen_set),) with min RMSD per generated conformer.
    """
    min_rmsds = np.zeros(len(gen_set))
    for g_idx, gen in enumerate(gen_set):
        rmsds = [kabsch_rmsd(gen, ref) for ref in ref_set]
        min_rmsds[g_idx] = min(rmsds)
    return min_rmsds
