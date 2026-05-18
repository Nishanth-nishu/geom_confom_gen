"""
metrics.py — Conformer evaluation metrics (COV-R/P, MAT-R/P, Diversity).

References:
    - GeoDiff: Xu et al., ICML 2022. Section 4.1 — COV and MAT metrics.
    - TorDiff: Jing et al., NeurIPS 2022. Section 3 — same protocol.
    - GeoMol: Ganea et al., NeurIPS 2021. Appendix A.

Metric Definitions:

    COV-R (Coverage Recall, δ):
        Fraction of reference conformers covered by at least one generated conformer
        within RMSD threshold δ:
            COV-R = (1/|R|) Σ_r 1[min_g RMSD(g, r) ≤ δ]

    MAT-R (Matching Recall):
        Mean of the best-match RMSD from each reference to the nearest generated:
            MAT-R = (1/|R|) Σ_r min_g RMSD(g, r)
        Lower is better.

    COV-P (Coverage Precision, δ):
        Fraction of generated conformers within RMSD δ of some reference:
            COV-P = (1/|G|) Σ_g 1[min_r RMSD(g, r) ≤ δ]
        Measures realism — are generated conformers chemically valid?

    MAT-P (Matching Precision):
        Mean of the best-match RMSD from each generated to the nearest reference:
            MAT-P = (1/|G|) Σ_g min_r RMSD(g, r)

    Diversity:
        Mean pairwise Kabsch-RMSD among generated conformers for one molecule:
            Div = (1 / C(n,2)) Σ_{i<j} RMSD(g_i, g_j)
        High diversity → multi-modal sampling (no mode collapse).

Standard thresholds:
    QM9:        δ = 0.5 Å
    GEOM-Drugs: δ = 1.25 Å
"""

from typing import Dict, List

import numpy as np
from itertools import combinations

from .kabsch import kabsch_rmsd, batch_min_rmsd, batch_min_rmsd_gen_to_ref


def molecule_metrics(
    ref_set: List[np.ndarray],
    gen_set: List[np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all 5 metrics for a single molecule.

    Args:
        ref_set   : List of M reference conformers, each shape (N, 3).
        gen_set   : List of K generated conformers, each shape (N, 3).
        threshold : RMSD threshold δ for COV metrics (Å).

    Returns:
        Dict with keys: 'cov_r', 'mat_r', 'cov_p', 'mat_p', 'diversity'.
    """
    # ── Recall-side (ref → nearest gen) ──────────────────────────────────────
    min_rmsds_recall = batch_min_rmsd(gen_set, ref_set)  # (M,) min over gens

    cov_r = float((min_rmsds_recall <= threshold).mean())
    mat_r = float(min_rmsds_recall.mean())

    # ── Precision-side (gen → nearest ref) ───────────────────────────────────
    min_rmsds_prec = batch_min_rmsd_gen_to_ref(gen_set, ref_set)  # (K,) min over refs

    cov_p = float((min_rmsds_prec <= threshold).mean())
    mat_p = float(min_rmsds_prec.mean())

    # ── Diversity: mean pairwise RMSD among generated conformers ─────────────
    if len(gen_set) >= 2:
        pair_rmsds = [
            kabsch_rmsd(gen_set[i], gen_set[j])
            for i, j in combinations(range(len(gen_set)), 2)
        ]
        diversity = float(np.mean(pair_rmsds))
    else:
        diversity = 0.0

    return {
        "cov_r": cov_r,
        "mat_r": mat_r,
        "cov_p": cov_p,
        "mat_p": mat_p,
        "diversity": diversity,
    }


def evaluate_metrics(
    all_refs: List[List[np.ndarray]],
    all_gens: List[List[np.ndarray]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Aggregate metrics over a dataset of molecules.

    Args:
        all_refs  : List of M molecules, each a list of reference conformers.
        all_gens  : List of M molecules, each a list of generated conformers.
        threshold : RMSD threshold for COV metrics (Å).

    Returns:
        Dict with mean COV-R, MAT-R, COV-P, MAT-P, Diversity across molecules.
    """
    assert len(all_refs) == len(all_gens), (
        f"Mismatched molecule count: {len(all_refs)} refs, {len(all_gens)} gens."
    )

    per_mol = [
        molecule_metrics(refs, gens, threshold)
        for refs, gens in zip(all_refs, all_gens)
    ]

    keys = ["cov_r", "mat_r", "cov_p", "mat_p", "diversity"]
    return {k: float(np.mean([m[k] for m in per_mol])) for k in keys}
