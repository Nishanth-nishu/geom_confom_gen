"""
reporter.py — Pretty-print conformer generation evaluation results.
"""

from typing import Dict


SOTA_BASELINES = {
    "GeoDiff (QM9)":  {"cov_r": 71.0, "mat_r": 0.297, "cov_p": None, "mat_p": None},
    "GeoMol (QM9)":   {"cov_r": 71.5, "mat_r": 0.225, "cov_p": None, "mat_p": None},
    "TorDiff (QM9)":  {"cov_r": 73.2, "mat_r": 0.219, "cov_p": None, "mat_p": None},
}


def print_results_table(
    results: Dict[str, float],
    label: str = "geo-model",
    threshold: float = 0.5,
    dataset: str = "QM9",
    n_mols: int = 0,
    n_gen: int = 0,
) -> None:
    """
    Print a formatted evaluation results table comparing to SOTA baselines.

    Args:
        results  : Dict with keys cov_r, mat_r, cov_p, mat_p, diversity.
        label    : Name for the current model run.
        threshold: RMSD threshold used for COV metrics.
        dataset  : Dataset name for the table header.
        n_mols   : Number of evaluated molecules.
        n_gen    : Number of generated conformers per molecule.
    """
    sep = "─" * 72

    print(f"\n{sep}")
    print(f"  Conformer Generation Evaluation — {dataset}")
    print(f"  δ = {threshold} Å  |  {n_mols} molecules  |  {n_gen} conformers/mol")
    print(sep)
    print(f"  {'Model':<24} {'COV-R':>8} {'MAT-R':>8} {'COV-P':>8} {'MAT-P':>8} {'Div':>8}")
    print(sep)

    # SOTA baselines
    for name, b in SOTA_BASELINES.items():
        cov_r = f"{b['cov_r']:.1f}%" if b["cov_r"] is not None else "   —"
        mat_r = f"{b['mat_r']:.3f}Å" if b["mat_r"] is not None else "   —"
        cov_p = f"{b['cov_p']:.1f}%" if b["cov_p"] is not None else "   —"
        mat_p = f"{b['mat_p']:.3f}Å" if b["mat_p"] is not None else "   —"
        print(f"  {name:<24} {cov_r:>8} {mat_r:>8} {cov_p:>8} {mat_p:>8} {'—':>8}")

    print(sep)

    # Current model
    cov_r_s   = f"{results['cov_r'] * 100:.1f}%"
    mat_r_s   = f"{results['mat_r']:.3f}Å"
    cov_p_s   = f"{results.get('cov_p', 0) * 100:.1f}%"
    mat_p_s   = f"{results.get('mat_p', 0):.3f}Å"
    div_s     = f"{results.get('diversity', 0):.3f}Å"
    print(f"  {label:<24} {cov_r_s:>8} {mat_r_s:>8} {cov_p_s:>8} {mat_p_s:>8} {div_s:>8}")

    print(sep)

    # Beat SOTA?
    best_matr_sota = min(b["mat_r"] for b in SOTA_BASELINES.values() if b["mat_r"])
    if results["mat_r"] < best_matr_sota:
        delta = best_matr_sota - results["mat_r"]
        print(f"  ✅  New SOTA MAT-R! Beats best baseline by {delta:.3f} Å")
    else:
        delta = results["mat_r"] - best_matr_sota
        print(f"  ↳  MAT-R is {delta:.3f} Å above best SOTA ({best_matr_sota:.3f} Å)")
    print(f"{sep}\n")
