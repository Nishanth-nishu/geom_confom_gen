"""
rbf.py — Gaussian Radial Basis Function distance features.

References:
    - SchNet: Schütt et al., NeurIPS 2017. Section 2.2.
    - DimeNet: Klicpera et al., ICLR 2020. Section 3.1.
    - GeoDiff: Xu et al., ICML 2022. Section 3.1 (uses RBF for distances).

Background:
    Raw scalar pairwise distances give the denoiser very little geometric
    resolution — a single number carries no information about which distance
    range we're in. RBF features expand each scalar distance d into a
    K-dimensional vector of Gaussian activations:

        φ_k(d) = exp(−γ · (d − μ_k)²),  μ_k ∈ [d_min, d_max]

    The network can then learn independent weights for each distance bin,
    giving much richer geometric discrimination power.
"""

import torch
import torch.nn as nn


class GaussianRBF(nn.Module):
    """
    Gaussian RBF expansion of pairwise distances.

    Fixed Gaussian centers μ_k uniformly spaced in [d_min, d_max], with
    a single fixed width σ derived from the center spacing (SchNet convention).

    Args:
        num_rbf : Number of Gaussian basis functions K (default 50, GeoDiff).
        d_min   : Minimum distance in Angstrom (default 0.5Å, below C-C bond).
        d_max   : Maximum distance in Angstrom (default 10.0Å, covers drug molecules).
    """

    def __init__(
        self,
        num_rbf: int = 50,
        d_min: float = 0.5,
        d_max: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_rbf = num_rbf
        self.d_min = d_min
        self.d_max = d_max

        # Fixed Gaussian centers: μ_k uniformly spaced in [d_min, d_max]
        centers = torch.linspace(d_min, d_max, num_rbf)
        self.register_buffer("centers", centers)

        # Width: inversely proportional to spacing between centers (SchNet Eq. 4)
        spacing = (d_max - d_min) / (num_rbf - 1)
        self.gamma = 1.0 / (2.0 * spacing ** 2)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Expand scalar distances into RBF feature vectors.

        Args:
            dist : Pairwise distances, shape (E,) or (E, 1).

        Returns:
            RBF features, shape (E, num_rbf). Values in (0, 1].
        """
        dist = dist.view(-1, 1)                                 # (E, 1)
        diff = dist - self.centers.unsqueeze(0)                 # (E, num_rbf)
        return torch.exp(-self.gamma * diff ** 2)               # (E, num_rbf)

    def extra_repr(self) -> str:
        return (
            f"num_rbf={self.num_rbf}, "
            f"d_min={self.d_min}, "
            f"d_max={self.d_max}, "
            f"gamma={self.gamma:.4f}"
        )
