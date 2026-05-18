"""
egnn.py — E(3)-Equivariant Graph Neural Network layer.

References:
    - EGNN: Satorras, Hoogeboom & Welling, ICML 2021.
      "E(n) Equivariant Graph Neural Networks." Equations 3–6.
    - Attention variant: Hoogeboom et al. (EDM), ICML 2022.
      "Equivariant Diffusion for Molecule Generation in 3D." Appendix A.

Design:
    Each layer updates both node features h_i (invariant scalars) and
    3D coordinates x_i (equivariant vectors) through message passing.

    E(3)-equivariance is preserved because:
      - The coordinate update is a weighted sum of displacement vectors (x_i - x_j).
        Displacement vectors transform as vectors under rotation/reflection.
      - The scalar weight w_ij is computed from invariant quantities
        (distances, node features). It is the same regardless of orientation.
      - Therefore x_i' = x_i + Σ_j w_ij (x_i - x_j) is equivariant.

    Translation invariance: only distances are used in message computation,
    not absolute coordinates. The translation cancels in (x_i - x_j).

Update equations (Satorras et al. 2021, Eqs. 3–6):
    m_ij = φ_e(h_i, h_j, RBF(‖x_i-x_j‖), e_ij)   [edge message]
    x_i  ← x_i + Σ_j (x_i-x_j)/‖x_i-x_j‖ · φ_x(m_ij)   [coord update]
    h_i  ← φ_h(h_i, Σ_j m_ij)                             [node update]
"""

from typing import Tuple

import torch
import torch.nn as nn

from .rbf import GaussianRBF


class EquivariantLayer(nn.Module):
    """
    Single E(3)-equivariant message passing layer (EGNN, Satorras et al. 2021).

    Args:
        hidden_dim    : Dimension of node feature vectors.
        num_rbf       : Number of RBF bases for distance encoding.
        d_min         : Minimum distance for RBF (Angstrom).
        d_max         : Maximum distance for RBF (Angstrom).
        use_attention : If True, apply softmax attention gate over neighbors.
        dropout       : Dropout probability in MLPs.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_rbf: int = 50,
        d_min: float = 0.5,
        d_max: float = 10.0,
        use_attention: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        # Distance encoder: raw distances → RBF features
        self.rbf = GaussianRBF(num_rbf=num_rbf, d_min=d_min, d_max=d_max)

        # Edge MLP φ_e: [h_i ‖ h_j ‖ RBF(d_ij)] → m_ij
        # Bond embedding is added as residual after this MLP.
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Coordinate MLP φ_x: m_ij → scalar weight for (x_i - x_j)
        # Tanh bounds the coordinate update, preventing coordinate explosion.
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

        # Optional attention gate: scalar attention score per edge (EDM Appendix A)
        if use_attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        # Node MLP φ_h: [h_i ‖ agg_i] → h_i' (with residual + LayerNorm)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,            # (N, hidden_dim) — node features
        x: torch.Tensor,            # (N, 3) — 3D coordinates
        edge_index: torch.Tensor,   # (2, E) — [source, destination] indices
        bond_feat: torch.Tensor,    # (E, hidden_dim) — bond type embedding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of one EGNN layer.

        Returns:
            h_new : Updated node features, shape (N, hidden_dim).
            x_new : Updated coordinates, shape (N, 3).
        """
        src, dst = edge_index           # src → message sender, dst → receiver

        # ── 1. Compute pairwise geometry ─────────────────────────────────────
        diff = x[src] - x[dst]                                    # (E, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)  # (E, 1)
        unit_vec = diff / dist                                     # (E, 3)
        rbf_feat = self.rbf(dist)                                  # (E, num_rbf)

        # ── 2. Compute edge messages ─────────────────────────────────────────
        edge_input = torch.cat([h[src], h[dst], rbf_feat], dim=-1) # (E, 2h+rbf)
        m_ij = self.edge_mlp(edge_input)                           # (E, hidden)
        m_ij = m_ij + bond_feat                                    # residual: add bond type

        # Optional attention gate: reweight messages by learned attention score
        if self.use_attention:
            att = self.attention_mlp(m_ij)                         # (E, 1), in (0,1)
            m_ij = m_ij * att                                      # gate messages

        # ── 3. Coordinate update (equivariant) ───────────────────────────────
        coord_weight = self.coord_mlp(m_ij)                        # (E, 1)
        coord_update = coord_weight * unit_vec                     # (E, 3)

        N = x.size(0)
        x_agg = torch.zeros(N, 3, device=x.device, dtype=x.dtype)
        x_agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, 3), coord_update)

        # Degree normalization: bidirectional graph counts each neighbor twice
        # in the dst index (once as dst, once as src). Divide by 2 for correctness.
        # See: EGNN Eq. 6 normalization; EDM Appendix A.
        degree = torch.zeros(N, 1, device=x.device)
        degree.scatter_add_(0, dst.unsqueeze(-1),
                            torch.ones(dst.size(0), 1, device=x.device))
        degree = (degree / 2.0).clamp(min=1.0)

        x_new = x + x_agg / degree                                # (N, 3)

        # ── 4. Node feature update (invariant) ───────────────────────────────
        m_agg = torch.zeros(N, self.hidden_dim, device=h.device, dtype=h.dtype)
        m_agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.hidden_dim), m_ij)

        h_new = self.node_mlp(torch.cat([h, m_agg], dim=-1))      # (N, hidden)
        h_new = self.layer_norm(h + h_new)                        # residual + norm

        return h_new, x_new
