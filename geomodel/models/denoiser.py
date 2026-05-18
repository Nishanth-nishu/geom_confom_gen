"""
denoiser.py — Conformer Denoiser network (ConformerDenoiser).

References:
    - x₀ parameterization: EDM, Hoogeboom et al. ICML 2022. Appendix B.
    - Architecture: EDM, Table 1 (hidden=256, layers=9).
    - Timestep conditioning: DDPM, Ho et al. NeurIPS 2020. Section 3.4.
    - Residual prediction: GeoDiff, Xu et al. ICML 2022. Section 3.2.

x₀ Parameterization Rationale:
    Standard DDPM predicts noise ε̂, then recovers x₀ via:
        x̂₀ = (x_t - √(1−ᾱ_t)·ε̂) / √ᾱ_t

    At high noise timesteps (t ≈ T), ᾱ_t ≈ 0, making the division
    catastrophically amplify any prediction error. For 3D coordinates
    this is especially severe because geometry losses applied on the
    resulting x̂₀ would receive chaotic gradient signals.

    By predicting x₀ directly, we eliminate this amplification entirely.
    The loss surface on x₀ is smooth and well-conditioned at all timesteps.

Architecture:
    Atom type embedding  →  +timestep embedding  →  N × EquivariantLayer
    → coordinate prediction head (residual from x_t via updated x inside EGNN)
"""

import torch
import torch.nn as nn

from .egnn import EquivariantLayer
from .noise_schedule import sinusoidal_embedding, remove_com


class ConformerDenoiser(nn.Module):
    """
    E(3)-equivariant denoiser that predicts clean coordinates x₀ from noisy x_t.

    The network is conditioned on:
        - Atom types z_i (embedded into node features h_i)
        - Bond types b_ij (embedded into edge features)
        - Diffusion timestep t (sinusoidal + MLP projection)

    Spatial information flows only through pairwise distances in the EGNN
    layers, which guarantees E(3)-equivariance of the output.

    Args:
        hidden_dim      : Node feature dimension. EDM uses 256.
        num_layers      : Number of EGNN message passing layers. EDM uses 9.
        num_rbf         : Number of RBF bases for distance encoding. GeoDiff uses 50.
        d_min           : Minimum RBF distance (Angstrom).
        d_max           : Maximum RBF distance (Angstrom). Use 10.0 for GEOM-Drugs.
        time_dim        : Sinusoidal timestep embedding dimension.
        num_atom_types  : Atom type vocabulary size (54 covers H→I, atomic 1–53).
        num_bond_types  : Bond type vocabulary (0=none, 1=single, ..., 4=aromatic).
        use_attention   : Enable attention gate in EGNN layers (EDM variant).
        dropout         : Dropout probability in edge/node MLPs.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 9,
        num_rbf: int = 50,
        d_min: float = 0.5,
        d_max: float = 10.0,
        time_dim: int = 256,
        num_atom_types: int = 54,
        num_bond_types: int = 5,
        use_attention: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        # ── Input embeddings ─────────────────────────────────────────────────
        # Atom type → node features (atomic numbers 1–53 map to indices 1–53)
        self.atom_embed = nn.Embedding(num_atom_types, hidden_dim)

        # Bond type → edge features (added as residual inside each EGNN layer)
        self.bond_embed = nn.Embedding(num_bond_types + 1, hidden_dim)

        # Timestep → per-atom conditioning signal
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # ── EGNN Layers (E(3)-equivariant message passing) ────────────────────
        self.layers = nn.ModuleList([
            EquivariantLayer(
                hidden_dim=hidden_dim,
                num_rbf=num_rbf,
                d_min=d_min,
                d_max=d_max,
                use_attention=use_attention,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Output head: node features → Δx correction ───────────────────────
        # Predicts a 3D displacement Δx. Final x₀_pred = x_updated + Δx.
        # (x_updated is the coordinate tensor after all EGNN layers.)
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(
        self,
        x_noisy: torch.Tensor,      # (N, 3) — noisy coordinates x_t
        t: torch.Tensor,            # (B,)  — integer timestep per molecule
        atom_types: torch.Tensor,   # (N,)  — atomic numbers (1–53)
        edge_index: torch.Tensor,   # (2, E) — [src, dst] edge pairs
        bond_types: torch.Tensor,   # (E,)  — bond orders (0–4)
        batch_idx: torch.Tensor,    # (N,)  — molecule index per atom
    ) -> torch.Tensor:              # (N, 3) — predicted clean coordinates x̂₀
        """
        Forward pass: denoise x_t to predict x₀.

        The predicted x₀ has its center of mass removed so it lives in the
        CoM-free subspace (consistent with forward process).
        """
        # ── 1. Node features ─────────────────────────────────────────────────
        h = self.atom_embed(atom_types.clamp(0, self.atom_embed.num_embeddings - 1))

        # ── 2. Timestep conditioning (broadcast to per-atom) ─────────────────
        t_emb = sinusoidal_embedding(t, self.time_dim)   # (B, time_dim)
        t_proj = self.time_mlp(t_emb)                    # (B, hidden_dim)
        h = h + t_proj[batch_idx]                        # (N, hidden_dim)

        # ── 3. Bond type embedding (per-edge, used inside EGNN layers) ────────
        bond_feat = self.bond_embed(
            bond_types.clamp(0, self.bond_embed.num_embeddings - 1)
        )                                                 # (E, hidden_dim)

        # ── 4. Equivariant message passing ────────────────────────────────────
        x = x_noisy
        for layer in self.layers:
            h, x = layer(h, x, edge_index, bond_feat)

        # ── 5. Predict x₀ = updated_x + learned correction ───────────────────
        delta_x = self.coord_head(h)                     # (N, 3)
        x0_pred = x + delta_x                            # (N, 3)

        # Remove CoM to keep prediction in CoM-free subspace (EDM Eq. 4)
        x0_pred = remove_com(x0_pred, batch_idx)

        return x0_pred

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
