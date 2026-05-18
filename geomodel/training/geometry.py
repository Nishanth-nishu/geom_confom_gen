"""
geometry.py — Geometry constraint losses for 3D molecular coordinates.

References:
    - GCDM: Morehead & Cheng, NeurIPS 2023. Section 3.2.
      "Geometry-Complete Diffusion for 3D Molecule Generation."
    - Bond lengths: Standard values from Pyykkö & Atsumi 2009,
      Chemistry: A European Journal.

Background:
    Auxiliary supervision on predicted bond lengths and angles teaches the
    denoiser to produce chemically valid geometries, not just low RMSD.
    This is particularly important for GEOM-Drugs where bond strain in
    generated conformers would make them unusable for docking.

    Applied only at low noise timesteps t < T*τ (see diffusion.py) because
    at high noise levels the x̂₀ prediction is too inaccurate for geometry
    gradients to be meaningful (GCDM Sec. 3.3).
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reference bond lengths (Angstrom)
# From: Pyykkö & Atsumi, Chemistry - A European Journal, 2009.
# Key: (atomic_number_i, atomic_number_j, bond_order)
# ---------------------------------------------------------------------------

_BOND_LENGTHS: Dict[Tuple[int, int, int], float] = {
    # Carbon-Carbon
    (6, 6, 1): 1.540,   # C–C single
    (6, 6, 2): 1.340,   # C=C double
    (6, 6, 3): 1.200,   # C≡C triple
    (6, 6, 4): 1.400,   # C:C aromatic (benzene ~1.40)
    # Carbon-Nitrogen
    (6, 7, 1): 1.469,   # C–N single
    (6, 7, 2): 1.279,   # C=N double
    (6, 7, 3): 1.158,   # C≡N triple
    (6, 7, 4): 1.335,   # C:N aromatic
    # Carbon-Oxygen
    (6, 8, 1): 1.420,   # C–O single
    (6, 8, 2): 1.220,   # C=O double
    (6, 8, 4): 1.310,   # C:O aromatic
    # Carbon-Fluorine
    (6, 9, 1): 1.350,   # C–F
    # Carbon-Sulfur
    (6, 16, 1): 1.810,  # C–S single
    (6, 16, 2): 1.610,  # C=S double
    # Carbon-Chlorine
    (6, 17, 1): 1.770,  # C–Cl
    # Nitrogen-Nitrogen
    (7, 7, 1): 1.450,   # N–N single
    (7, 7, 2): 1.250,   # N=N double
    (7, 7, 3): 1.100,   # N≡N triple
    # Nitrogen-Oxygen
    (7, 8, 1): 1.400,   # N–O single
    (7, 8, 2): 1.210,   # N=O double
    # Oxygen-Oxygen
    (8, 8, 1): 1.480,   # O–O single (peroxide)
}


def get_reference_bond_length(
    z_i: int, z_j: int, bond_order: int, default: float = 1.50
) -> float:
    """
    Look up the ideal bond length for a given atom pair and bond order.

    Args:
        z_i, z_j   : Atomic numbers of the two bonded atoms.
        bond_order  : Bond order (1=single, 2=double, 3=triple, 4=aromatic).
        default     : Fallback length if the pair is not in the table.

    Returns:
        Ideal bond length in Angstrom.
    """
    key = (min(z_i, z_j), max(z_i, z_j), bond_order)
    return _BOND_LENGTHS.get(key, default)


# ---------------------------------------------------------------------------
# Reference bond angles (degrees → radians)
# Approximate ideal angles based on hybridization.
# ---------------------------------------------------------------------------

import math

_IDEAL_ANGLE_RAD = {
    "sp3": math.radians(109.5),   # tetrahedral: C(sp³), N(sp³), O(sp³)
    "sp2": math.radians(120.0),   # trigonal planar: C(sp²), aromatic
    "sp":  math.radians(180.0),   # linear: C(sp)
}


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class GeometryLoss(nn.Module):
    """
    Geometry constraint losses on predicted 3D coordinates.

    Computes:
        L_bond  = (1/|E|) Σ_{(i,j)∈E} (‖x̂_i − x̂_j‖ − d*_{ij})²
        L_angle = (1/|A|) Σ_{i–j–k} (θ̂_{ijk} − θ*_{ijk})²

    These losses push the predicted x₀ to have chemically correct local geometry,
    acting as an inductive bias that the network would otherwise have to learn
    from data alone.
    """

    def bond_length_loss(
        self,
        pos: torch.Tensor,          # (N, 3) predicted coordinates
        atom_types: torch.Tensor,   # (N,) atomic numbers
        edge_index: torch.Tensor,   # (2, E)
        bond_types: torch.Tensor,   # (E,) bond orders
    ) -> torch.Tensor:
        """
        MSE between predicted and ideal bond lengths.

        Returns:
            Scalar loss tensor.
        """
        src, dst = edge_index
        # Only compute for unique undirected edges (src < dst)
        mask = src < dst
        if not mask.any():
            return torch.tensor(0.0, device=pos.device)

        src_u, dst_u, bo_u = src[mask], dst[mask], bond_types[mask]

        # Predicted bond lengths
        diff = pos[src_u] - pos[dst_u]                         # (E', 3)
        pred_dist = torch.norm(diff, dim=-1)                    # (E',)

        # Reference bond lengths (vectorized via a lookup per edge)
        ref_dists = []
        z = atom_types.cpu().tolist()
        for i, j, bo in zip(src_u.tolist(), dst_u.tolist(), bo_u.tolist()):
            ref_dists.append(get_reference_bond_length(z[i], z[j], bo))
        ref_dist = torch.tensor(ref_dists, device=pos.device, dtype=pos.dtype)

        return ((pred_dist - ref_dist) ** 2).mean()

    def bond_angle_loss(
        self,
        pos: torch.Tensor,          # (N, 3)
        atom_types: torch.Tensor,   # (N,)
        edge_index: torch.Tensor,   # (2, E)
    ) -> torch.Tensor:
        """
        MSE between predicted and ideal bond angles at each central atom.

        For each triplet i–j–k (j is center), computes θ_ijk and compares to
        the ideal angle based on j's hybridization (sp3/sp2/sp).

        Returns:
            Scalar loss tensor.
        """
        src, dst = edge_index
        N = pos.size(0)
        device = pos.device

        # Build neighbor list per atom
        neighbors = [[] for _ in range(N)]
        for s, d in zip(src.tolist(), dst.tolist()):
            neighbors[s].append(d)

        angle_losses = []
        z = atom_types.cpu().tolist()

        for center in range(N):
            nbrs = neighbors[center]
            if len(nbrs) < 2:
                continue

            center_pos = pos[center]

            # Estimate hybridization based on neighbor count
            n_nbrs = len(nbrs)
            if n_nbrs <= 2:
                ideal = _IDEAL_ANGLE_RAD["sp"]
            elif n_nbrs == 3:
                ideal = _IDEAL_ANGLE_RAD["sp2"]
            else:
                ideal = _IDEAL_ANGLE_RAD["sp3"]

            ideal_t = torch.tensor(ideal, device=device, dtype=pos.dtype)

            # All pairs of neighbors define a bond angle at `center`
            for i_idx in range(len(nbrs)):
                for j_idx in range(i_idx + 1, len(nbrs)):
                    a, b = nbrs[i_idx], nbrs[j_idx]
                    u = (pos[a] - center_pos)
                    v = (pos[b] - center_pos)
                    cos_theta = (u * v).sum() / (
                        u.norm().clamp(min=1e-6) * v.norm().clamp(min=1e-6)
                    )
                    cos_theta = cos_theta.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                    theta = torch.acos(cos_theta)
                    angle_losses.append((theta - ideal_t) ** 2)

        if not angle_losses:
            return torch.tensor(0.0, device=device)

        return torch.stack(angle_losses).mean()

    def forward(
        self,
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        edge_index: torch.Tensor,
        bond_types: torch.Tensor,
        batch_idx: torch.Tensor,
        include_angles: bool = True,
    ) -> torch.Tensor:
        """
        Compute total geometry loss L_geo = L_bond + L_angle.

        Args:
            include_angles: If True, add bond angle loss (more expensive).

        Returns:
            Scalar loss tensor.
        """
        loss = self.bond_length_loss(pos, atom_types, edge_index, bond_types)
        if include_angles:
            loss = loss + self.bond_angle_loss(pos, atom_types, edge_index)
        return loss
