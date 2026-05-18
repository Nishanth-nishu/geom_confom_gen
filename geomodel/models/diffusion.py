"""
diffusion.py — Conformer Diffusion Model (ConformerDiffusion).

This is the top-level model class that combines:
    - The denoiser network (ConformerDenoiser)
    - The cosine noise schedule and precomputed buffers
    - The forward diffusion process q(x_t | x_0)
    - The training loss L_total
    - The DDIM reverse sampling process

References:
    - DDPM forward/reverse process: Ho et al., NeurIPS 2020.
    - DDIM sampling: Song et al., ICLR 2021. Algorithm 1.
    - x₀ parameterization: EDM, Hoogeboom et al., ICML 2022. Appendix B.
    - Cosine schedule: Nichol & Dhariwal, ICML 2021. Section 3.2.
    - Min-SNR weighting: Hang et al., ICCV 2023. Eq. 7.
    - Geometry-gated loss: GCDM, Morehead & Cheng, NeurIPS 2023. Section 3.3.
    - CoM removal: EDM, Hoogeboom et al., ICML 2022. Eq. 4.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .denoiser import ConformerDenoiser
from .noise_schedule import (
    cosine_beta_schedule,
    get_schedule_buffers,
    min_snr_weight,
    remove_com,
)


class ConformerDiffusion(nn.Module):
    """
    E(3)-equivariant diffusion model for 3D molecular conformer generation.

    Implements the full training and sampling pipeline:
        - Forward process: q(x_t|x_0) = N(√ᾱ_t·x_0, (1−ᾱ_t)·I)
        - Denoiser: f_θ(x_t, t, G) → x̂₀  [x₀ parameterization]
        - Reverse (DDIM): deterministic trajectory x_T → x_0

    Args:
        model_cfg   : Dict of denoiser hyperparameters (from egnn_256.yaml).
        num_timesteps: Total diffusion steps T.
        noise_schedule: 'cosine' (default) or 'linear'.
        s           : Cosine schedule offset (Nichol 2021).
    """

    def __init__(
        self,
        model_cfg: Dict,
        num_timesteps: int = 1000,
        noise_schedule: str = "cosine",
        s: float = 0.008,
    ) -> None:
        super().__init__()

        self.num_timesteps = num_timesteps

        # ── Noise schedule ────────────────────────────────────────────────────
        if noise_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps, s=s)
        else:
            raise ValueError(f"Unknown noise_schedule: '{noise_schedule}'. Use 'cosine'.")

        # Register all derived quantities as non-trainable buffers
        for name, tensor in get_schedule_buffers(betas).items():
            self.register_buffer(name, tensor)

        # ── Denoiser network ─────────────────────────────────────────────────
        self.denoiser = ConformerDenoiser(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 9),
            num_rbf=model_cfg.get("num_rbf", 50),
            d_min=model_cfg.get("d_min", 0.5),
            d_max=model_cfg.get("d_max", 10.0),
            time_dim=model_cfg.get("time_dim", 256),
            num_atom_types=model_cfg.get("num_atom_types", 54),
            num_bond_types=model_cfg.get("num_bond_types", 5),
            use_attention=model_cfg.get("use_attention", True),
            dropout=model_cfg.get("dropout", 0.1),
        )

    # ── Forward diffusion process ─────────────────────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,           # (N, 3) clean coordinates
        t: torch.Tensor,            # (B,)  timestep per molecule
        batch_idx: torch.Tensor,    # (N,)  molecule index per atom
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t from the forward process q(x_t|x_0) (DDPM Eq. 4).

            x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε,  ε ∼ N(0, I)

        Noise ε has its CoM removed so x_t stays in CoM-free subspace.

        Returns:
            x_t   : Noisy coordinates, shape (N, 3).
            noise : The sampled noise ε, shape (N, 3).
        """
        noise = torch.randn_like(x0)
        noise = remove_com(noise, batch_idx)    # keep noise in CoM-free subspace

        # Broadcast per-molecule schedule coefficients to per-atom
        sqrt_acp = self.sqrt_alphas_cumprod[t][batch_idx].unsqueeze(-1)      # (N, 1)
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t][batch_idx].unsqueeze(-1)  # (N, 1)

        x_t = sqrt_acp * x0 + sqrt_omac * noise
        return x_t, noise

    # ── Training ─────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        x0: torch.Tensor,                   # (N, 3) ground-truth coordinates
        atom_types: torch.Tensor,            # (N,)  atomic numbers
        edge_index: torch.Tensor,            # (2, E)
        bond_types: torch.Tensor,            # (E,)
        batch_idx: torch.Tensor,             # (N,)
        geo_loss_fn: Optional[callable] = None,  # geometry constraint loss function
        geo_weight: float = 0.5,             # λ_geo
        geo_t_fraction: float = 0.3,         # τ: gate geo loss to t < T*τ
        min_snr_gamma: float = 5.0,          # γ for Min-SNR weighting
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total training loss L_total = L_mse + λ_geo · L_geo.

        L_mse:
            Direct x₀ MSE, weighted by Min-SNR (Hang et al. 2023):
            L_mse = E_t [ w_t · (1/N_mol) · Σ_i ‖x̂₀_i − x₀_i‖² ]

        L_geo (geometry constraint loss, GCDM Section 3.3):
            Applied only for molecules with t < T*τ (low-noise timesteps).
            At high t, x̂₀ is unreliable → geometry gradients are chaotic.
            For low-t molecules x̂₀ is accurate → geometry gradients are useful.

        Returns:
            Dict with keys: 'total', 'mse', 'geo'.
        """
        device = x0.device
        B = int(batch_idx.max().item()) + 1

        # ── 1. Sample random timestep per molecule ────────────────────────────
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # ── 2. Forward diffuse ────────────────────────────────────────────────
        x_t, _noise = self.q_sample(x0, t, batch_idx)

        # ── 3. Denoise: predict x₀ ────────────────────────────────────────────
        x0_pred = self.denoiser(x_t, t, atom_types, edge_index, bond_types, batch_idx)

        # ── 4. Per-molecule MSE (not per-atom — avoids large-molecule bias) ────
        err_per_atom = ((x0_pred - x0) ** 2).sum(dim=-1)   # (N,) squared error

        mse_per_mol = torch.zeros(B, device=device)
        count_per_mol = torch.zeros(B, device=device)
        mse_per_mol.scatter_add_(0, batch_idx, err_per_atom)
        count_per_mol.scatter_add_(0, batch_idx, torch.ones_like(err_per_atom))
        mse_per_mol = mse_per_mol / count_per_mol.clamp(min=1)

        # ── 5. Min-SNR weighting ─────────────────────────────────────────────
        snr_t = self.snr[t]                                # (B,)
        weights = min_snr_weight(snr_t, gamma=min_snr_gamma)  # (B,)
        mse_loss = (weights * mse_per_mol).mean()

        # ── 6. Geometry loss (t-gated) ────────────────────────────────────────
        geo_loss = torch.tensor(0.0, device=device)
        if geo_loss_fn is not None and geo_weight > 0.0:
            t_threshold = int(self.num_timesteps * geo_t_fraction)
            geo_mask = t < t_threshold                     # (B,) bool: low-noise mols

            if geo_mask.any():
                # Extract atoms belonging to low-t molecules
                atom_mask = geo_mask[batch_idx]            # (N,) bool

                # Re-index batch_idx to 0-based for the subset
                low_mol_ids = geo_mask.nonzero(as_tuple=True)[0]
                remap = torch.full((B,), -1, dtype=torch.long, device=device)
                remap[low_mol_ids] = torch.arange(geo_mask.sum(), device=device)
                bi_sub = remap[batch_idx[atom_mask]]

                # Filter edges to low-t molecules and remap to local atom indices
                row, col = edge_index
                edge_mask = atom_mask[row] & atom_mask[col]
                ei_sub = edge_index[:, edge_mask]
                bt_sub = bond_types[edge_mask]
                at_sub = atom_types[atom_mask]

                global_ids = atom_mask.nonzero(as_tuple=True)[0]
                g2l = torch.full((x0.size(0),), -1, dtype=torch.long, device=device)
                g2l[global_ids] = torch.arange(global_ids.size(0), device=device)
                ei_sub_local = g2l[ei_sub]

                geo_loss = geo_loss_fn(
                    x0_pred[atom_mask], at_sub, ei_sub_local, bt_sub, bi_sub
                )

        total_loss = mse_loss + geo_weight * geo_loss

        return {
            "total": total_loss,
            "mse": mse_loss.detach(),
            "geo": geo_loss.detach() if isinstance(geo_loss, torch.Tensor) else torch.tensor(0.0),
        }

    # ── Inference (DDIM sampling) ─────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        atom_types: torch.Tensor,   # (N,)
        edge_index: torch.Tensor,   # (2, E)
        bond_types: torch.Tensor,   # (E,)
        batch_idx: torch.Tensor,    # (N,)
        num_steps: int = 100,       # DDIM steps (fewer than T=1000, equally spaced)
        eta: float = 0.0,           # Stochasticity (0=deterministic DDIM)
    ) -> torch.Tensor:              # (N, 3) generated coordinates
        """
        Generate 3D conformer(s) from Gaussian noise via DDIM (Song et al. 2021).

        Algorithm:
            1. Sample x_T ∼ N(0, I), remove CoM.
            2. For t = T, T-Δ, ..., 0 (num_steps equally spaced):
               a. Predict x̂₀ = f_θ(x_t, t)
               b. Compute ε̂ from x̂₀ and x_t
               c. Interpolate: x_{t-1} = √ᾱ_{t-1}·x̂₀ + √(1−ᾱ_{t-1}−σ²)·ε̂ + σ·z
            3. Return x₀.

        Args:
            num_steps : Number of DDIM denoising steps (default 100).
            eta       : DDIM noise parameter. 0 = deterministic; 1 = DDPM.

        Returns:
            Generated heavy-atom coordinates, shape (N, 3).
        """
        device = atom_types.device
        N = atom_types.size(0)

        # Step 1: sample initial noise (CoM-free)
        x_t = torch.randn(N, 3, device=device)
        x_t = remove_com(x_t, batch_idx)

        # Evenly spaced timestep sequence from T-1 down to 0
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps + 1, dtype=torch.long, device=device
        )

        for i in range(num_steps):
            t_now  = timesteps[i].item()
            t_next = timesteps[i + 1].item()

            t_batch = torch.full((int(batch_idx.max()) + 1,), t_now,
                                 dtype=torch.long, device=device)

            # Predict clean coordinates
            x0_pred = self.denoiser(x_t, t_batch, atom_types, edge_index,
                                    bond_types, batch_idx)

            acp_now  = self.alphas_cumprod[int(t_now)]
            acp_next = self.alphas_cumprod[int(t_next)]

            # Derive noise prediction ε̂ from x₀ prediction (DDPM Eq. 4 rearranged)
            eps_hat = (x_t - acp_now.sqrt() * x0_pred) / (1 - acp_now).sqrt().clamp(min=1e-8)

            # DDIM variance (Song et al. 2021, Eq. 16)
            sigma_sq = eta ** 2 * (
                (1 - acp_next) / (1 - acp_now).clamp(min=1e-8)
                * (1 - acp_now / acp_next.clamp(min=1e-8)).clamp(min=0.0)
            )
            sigma = sigma_sq.sqrt()

            # DDIM update (Song et al. 2021, Eq. 12)
            dir_xt = (1 - acp_next - sigma_sq).clamp(min=0.0).sqrt() * eps_hat

            if i < num_steps - 1:
                noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
                noise = remove_com(noise, batch_idx)
                x_t = acp_next.sqrt() * x0_pred + dir_xt + sigma * noise
            else:
                # Final step: directly output x₀ prediction
                x_t = x0_pred

        return x_t

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
