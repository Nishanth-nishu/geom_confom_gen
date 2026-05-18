"""
noise_schedule.py — Diffusion noise schedules and SNR utilities.

References:
    - DDPM: Ho et al., NeurIPS 2020. Equation 4 (beta schedule).
    - Cosine schedule: Nichol & Dhariwal, ICML 2021. Section 3.2.
    - Min-SNR weighting: Hang et al., ICCV 2023. Equation 7.
    - EDM CoM removal: Hoogeboom et al., ICML 2022. Equation 4.
"""

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine variance schedule (Nichol & Dhariwal 2021, Eq. 17).

    f(t) = cos((t/T + s) / (1 + s) * π/2)²
    ᾱ_t  = f(t) / f(0)
    β_t  = 1 - ᾱ_t / ᾱ_{t-1}

    The small offset `s=0.008` prevents β_t from being too small near t=0,
    which would cause numerically negligible gradient updates early in training.

    Args:
        timesteps: Total diffusion steps T.
        s: Offset parameter (default 0.008 from paper).

    Returns:
        Tensor of shape (T,) with β_t values clamped to [1e-4, 0.9999].
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.9999).float()


# ---------------------------------------------------------------------------
# Precomputed schedule buffers
# ---------------------------------------------------------------------------

def get_schedule_buffers(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Precompute all quantities derived from β_t needed for training and inference.

    Returns a dict suitable for use with nn.Module.register_buffer().

    Quantities:
        alphas              : α_t = 1 - β_t
        alphas_cumprod      : ᾱ_t = ∏_{s=1}^{t} α_s
        alphas_cumprod_prev : ᾱ_{t-1} (ᾱ_0 = 1.0 by convention)
        sqrt_alphas_cumprod : √ᾱ_t  — signal scale in forward process
        sqrt_one_minus_ac   : √(1-ᾱ_t) — noise scale in forward process
        posterior_variance  : β̃_t = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)
        log_posterior_var   : log(β̃_t) — used in ELBO computation
        snr                 : SNR_t = ᾱ_t / (1 - ᾱ_t) — for Min-SNR weighting
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "posterior_variance": posterior_variance,
        "log_posterior_variance": torch.log(posterior_variance.clamp(min=1e-20)),
        "snr": alphas_cumprod / (1.0 - alphas_cumprod),
    }


# ---------------------------------------------------------------------------
# Loss weighting
# ---------------------------------------------------------------------------

def min_snr_weight(snr: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    """
    Min-SNR loss weighting (Hang et al., ICCV 2023, Eq. 7).

    Clamps SNR_t to γ to avoid over-weighting high-signal (low-noise) timesteps:
        w_t = min(SNR_t, γ) / SNR_t

    This balances gradient magnitudes across all timesteps, preventing the
    optimizer from focusing almost exclusively on easy low-noise examples.

    Args:
        snr   : Signal-to-noise ratio at sampled timesteps, shape (B,).
        gamma : Clamp threshold γ. Paper recommends 5.0 for x₀-prediction.

    Returns:
        Per-sample weight tensor of shape (B,), values in (0, 1].
    """
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def remove_com(x: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    """
    Remove center-of-mass from 3D coordinates, per molecule in a batch.

    Enforces translation invariance by projecting coordinates onto the
    CoM-free subspace ℝ^{3N}_{Σx=0} (EDM, Eq. 4).

    All coordinate tensors (x_0, x_t, sampled noise) must pass through this
    function to keep the generative distribution in CoM-free subspace.

    Args:
        x         : Atom coordinates, shape (N_total, 3).
        batch_idx : Molecule index per atom, shape (N_total,).

    Returns:
        CoM-centered coordinates, same shape as x.
    """
    num_mols = int(batch_idx.max().item()) + 1
    mol_sums = torch.zeros(num_mols, 3, device=x.device, dtype=x.dtype)
    mol_counts = torch.zeros(num_mols, device=x.device, dtype=x.dtype)

    mol_counts.scatter_add_(0, batch_idx, torch.ones(x.size(0), device=x.device))
    mol_sums.scatter_add_(0, batch_idx.unsqueeze(-1).expand(-1, 3), x)

    mol_means = mol_sums / mol_counts.unsqueeze(1).clamp(min=1)
    return x - mol_means[batch_idx]


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal positional embedding for diffusion timesteps (DDPM, §3.4).

    Encodes scalar timestep t as a d-dimensional vector that the denoiser
    uses to condition on the current noise level.

    Args:
        timesteps : Integer timestep indices, shape (B,).
        dim       : Embedding dimension (must be even).

    Returns:
        Embedding tensor, shape (B, dim).
    """
    assert dim % 2 == 0, "Sinusoidal embedding dim must be even."
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    frequencies = torch.exp(
        torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
        * -emb_scale
    )
    angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)  # (B, half_dim)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)     # (B, dim)
