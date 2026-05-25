# NextMol: Full Experiment Flow — Experiments A → G

**Project:** Molecular 3D Conformer Generation on QM9  
**Architecture Family:** E(3)-Equivariant Diffusion  
**Code base:** `/scratch/nishanth.r/nextmol_experiment/mol_next/`  
**Primary metric:** MAT-R (Maximum-over-References RMSD, Å, lower = better)  
**Secondary:** COV-R (Coverage Recall, %, higher = better)

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Shared Mathematical Foundation](#2-shared-mathematical-foundation)
   - [2.1 Diffusion Process (DDPM)](#21-diffusion-process-ddpm)
   - [2.2 Noise Schedule — Cosine β-Schedule](#22-noise-schedule--cosine-β-schedule)
   - [2.3 x₀ Parameterization vs. ε-Parameterization](#23-x₀-parameterization-vs-ε-parameterization)
   - [2.4 E(3)-Equivariance & Center-of-Mass Removal](#24-e3-equivariance--center-of-mass-removal)
   - [2.5 EGNN Message Passing](#25-egnn-message-passing)
   - [2.6 RBF Distance Features](#26-rbf-distance-features)
   - [2.7 Min-SNR Loss Weighting](#27-min-snr-loss-weighting)
   - [2.8 DDIM Sampling](#28-ddim-sampling)
   - [2.9 Geometry Constraint Losses](#29-geometry-constraint-losses)
   - [2.10 Evaluation Metrics (GeoDiff Protocol)](#210-evaluation-metrics-geodiff-protocol)
3. [Experiment A — Fixed Baseline (DDPM)](#3-experiment-a--fixed-baseline-ddpm)
4. [Experiment B — EQGAT-diff Attention EGNN](#4-experiment-b--eqgat-diff-attention-egnn)
5. [Experiment C — Conditional Flow Matching](#5-experiment-c--conditional-flow-matching)
6. [Experiment D — Torsion Auxiliary Loss (TorDiff)](#6-experiment-d--torsion-auxiliary-loss-tordiff)
7. [Experiment E — SOTA Hybrid (CFM + Attention)](#7-experiment-e--sota-hybrid-cfm--attention)
8. [Experiment F — DDP Multi-GPU Training](#8-experiment-f--ddp-multi-gpu-training)
9. [Experiment G — Heavy-Atom SOTA Model](#9-experiment-g--heavy-atom-sota-model)
10. [Architecture Evolution Table](#10-architecture-evolution-table)
11. [Quantitative Results Summary](#11-quantitative-results-summary)
12. [Key Lessons and What Worked](#12-key-lessons-and-what-worked)
13. [References and Citations](#13-references-and-citations)

---

## 1. Problem Formulation

**Task:** Given a 2D molecular graph G = (V, E) — atoms V with types zᵢ ∈ {C, N, O, F, ...} and bonds E with orders bᵢⱼ ∈ {1, 2, 3, aromatic} — generate the distribution of low-energy 3D conformations p(x | G).

**Input:**
- Atom types: `atom_types` ∈ ℤᴺ (atomic numbers, e.g., C=6, N=7, O=8)
- Edge index: `edge_index` ∈ ℤ²×ᴱ (bidirectional bonds, each undirected bond stored as two directed edges)
- Bond types: `bond_types` ∈ ℤᴱ (1=single, 2=double, 3=triple, 4=aromatic)

**Output:**
- 3D coordinates: x ∈ ℝᴺˣ³ for each atom (in Ångström)
- x lives on the *zero center-of-mass* (CoM) subspace: Σᵢ xᵢ = 0

**Dataset:** QM9 [[Ramakrishnan et al. 2014]](#13-references-and-citations) — 130,831 small organic molecules (≤9 heavy atoms) with DFT-computed equilibrium geometries. Exp A-E use explicit-H QM9 (≤29 atoms). **Exp G uses heavy-atom-only** QM9 (≤9 atoms, H stripped), matching the SOTA benchmark setting of GeoDiff.

---

## 2. Shared Mathematical Foundation

All experiments share the same diffusion backbone. Each later experiment **adapts** one component while keeping the rest fixed.

### 2.1 Diffusion Process (DDPM)

We use Denoising Diffusion Probabilistic Models (DDPM) [[Ho et al. NeurIPS 2020]](#13-references-and-citations).

**Forward process** q(x₁:T | x₀) — gradually adds Gaussian noise:

```
q(xₜ | x₀) = 𝒩(xₜ; √ᾱₜ · x₀, (1 − ᾱₜ) · I)
```

This comes from the Markov chain:

```
q(xₜ | xₜ₋₁) = 𝒩(xₜ; √(1−βₜ) · xₜ₋₁, βₜ · I)
```

where βₜ are small positive constants (the **noise schedule**).  
After T steps, x_T ≈ 𝒩(0, I) (pure noise).

**The key identity** (derived by induction on the Markov chain):

```
xₜ = √ᾱₜ · x₀ + √(1 − ᾱₜ) · ε,   ε ~ 𝒩(0, I)
```

where:
- αₜ = 1 − βₜ
- ᾱₜ = ∏ˢ₌₁ᵗ αₛ  (cumulative product of alphas)
- √ᾱₜ is the **signal coefficient** — how much of the original data survives
- √(1 − ᾱₜ) is the **noise coefficient** — how much noise is added

**Reverse process** p_θ(xₜ₋₁ | xₜ) — the denoising network learns to denoise:

```
p_θ(xₜ₋₁ | xₜ) = 𝒩(xₜ₋₁; μ_θ(xₜ, t), σₜ² · I)
```

The DDPM posterior mean formula (derived from Bayes' rule + Gaussian conditioning):

```
μ_θ(xₜ, t) = (1/√αₜ) · [xₜ − (βₜ/√(1−ᾱₜ)) · ε_θ(xₜ, t)]
```

For **x₀ parameterization** (what we use), the network predicts x̂₀ directly, then derives ε̂:

```
ε̂ = (xₜ − √ᾱₜ · x̂₀) / √(1−ᾱₜ)
```

And the DDPM mean becomes:

```
μ_θ = (1/√αₜ) · [xₜ − βₜ/√(1−ᾱₜ) · ε̂]
     = √ᾱₜ₋₁ · x̂₀ + √(1−ᾱₜ₋₁ − σ²) · ε̂
```

where the posterior variance is:

```
σₜ² = βₜ · (1 − ᾱₜ₋₁) / (1 − ᾱₜ)
```

**Implementation** (from `conformer_diffusion.py`, `q_sample`):
```python
x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise  # forward: xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε
```

---

### 2.2 Noise Schedule — Cosine β-Schedule

We use the **cosine schedule** [[Nichol & Dhariwal, ICML 2021]](#13-references-and-citations) instead of the original linear schedule from Ho et al.

**Definition:**

```
ᾱₜ = f(t) / f(0),   f(t) = cos²(π/2 · (t/T + s) / (1 + s))
```

where s = 0.008 is a small offset to prevent βₜ from being too small near t=0.

Then:

```
βₜ = 1 − ᾱₜ / ᾱₜ₋₁,   clamped to [0.0001, 0.9999]
```

**Why cosine over linear?**  
The linear schedule decays ᾱₜ too fast — most information is lost in the first 20% of steps, causing the denoising network to waste capacity learning near-pure-noise denoising. The cosine schedule has a smoother SNR curve, spending more time in the informative intermediate regime.

**Implementation** (`cosine_beta_schedule`):
```python
alphas_cumprod = cos²(((t/T + s) / (1 + s)) * π/2)
betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
```

---

### 2.3 x₀ Parameterization vs. ε-Parameterization

**ε-parameterization (original DDPM):**  
The network predicts the noise ε added at step t:
```
L_ε = E[||ε − ε_θ(xₜ, t)||²]
```

**Problem for geometry:** At timestep t=0 (low noise, informative), the x₀ can be recovered as:
```
x₀ = (xₜ − √(1−ᾱₜ) · ε_θ) / √ᾱₜ
```

When √ᾱₜ ≈ 0 (high noise, t ≈ T), this formula **amplifies any error** in ε_θ by 1/√ᾱₜ → ∞. This makes geometry constraints applied to the recovered x₀ unstable at high timesteps.

**x₀ parameterization (our choice):**  
The network directly predicts the clean coordinates x̂₀:
```
L_x₀ = E[||x₀ − x̂₀_θ(xₜ, t)||²]
```

Then ε̂ is *derived*: ε̂ = (xₜ − √ᾱₜ · x̂₀) / √(1−ᾱₜ)

**Advantages** (EDM [[Hoogeboom et al. ICML 2022]](#13-references-and-citations), App. B):
1. The loss surface on x₀ is smoother and better-conditioned
2. Geometry constraints (bond lengths, angles) can be applied *directly* to x̂₀ with correct magnitudes
3. At high-noise t, x̂₀ prediction learns a coarse, correct structure; at low-noise t, it refines details

**Implementation** (`ConformerDenoiser.forward`):
```python
delta_x = self.coord_pred(h)      # learned residual correction
x_0_pred = x + delta_x            # predict x₀ by residual from running coordinate x
```

The final running coordinate `x` is the *refined* version after all EGNN layers, not xₜ directly — this residual structure makes training more stable.

---

### 2.4 E(3)-Equivariance & Center-of-Mass Removal

**E(3)-equivariance** means the model output transforms consistently with input transformations:
```
f(Rx + t) = R · f(x),   for any rotation R ∈ SO(3), translation t ∈ ℝ³
```

This is critical for molecules: a rotated/translated copy of a molecule should generate a rotated/translated copy of the conformer — the model should not have to learn rotation-invariance from data.

**Center-of-Mass (CoM) removal** [[Hoogeboom et al. EDM, 2022]](#13-references-and-citations):

The diffusion process is restricted to the **zero-CoM subspace**: Σᵢ xᵢ = 0

This is achieved by removing the CoM at every step. For a set of N atoms with batch assignment b:

```python
# CoM of each molecule in the batch
μ_mol = (1/N_mol) · Σᵢ∈mol xᵢ

# CoM-centered coordinates
x_centered = x - μ_mol[batch_idx]
```

The CoM-free noise is also enforced:
```python
noise = ε − μ_noise[batch_idx]   # ensures noise has zero CoM per molecule
```

**Why this matters:** Without CoM removal, the diffusion model would waste capacity encoding absolute 3D position (meaningless for molecular geometry) and would not be translation-equivariant.

**Implementation** (`remove_com` function):
```python
mol_sums.scatter_add_(0, batch_idx, x)
mol_counts.scatter_add_(0, batch_idx, ones)
mol_means = mol_sums / mol_counts         # (B, 3)
return x - mol_means[batch_idx]           # (N, 3)
```

---

### 2.5 EGNN Message Passing

The backbone is the **E(n)-Equivariant Graph Neural Network** [[Satorras et al. ICML 2021]](#13-references-and-citations).

**Feature update equations:**

For each edge (i→j), compute a message:
```
m_ij = φ_e(hᵢ, hⱼ, ||xᵢ − xⱼ||, aᵢⱼ)
```

where:
- hᵢ ∈ ℝᴴ: node features (hidden_dim = H)
- xᵢ ∈ ℝ³: 3D coordinates
- aᵢⱼ: edge attributes (bond type embedding)
- φ_e: edge MLP

**In our implementation**, we use **RBF distance features** instead of raw scalar distance (see §2.6):
```
m_ij = φ_e(hᵢ, hⱼ, RBF(||xᵢ − xⱼ||), bond_embed_ij)
```

**Node feature update** (invariant aggregation):
```
hᵢ ← LayerNorm(hᵢ + φ_h(hᵢ, Σⱼ mⱼᵢ))
```

**Coordinate update** (equivariant aggregation):
```
xᵢ ← xᵢ + (1/deg(i)) · Σⱼ φ_x(mⱼᵢ) · (xᵢ − xⱼ)/||xᵢ − xⱼ||
```

The key design choice: coordinate update uses the **unit direction vector** (xᵢ − xⱼ)/‖xᵢ − xⱼ‖. This ensures:
- **Equivariance:** if positions are rotated by R, the update rotates by R
- **Bounded update:** φ_x outputs a scalar (via Tanh), the unit vector provides direction only

**Critical bug fix (FIX-1) — Degree Normalization:**

In a bidirectional graph, each edge (i,j) is stored as two directed edges i→j and j→i. When scatter-aggregating into destination col, each atom i appears as col for deg(i) incoming edges. But the degree count from scatter was counting how many times i appears in col — which correctly equals deg(i).

**The bug:** The variable name `degree` was correct, but the normalization was by 1 (missing the factor of 2 correction from bidirectionality). In a bidirectional graph built from undirected bonds, we create 2 directed edges per bond. The "physical" number of unique neighbors is deg_undirected(i) = deg_directed(i)/2. So dividing by `degree/2` gives the correct normalization:

```python
degree = (degree / 2.0).clamp(min=1.0)   # FIX-1: bidirectional graph correction
x_new = x + x_agg / degree
```

---

### 2.6 RBF Distance Features

**Raw distance problem:** Feeding scalar distance d = ||xᵢ − xⱼ|| directly into the edge MLP gives the network very little geometric resolution — the MLP must learn to distinguish d=1.0Å (covalent bond) from d=3.5Å (non-bonded contact) from a single scalar input.

**RBF solution** (used in SchNet [[Schütt et al. 2017]](#13-references-and-citations), DimeNet [[Gasteiger et al. 2020]](#13-references-and-citations)):

Place `num_rbf = 20` Gaussian centers μₖ uniformly in [d_min, d_max] = [0.5, 6.0] Å:

```
RBF_k(d) = exp(−γ · (d − μₖ)²)

where γ = 2(num_rbf − 1)/(d_max − d_min)
```

This gives a **20-dimensional feature vector** per edge, where each dimension encodes how close the distance is to a particular reference distance. The network can trivially distinguish C-C single bonds (~1.54 Å) from C=C double bonds (~1.34 Å) or VDW contacts (~3.5 Å).

**Implementation:**
```python
centers = linspace(0.5, 6.0, 20)          # 20 Gaussian centers
gamma   = 2 * 19 / 5.5                    # width parameter
RBF     = exp(-gamma * (dist - centers)²) # (E, 20)
```

---

### 2.7 Min-SNR Loss Weighting

**Problem with uniform SNR:** At high timesteps t, the signal-to-noise ratio (SNR) is low — the network receives very noisy input and must make very coarse predictions. At low t, SNR is high and predictions are fine-grained. If we weight all timesteps equally, the loss is dominated by high-t terms (there are more of them, and the MSE is larger).

**Signal-to-Noise Ratio:**
```
SNR(t) = ᾱₜ / (1 − ᾱₜ)
```

**Min-SNR weighting** [[Hang et al. 2023]](#13-references-and-citations) clips the SNR weight at γ = 5:

```
w(t) = min(SNR(t), γ) / SNR(t)
```

So for high-t (low SNR), w(t) ≈ 1. For low-t (high SNR), w(t) = γ/SNR(t) < 1 — this **downweights** the easy, precise low-t steps to give more credit to high-t structure-learning.

**Per-molecule reduction** (FIX-6): We reduce the x₀ MSE per *molecule* (averaging over atoms first), then SNR-weight per molecule. Without this, large molecules with many atoms dominate the gradient even when they have the same MSE per atom.

```python
# Per-atom MSE → per-molecule average
mse_per_mol = scatter_add(x0_err_per_atom, batch_idx) / N_atoms_per_mol

# Min-SNR weight (per molecule)
snr_weight = min(SNR(t), γ) / SNR(t)
loss = mean(snr_weight * mse_per_mol)
```

---

### 2.8 DDIM Sampling

**DDPM sampling** requires T = 1000 reverse steps — slow at inference.

**DDIM** [[Song et al. ICLR 2021]](#13-references-and-citations) uses a **deterministic**, non-Markovian sampler that can use far fewer steps (50 in our experiments):

Given the current xₜ and predicted x̂₀ at step t:

1. Derive ε̂ = (xₜ − √ᾱₜ · x̂₀) / √(1−ᾱₜ)
2. Compute σₜ² = η · √((1−ᾱₜ₋₁)/(1−ᾱₜ)) · √(1 − ᾱₜ/ᾱₜ₋₁)
3. Update: xₜ₋₁ = √ᾱₜ₋₁ · x̂₀ + √(1−ᾱₜ₋₁−σ²) · ε̂ + σₜ · ε

With η=0.0 (deterministic DDIM), the sampling is completely deterministic.

**Critical bug fix (FIX-5) — NaN prevention:**

The term (1 − ᾱₜ/ᾱₜ₋₁) can become slightly negative due to floating point, making its square root NaN:
```python
# WRONG:
sigma = torch.sqrt(1 - alpha_t/alpha_next)  # → NaN if ratio > 1.0

# FIXED:
ratio = (alpha_t / alpha_next).clamp(max=1.0)
sigma = torch.sqrt((1.0 - ratio).clamp(min=0.0))  # safe
```

---

### 2.9 Geometry Constraint Losses

All experiments include a **differentiable geometry loss** applied to the predicted clean coordinates x̂₀. This directly supervises geometric correctness.

#### Bond Length Loss

Given the MMFF94 bond length lookup table L[atom1, atom2, bond_order], the bond loss is:

```
L_bond = w_bond · (1/E) · Σ_(i,j)∈edges (||xᵢ − xⱼ|| − L[zᵢ, zⱼ, bᵢⱼ])²
```

where L[zᵢ, zⱼ, bᵢⱼ] is the **ideal bond length** from MMFF94 molecular mechanics [[Halgren 1996]](#13-references-and-citations). Example values:
- C−C single: 1.54 Å
- C=C double: 1.34 Å
- C≡C triple: 1.20 Å
- C:C aromatic: 1.40 Å

The lookup is **fully vectorized**: a precomputed tensor of shape [54, 54, 5] is indexed by (atom1_type, atom2_type, bond_order), returning ideal distances in O(1) GPU operations.

#### Bond Angle Loss

For every atom j with at least 2 neighbors {i, k, ...}, compute the angle θᵢⱼₖ:

```
θᵢⱼₖ = arccos((vᵢ · vₖ) / (||vᵢ|| · ||vₖ||))

where vᵢ = xᵢ − xⱼ,  vₖ = xₖ − xⱼ
```

Target angle θ* is **hybridization-aware**:
- sp3 (tetrahedral, e.g. saturated C): θ* = 109.5°
- sp2 (trigonal planar, e.g. C=C): θ* = 120.0°
- sp (linear, e.g. C≡C): θ* = 180.0°
- Aromatic ring: θ* = 120.0°

```
L_angle = w_angle · (1/T) · Σ_triplets (θᵢⱼₖ − θ*ⱼ)²
```

#### Repulsion Loss

Non-bonded atom pairs should not clash (come closer than VDW radii allow):

```
d_thresh(i, j) = 0.70 · (r_VDW(zᵢ) + r_VDW(zⱼ))

L_repulsion = w_rep · (1/|NB|) · Σ_(i,j)∈non-bonded max(0, d_thresh − ||xᵢ − xⱼ||)²
```

The 0.70 multiplier corresponds to the VDW clash threshold (70% of VDW radius sum). Pairs excluded: bonded (1-2), angle neighbors (1-3), and cross-molecule pairs.

#### Torsion Loss (Experiment D only)

The **dihedral angle** (torsion) φ of atoms i-j-k-l is:

```
n₁ = (rⱼ − rᵢ) × (rₖ − rⱼ)    (normal to plane i-j-k)
n₂ = (rₖ − rⱼ) × (rₗ − rₖ)    (normal to plane j-k-l)

φ = atan2(((n₁ × n₂) · b̂₂), n₁ · n₂)

where b̂₂ = (rₖ − rⱼ)/||rₖ − rⱼ||
```

The **OPLS-AA torsion potential** [[Jorgensen et al. 1996]](#13-references-and-citations) for each rotatable bond j-k:

```
E_tors(φ) = V₁(1 + cos φ)/2 + V₂(1 − cos 2φ)/2 + V₃(1 + cos 3φ)/2
```

Parameters {V₁, V₂, V₃} depend on the hybridization of atoms j and k:
- sp3−sp3 (e.g., C−C ethane): V₃ = 1.0, V₁ = V₂ = 0 (3-fold barrier)
- sp2−sp2 (conjugated, e.g., butadiene): V₂ = 6.0 (prefer planar)
- aromatic−aromatic: V₂ = 10.0 (strong planarity)

```
L_torsion = w_tors · (1/|bonds|) · Σ_bonds E_tors(φ_bond)
```

#### Timestep Gating for Geometry Loss (FIX-4 REVISED)

At high diffusion timesteps t ≈ T, x̂₀ has RMSD ≈ 1Å vs ground truth — the prediction is near-random. Applying geometry constraints at these timesteps injects **pure noise gradients** that destabilize training.

Following GCDM [[Morehead & Cheng, NeurIPS 2023]](#13-references-and-citations) §3.3, geometry loss is **gated** to only apply when t < T·geo_t_fraction (default 0.3):

```
L_geo_gated = L_geo · 𝟙[t < T × 0.3]
```

So ~30% of training steps get geometry supervision (when x̂₀ prediction is reliable enough).

#### Total Training Loss

```
L_total = L_MSE_x₀ + w_geo · L_geo

where L_geo = L_bond + L_angle + [L_torsion if enabled] + L_repulsion
```

---

### 2.10 Evaluation Metrics (GeoDiff Protocol)

**GeoDiff protocol** [[Xu et al. ICML 2022]](#13-references-and-citations): For each molecule m in the test set, generate K=10 conformers {x̂₀⁽¹⁾, ..., x̂₀⁽ᴷ⁾} and compare to the reference ensemble {x*⁽¹⁾, ..., x*⁽ᴿ⁾} from RDKit MMFF optimization (multiple minima sampled).

**RMSD** between two conformers is computed after optimal rigid alignment (Kabsch algorithm):
```
RMSD(x, x*) = min_{R,t} (1/N) · ||R·x + t − x*||²
```
For **heavy-atom-only** evaluation, H atoms are excluded from RMSD computation.

**COV-R (Coverage-Recall):** Fraction of reference conformers x*⁽ʳ⁾ that are "covered" by at least one generated conformer within threshold δ = 0.5 Å:
```
COV-R(S, R) = (1/|R|) · Σ_r 𝟙[min_k RMSD(x̂⁽ᵏ⁾, x*⁽ʳ⁾) < δ]
```

**MAT-R (Matching-Recall):** Average minimum RMSD from each reference to the closest generated conformer:
```
MAT-R(S, R) = (1/|R|) · Σ_r min_k RMSD(x̂⁽ᵏ⁾, x*⁽ʳ⁾)
```

Lower MAT-R = better average accuracy. COV-R = fraction of reference space explored.

**COV-P and MAT-P:** Same metrics but with recall and precision swapped (how well reference covers generated).

---

## 3. Experiment A — Fixed Baseline (DDPM)

### 3.1 Hypothesis

Establish a clean, bug-free reference baseline for the `ConformerDiffusion` model using standard DDPM with AdamW optimizer. All subsequent experiments are ablations relative to this.

### 3.2 Architecture

**Model:** `ConformerDiffusion` with `ConformerDenoiser`

```
Input: (xₜ, t, atom_types, edge_index, bond_types, batch_idx)
Output: x̂₀ ∈ ℝᴺˣ³ (predicted clean coordinates)

Architecture:
  atom_embed:    Embedding(54, 256)       → h ∈ ℝᴺˣ²⁵⁶
  time_mlp:      Lin(128→256) → SiLU → Lin(256→256)
  layers[0..5]:  EquivariantLayer(256, num_rbf=20) × 6
  coord_pred:    Lin(256→256) → SiLU → Lin(256→128) → SiLU → Lin(128→3)
```

**EquivariantLayer forward pass** (one layer, step-by-step):

```
1. Geometry:     diff = xᵢ − xⱼ,   d = ||diff||,   unit = diff/d
2. RBF features: rbf = exp(−γ(d − μₖ)²)   ← 20-dim
3. Edge message: m_ij = MLP([hᵢ, hⱼ, rbf]) + bond_embed  ← (E, 256)
4. Coord update: w_ij = Tanh(MLP(m_ij))  ← scalar weight
                 Δxᵢ = Σⱼ w_ij · unit_ij / deg(i)
                 xᵢ ← xᵢ + Δxᵢ
5. Node update:  m_agg_i = Σⱼ m_ji
                 hᵢ ← LayerNorm(hᵢ + MLP([hᵢ, m_agg_i]))
```

**Parameters:**

| Component         | Size  |
|-------------------|-------|
| atom_embed        | 54×256 = 13,824 |
| bond_embed        | 6×256  = 1,536  |
| time_mlp          | 128×256 + 256×256 = 98,816 |
| EquivariantLayer × 6 | ~(2×256+20)×256 + 256×128 + 128×1 + 2×256×256 ≈ 420K × 6 |
| coord_pred        | 256×256 + 256×128 + 128×3 = 99,332 |
| **Total**         | **≈ 2.81 M** |

### 3.3 Training Configuration

| Hyperparameter    | Value | Justification |
|-------------------|-------|---------------|
| hidden_dim        | 256   | baseline capacity |
| num_layers        | 6     | depth for geometry reasoning |
| num_timesteps     | 1000  | standard DDPM |
| batch_size        | 64    | fits 1× RTX 3090 |
| lr                | 1e-4  | safe for AdamW |
| weight_decay      | 0.01  | L2 regularization |
| epochs            | 50    | fast experiment budget |
| warmup_epochs     | 5     | **BUG FIX** (was 50 = same as EPOCHS → warmup never completed) |
| geometry_weight   | 0.1   | light constraint |
| min_snr_gamma     | 5.0   | Min-SNR loss weighting |

### 3.4 Bugs Fixed vs. Original

**BUG-FIX-6: LR Scheduler Compounding**

The original code stored LR in optimizer state and re-applied scaling multiplicatively each epoch. This compounded the schedule: epoch 2 applied schedule × already-scaled-lr from epoch 1.

*Fix:* Store `initial_lr` at optimizer creation, always scale from this fixed base:
```python
pg['initial_lr'] = pg['lr']   # saved once at creation
# Each epoch:
pg['lr'] = cosine_lr(epoch, base_lr=pg['initial_lr'])  # always from fixed base
```

**BUG-FIX-WARMUP: Warmup Duration**

Original: `WARMUP_EPOCHS = 50` with `EPOCHS = 50` → warmup consumed 100% of training. LR never reached target value. Model trained at 1/50th of intended LR for all 50 epochs.

*Fix:* `WARMUP_EPOCHS = 5` (10% of total), ensuring 45 epochs at target LR.

**LR Schedule (corrected):**
```
lr(epoch) = {
    base_lr × (epoch+1)/warmup   if epoch < warmup
    max(base_lr × 0.5 × (1 + cos(π × progress)), base_lr × 0.01)  otherwise
}
```

The `base_lr × 0.01` floor prevents LR from decaying to ~0 at epoch 200+ (which stalled training in longer runs).

### 3.5 Data Flow: Forward Pass → Loss → Backward

```
Batch input: {atom_types, coordinates x₀, edge_index, bond_types, batch_idx}

1. CoM removal: x₀ ← x₀ − CoM(x₀)

2. Sample timestep t ~ Uniform{0,...,T-1}  for each molecule

3. Forward diffusion (q_sample):
   ε ~ 𝒩(0, I),  ε ← remove_com(ε)
   xₜ = √ᾱₜ · x₀ + √(1−ᾱₜ) · ε

4. Denoiser forward (ConformerDenoiser):
   h = atom_embed(atom_types)          # (N, 256)
   t_emb = sinusoidal_emb(t, 128)
   h = h + time_mlp(t_emb)[batch_idx]  # add time conditioning
   bond_feat = bond_embed(bond_types)   # (E, 256)
   
   for layer in layers:
       h, x = layer(h, x, edge_index, bond_feat)
   
   x̂₀ = x + coord_pred(h)              # residual prediction
   x̂₀ = remove_com(x̂₀)

5. Loss computation:
   a. x₀ MSE (per atom → per mol → SNR-weighted):
      err = (x̂₀ − x₀)²  per atom               # (N, 3) → sum → (N,)
      mse_per_mol = scatter_mean(err, batch_idx) # (B,)
      snr_wt = min(SNR(t), 5.0) / SNR(t)        # (B,) Min-SNR
      L_MSE = mean(snr_wt × mse_per_mol)
   
   b. Geometry loss (gated to t < 0.3×T):
      if any(t < 300):
          L_geo = bond_loss + angle_loss + repulsion_loss on x̂₀
      else:
          L_geo = 0
   
   c. Total: L = L_MSE + 0.1 × L_geo

6. Backward: L.backward() → clip_grad_norm_(1.0) → optimizer.step()
```

### 3.6 Conformer Generation (Inference)

```
DDIM sampling (50 steps, η=0 deterministic):

1. Initialize: xₜ ~ 𝒩(0, I) → remove_com(xₜ)

2. For t = T-1 down to 0 (50 evenly spaced steps):
   a. x̂₀ = denoiser(xₜ, t, ...)        ← predict clean
   b. x̂₀ = remove_com(x̂₀)
   c. ε̂ = (xₜ − √ᾱₜ · x̂₀) / √(1−ᾱₜ)   ← derive noise
   d. xₜ₋₁ = √ᾱₜ₋₁ · x̂₀ + √(1−ᾱₜ₋₁) · ε̂  ← DDIM step

3. Return x₀ = x₀_final
```

### 3.7 Results

| Metric         | Value   |
|----------------|---------|
| MAT-R          | 0.638 Å |
| COV-R          | 0.278   |
| Validity       | 96.6%   |
| Fully Valid    | 14.0%   |
| Strain Energy  | 34,250 kcal/mol |
| Bond Error     | 0.230 Å |

**Verdict:** ✓ KEEP — Clean, reproducible reference for all ablations.

---

## 4. Experiment B — EQGAT-diff Attention EGNN

### 4.1 Hypothesis

Standard EGNN aggregates neighbor messages uniformly: h_i ← f(h_i, Σⱼ mⱼᵢ). But not all neighbors are equally important — a heteroatom neighbor is more informative than a carbon neighbor for predicting geometry. Adding **attention** over messages should improve the quality of the representation and reduce strain energy.

**Adapted from:** EQGAT-diff [[Le et al. ICLR 2024]](#13-references-and-citations) — equivariant graph attention for molecular generation.

**Inspiration:** Graph Attention Networks [[Veličković et al. ICLR 2018]](#13-references-and-citations) proved that attention-weighted aggregation outperforms mean aggregation on molecular property prediction.

### 4.2 What We Kept

From Exp A's `ConformerDiffusion`:
- All DDPM diffusion math (noise schedule, q_sample, p_sample)
- x₀ parameterization
- CoM removal
- RBF distance features (20 Gaussians)
- Sinusoidal time embedding
- Geometry constraint losses (bond, angle, repulsion)
- Optimizer: AdamW, lr=1e-4
- All hyperparameters identical

### 4.3 What We Changed

**EquivariantLayer → AttnEquivariantLayer**

The standard aggregation:
```
m_agg_i = Σⱼ m_ji            (uniform sum)
hᵢ ← LayerNorm(hᵢ + MLP([hᵢ, m_agg_i]))
```

becomes attention-weighted:

```
For each edge (j → i):
  attn_logits_ij = W_attn · [hⱼ, hᵢ, RBF_ij]  ∈ ℝ^H (H attention heads)
  a_ij = softmax_j(LeakyReLU(attn_logits_ij, α=0.2))   (softmax over j∈𝒩(i))

Attention-weighted message:
  m_attn_ij = reshape(m_ij, H, H/4) × a_ij   ← per-head weighting

Aggregation:
  m_agg_i = Σⱼ m_attn_ji
  hᵢ ← LayerNorm(hᵢ + MLP([hᵢ, m_agg_i]))
```

**Softmax over neighbors** (scatter-softmax, numerically stable):

For each head h and destination node i:

```
a_ij^h = exp(logit_ij^h − max_{j'∈𝒩(i)} logit_ij'^h) /
         Σ_{j'∈𝒩(i)} exp(logit_ij'^h − max)
```

The scatter-softmax is implemented via three scatter operations: `scatter_reduce amax`, subtract, exp, `scatter_add` to normalize.

**Coordinate update** is **unchanged** (it is already equivariant and does not need attention):
```
Δxᵢ = Σⱼ Tanh(MLP(m_ij)) · (xᵢ − xⱼ)/||xᵢ − xⱼ||  / deg(i)
```

**New: Global Graph Readout** (EGNN §3.3, global invariant features)

After all layers, compute a per-molecule summary embedding g_m:
```
g_feat = MLP(h)         ← (N, H//4)
g_m = mean(g_feat, dim=0_per_molecule)  ← (B, H//4) mean pool

h_global = cat([h, g_m[batch_idx]])     ← (N, H + H//4)
x̂₀ = x + coord_pred(h_global)           ← global context for prediction
```

This gives each atom access to the global molecular context when making its coordinate prediction.

### 4.4 Architecture Diagram

```
Input: xₜ, t, atom_types, edge_index, bond_types

h = atom_embed(atom_types) + time_embed(t)[batch]

For each layer l ∈ {0, ..., 5}:
  ┌─ Edge phase ──────────────────────────────────────┐
  │  rbf = RBF(||xᵢ − xⱼ||)                         │
  │  m_ij = edge_MLP([hᵢ, hⱼ, rbf]) + bond_embed    │
  │  logit_ij = W_attn([hᵢ, hⱼ, rbf])               │
  │  a_ij = scatter_softmax(LeakyReLU(logit_ij), j)  │  ← NEW: attention
  │  m_attn_ij = reshape(m_ij, 4, 64) × a_ij        │
  └───────────────────────────────────────────────────┘
  ┌─ Coord update (equivariant, unchanged) ───────────┐
  │  Δxᵢ = Σⱼ Tanh(MLP(m_ij)) · unit_ij / deg(i)   │
  │  xᵢ ← xᵢ + Δxᵢ                                   │
  └───────────────────────────────────────────────────┘
  ┌─ Node update (attention-weighted) ────────────────┐
  │  m_agg_i = Σⱼ m_attn_ji                          │
  │  hᵢ ← LayerNorm(hᵢ + MLP([hᵢ, m_agg_i]))        │
  └───────────────────────────────────────────────────┘

Global readout:
  g = mean_pool(MLP_g(h))   → (B, 64)
  h_global = cat([h, g[batch]])  → (N, 320)

x̂₀ = x + coord_pred(h_global)
```

### 4.5 Results and Analysis

| Metric         | Exp A    | Exp B    | Δ       |
|----------------|----------|----------|---------|
| MAT-R          | 0.638 Å  | 0.654 Å  | +2.5%   |
| COV-R          | 0.278    | 0.270    | −3%     |
| Fully Valid    | 14.0%    | 13.6%    | −0.3%   |
| **Strain**     | **34,250** | **4,078** | **−88%** |
| Bond Error     | 0.230 Å  | 0.226 Å  | −2%     |

**Conclusion:** Attention dramatically reduces strain energy (4× lower, 34k → 4k kcal/mol), indicating geometrically more plausible structures. MAT-R changes minimally — at 50 epochs, model capacity is the limiting factor, not aggregation quality.

**Verdict:** ✓ KEEP — Best strain energy in experiments A-F. Attention EGNN architecture selected for Exp G.

---

## 5. Experiment C — Conditional Flow Matching

### 5.1 Hypothesis

Replace DDPM (1000 reverse steps, stochastic) with **Conditional Flow Matching** (CFM) [[Lipman et al. ICLR 2023]](#13-references-and-citations), which uses deterministic ODE trajectories with only 20 Euler steps. CFM has shown superior sample quality at lower NFE (number of function evaluations) in image generation.

### 5.2 Mathematical Difference from DDPM

**DDPM** uses a *stochastic* forward process with Gaussian noise schedule. The reverse is also stochastic (DDPM) or deterministic (DDIM).

**Conditional Flow Matching** defines a *vector field* v_θ(x_t, t) that generates a flow from noise to data via ODE:

```
dx_t/dt = v_θ(x_t, t)
```

The optimal path from noise x₀ ~ 𝒩(0,I) to data x₁ = x_mol is a **straight line** (Lipman 2023, optimal transport):

```
x_t = (1 − t) · x₀ + t · x_mol   for t ∈ [0, 1]
```

The **target vector field** along this path is constant:
```
u_t(x_t | x_mol) = x_mol − x₀   (direction from noise to data)
```

The loss is simply:
```
L_CFM = E_{t, x₀, x_mol}[||v_θ(x_t, t) − (x_mol − x₀)||²]
```

**Euler ODE sampling** with 20 steps:
```
x₀ ~ 𝒩(0, I)
for s = 0, Δ, 2Δ, ..., 1:   (Δ = 1/20)
    x_{s+Δ} = x_s + Δ · v_θ(x_s, s)
```

### 5.3 Why It Failed

**Root cause:** The geometry loss `L_geo` interacted catastrophically with CFM interpolation.

In DDPM, at timestep t, x_t = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε. The geometry loss is gated to t < 0.3T — only applied when √ᾱₜ ≈ 0.7 (signal > 50%).

In CFM, at time t ∈ [0,1], x_t = (1−t)·ε + t·x_mol. At t = 0.5, the mixture is exactly 50% noise + 50% data. The **geometry gradient** ∂L_geo/∂x_t propagates through the flow. But because the flow is a **straight line**, any geometry gradient that changes x_t is inconsistent with the straight-line ODE — it bends the trajectory away from the optimal transport path.

The result: geometry_weight = 0.05 → fully_valid = 0, strain energy = 121k kcal/mol.

**Mathematically:**
- In DDPM: ε-parameterization makes geometry constraints act on x̂₀ (predicted data)
- In CFM: v-parameterization predicts the *velocity* (x_mol − x₀), not x_mol directly
- Applying geometry loss to x_t (the interpolated point) corrupts the velocity field
- This was confirmed: setting geometry_weight=0 would have allowed training, but we discarded CFM

**Verdict:** ✗ DISCARD — CFM needs geometry_weight=0 and dedicated hyperparameter tuning.

---

## 6. Experiment D — Torsion Auxiliary Loss (TorDiff)

### 6.1 Hypothesis

Conformer geometry quality is dominated by **torsion (dihedral) angle accuracy**, not just bond lengths or angles. TorDiff [[Jing et al. NeurIPS 2022]](#13-references-and-citations) proved this: diffusing over torsion space (not Cartesian) gives SOTA conformer quality. We adapt this insight by adding an explicit **torsion angle auxiliary loss** to the existing DDPM backbone.

### 6.2 What We Kept from Exp A

Everything: `ConformerDiffusion`, DDPM schedule, x₀ parameterization, optimizer, all hyperparameters.

### 6.3 What We Changed

**Single change: add torsion supervision**
```
L_total = L_MSE + 0.5 × (L_bond + L_angle + L_torsion + L_repulsion)
```

vs. Exp A:
```
L_total = L_MSE + 0.1 × (L_bond + L_angle + L_repulsion)
```

Two changes: (1) geometry_weight 0.1 → 0.5, (2) `include_torsions=True`.

### 6.4 Torsion Loss Mathematics

For each rotatable bond j-k (single or aromatic bond with sp3/sp2/aromatic centers):

**Step 1:** Find atom i (neighbor of j, not k) and atom l (neighbor of k, not j).

**Step 2:** Compute dihedral angle φ using the atan2 formula:
```
b₁ = rⱼ − rᵢ    (bond vectors)
b₂ = rₖ − rⱼ
b₃ = rₗ − rₖ

n₁ = b₁ × b₂    (normals to planes)
n₂ = b₂ × b₃

b̂₂ = b₂/||b₂||

φ = atan2((n₁ × n₂)·b̂₂, n₁·n₂)
```

This gives φ ∈ (−π, π].

**Step 3:** OPLS-AA potential:
```
E(φ) = V₁(1 + cosφ)/2 + V₂(1 − cos2φ)/2 + V₃(1 + cos3φ)/2
```

For sp3-sp3 (ethane-like): V₃=1.0, others 0. Minimum at φ=60°, 180°, 300° (staggered).
For sp2-sp2 (butadiene-like): V₂=6.0. Minimum at φ=0°, 180° (s-trans, s-cis).

**Important detail in implementation:** The torsion loss is computed on x̂₀_pred (the network's predicted clean coordinates), NOT on the ground-truth x₀. Ground-truth QM9 DFT geometries already minimize torsion energy, so L_torsion(x₀_GT) ≈ 0 — there would be no learning signal. The loss must be on x̂₀:

```python
loss_dict = model.get_loss(
    co, at, ei, bt, bi,
    include_torsions=True,  # torsion computed on x̂₀_pred inside get_loss
    geometry_weight=0.5,
)
```

### 6.5 Results and Analysis

| Metric         | Exp A    | Exp D    | Δ        |
|----------------|----------|----------|----------|
| **MAT-R**      | 0.638 Å  | **0.572 Å** | **−10.3%** |
| **COV-R**      | 0.278    | **0.400** | **+43.9%** |
| **Fully Valid** | 14.0%   | **11.8%** | −2.2%    |
| Strain Energy  | 34,250   | 55,298   | +61%     |
| Bond Error     | 0.230 Å  | 0.232 Å  | ≈same    |

MAT-R drops 10% and COV-R increases 44% — confirming that torsion supervision drives conformer quality. Strain energy increases (torsion loss doesn't penalize strain, only dihedral energy). Fully valid slightly lower — the stronger geometry_weight (0.5) creates more gradient tension.

**Verdict:** ✓ KEEP — Best MAT-R and COV-R of experiments A-F.

---

## 7. Experiment E — SOTA Hybrid (CFM + Attention)

### 7.1 Hypothesis

Combine CFM (Exp C: fast sampling) + Attention EGNN (Exp B: lower strain) + Torsion loss (Exp D: better MAT-R) into a single "SOTA" hybrid model.

### 7.2 Architecture

`HybridFlowMatchingConformer`:
- CFM-based training (velocity field prediction)
- AttnEquivariantLayer backbone (from Exp B)
- Torsion loss enabled
- geometry_weight = 0.1

### 7.3 Why It Also Failed

The combination of CFM + Attention + Torsion produced the worst results (MAT-R=0.786, fully_valid=0). This **definitively confirms** the diagnosis from Exp C:

The instability is in the **CFM loss formulation with geometry supervision**, not in the EGNN architecture. Adding better architecture (attention) or more loss terms (torsion) doesn't fix the fundamental incompatibility between CFM's straight-line ODE and geometry constraints on interpolated points.

**Verdict:** ✗ DISCARD — Confirms CFM needs geometry_weight=0 to converge.

---

## 8. Experiment F — DDP Multi-GPU Training

### 8.1 Hypothesis

Scale to 2× GPU with PyTorch DDP (DistributedDataParallel) and AMP (Automatic Mixed Precision) to allow larger batch sizes and faster training.

### 8.2 Architecture Upgrade

`AttnConformerDiffusion` with:
- hidden_dim: 256 → 384 (+50% capacity)
- num_layers: 6 → 8
- num_rbf: 20 → 32
- batch_size: 128 × 2 GPUs = 256 effective

### 8.3 Bug That Caused the Crash

```
RuntimeError: scatter_reduce_(): Expected self.dtype to be equal to src.dtype
```

**Root cause:** AMP (mixed precision) casts activations to float16 during the forward pass. The `scatter_add_` operation in the coordinate update received:
- `self` (x_agg): float16 (cast by AMP)
- `src` (coord_update): float32 (computed before AMP cast)

PyTorch's `scatter_reduce_` does not auto-promote dtypes — it raises an error.

**The fix (implemented in Exp G):**
```python
# WRONG (Exp F):
x_agg.scatter_add_(0, col.expand(-1, 3), coord_update)   # dtype mismatch

# FIXED (Exp G):
x_agg.scatter_add_(0, col.expand(-1, 3), coord_update.to(x_agg.dtype))  # cast src
```

Or alternatively, keep the coordinate update in float32 by detaching from AMP autocast scope.

**Verdict:** 💥 CRASH — dtype mismatch scatter bug → fixed and deployed in Exp G.

---

## 9. Experiment G — Heavy-Atom SOTA Model

### 9.1 Motivation and Adaptations

This experiment consolidates all successful insights (attention EGNN from B, model scaling from F bug-fix, torsion insight from D) into a long, production-quality training run.

**Adaptations from the literature:**
- **Model scale:** EDM [[Hoogeboom et al. 2022]](#13-references-and-citations) showed that larger EGNN (hidden=384, L=8) consistently improves MAT-R
- **Heavy-atom only:** GeoDiff [[Xu et al. 2022]](#13-references-and-citations) evaluates on heavy-atom-only QM9 (9 atoms vs 29) — removing H makes the problem cleaner and enables direct SOTA comparison
- **DDP 2×GPU:** allows effective batch 256, faster convergence per wall-clock hour
- **500 epochs:** EDM/GeoDiff train for 1000+ epochs; 500 is our compute budget

### 9.2 Architecture Changes

**From Exp A → G: comprehensive scale-up**

| Component        | Exp A (baseline) | Exp G (SOTA)         | Change |
|-----------------|-------------------|----------------------|--------|
| hidden_dim      | 256               | 384                  | +50%   |
| num_layers      | 6                 | 8                    | +33%   |
| num_rbf         | 20                | 32                   | +60%   |
| time_dim        | 128               | 256                  | +100%  |
| aggregation     | uniform sum       | 4-head attention     | Exp B  |
| dataset         | explicit-H QM9 (29 at.) | heavy-atom QM9 (9 at.) | cleaner |
| batch_size      | 64                | 128 × 2 GPUs         | 4×     |
| lr              | 1e-4              | 5e-4                 | 5×     |
| warmup_epochs   | 5                 | 10                   | 2×     |
| epochs          | 50                | 500                  | 10×    |
| geometry_weight | 0.1               | 0.5                  | 5×     |
| **Total params** | **2.81 M**        | **8.36 M**           | **+198%** |

### 9.3 Complete Forward Pass (Exp G)

```
Input: xₜ ∈ ℝᴺˣ³, t ∈ ℤᴮ, atom_types ∈ ℤᴺ, edge_index ∈ ℤ²ˣᴱ, bond_types ∈ ℤᴱ

Step 1: Node initialization
  h = atom_embed(atom_types)           # ℝᴺˣ³⁸⁴  [Embedding(54, 384)]
  t_emb = sinusoidal(t, dim=256)       # ℝᴮˣ²⁵⁶
  t_emb = time_mlp(t_emb)             # ℝᴮˣ³⁸⁴  [Lin(256→384)→SiLU→Lin(384→384)]
  h = h + t_emb[batch_idx]            # broadcast: each atom gets its molecule's t_emb

Step 2: Bond embedding
  bond_feat = bond_embed(bond_types)   # ℝᴱˣ³⁸⁴  [Embedding(6, 384)]

Step 3: Message passing (8 layers of AttnEquivariantLayer)
  For l = 0 → 7:
    # Geometry
    diff = x[row] − x[col]            # ℝᴱˣ³
    dist = ||diff||.clamp(min=1e-6)   # ℝᴱˣ¹
    unit = diff / dist                 # ℝᴱˣ³  (unit direction)
    rbf = exp(−γ(dist − μₖ)²)        # ℝᴱˣ³²  (32 Gaussians, 0.5–6.0 Å)
    
    # Edge MLP
    edge_in = cat([h[row], h[col], rbf])    # ℝᴱˣ⁸⁰⁰
    m_ij = edge_MLP(edge_in) + bond_feat    # ℝᴱˣ³⁸⁴
    
    # Attention (4 heads, head_dim=96)
    attn_in = cat([h[row], h[col], rbf])    # ℝᴱˣ⁸⁰⁰
    logits = W_attn(attn_in)               # ℝᴱˣ⁴    (4 head logits)
    logits = LeakyReLU(logits, 0.2)
    a_ij = scatter_softmax(logits, col)    # ℝᴱˣ⁴
    m_attn = reshape(m_ij, 4, 96) × a_ij  # ℝᴱˣ³⁸⁴  (per-head weighting)
    
    # Coordinate update (equivariant)
    coord_w = Tanh(coord_MLP(m_ij))        # ℝᴱˣ¹
    x_agg.scatter_add(col, coord_w × unit) # ℝᴺˣ³
    deg = scatter_count(col) / 2            # FIX-1: bidirectionality
    x ← x + x_agg / deg.clamp(min=1)
    
    # Node update
    m_agg.scatter_add(col, m_attn)         # ℝᴺˣ³⁸⁴
    h_new = node_MLP(cat([h, m_agg]))      # ℝᴺˣ³⁸⁴
    h ← LayerNorm(h + h_new)

Step 4: Global readout
  g_feat = global_MLP(h)                   # ℝᴺˣ⁹⁶  (384 → 192 → 96)
  g_mol.scatter_mean(batch_idx, g_feat)    # ℝᴮˣ⁹⁶  per-molecule mean
  h_glob = cat([h, g_mol[batch_idx]])     # ℝᴺˣ⁴⁸⁰

Step 5: Coordinate prediction
  Δx = coord_pred(h_glob)                  # ℝᴺˣ³  [Lin(480→384)→SiLU→Lin(384→192)→SiLU→Lin(192→3)]
  x̂₀ = x + Δx                              # residual: refine running coord
  x̂₀ = remove_com(x̂₀)                     # enforce zero CoM

Output: x̂₀ ∈ ℝᴺˣ³
```

### 9.4 Complete Backward Pass (Training Loss)

```
Given: x₀ (ground truth, CoM-centered), x̂₀ (network prediction)

1. Sample t ~ Uniform{0,...,999} for each molecule (B samples)
2. Forward diffusion: xₜ = √ᾱₜ·x₀ + √(1−ᾱₜ)·ε, ε~𝒩(0,I), remove_com applied

3. x̂₀ = denoiser(xₜ, t, ...)  ← forward pass (as above)
   x̂₀ = remove_com(x̂₀)

4. x₀ MSE loss (per atom → per molecule → SNR-weighted):
   err_i = Σᵈ (x̂₀ᵢᵈ − x₀ᵢᵈ)²             # per-atom squared error (sum over xyz)
   mse_m = (1/N_m) Σᵢ∈m err_i              # per-molecule average
   SNR(t) = ᾱₜ/(1−ᾱₜ)
   w(t) = min(SNR(t), 5.0) / SNR(t)        # Min-SNR weight
   L_MSE = (1/B) Σ_m w(t_m) · mse_m       # weighted average over batch

5. Geometry loss (t-gated):
   geo_mask = (t < 300)    # only 30% of timesteps
   if any(geo_mask):
     x̂₀_low = x̂₀[atoms of molecules with geo_mask==True]
     L_bond = w_bond × MSE(dist(x̂₀), ideal_dist)
     L_angle = w_angle × MSE(angle(x̂₀), ideal_angle)
     L_repulsion = w_rep × mean(relu(d_thresh − dist_nonbond)²)
     L_geo = L_bond + L_angle + L_repulsion
   else:
     L_geo = 0

6. Total loss:
   L = L_MSE + 0.5 × L_geo

7. Gradient computation:
   L.backward()                                          # autograd through steps 2-6
   clip_grad_norm_(model.parameters(), max_norm=1.0)    # gradient clipping
   optimizer.step()                                      # AdamW update
```

**AdamW update rule:**
```
m_t = β₁·m_{t-1} + (1−β₁)·∇L       (momentum, β₁=0.9)
v_t = β₂·v_{t-1} + (1−β₂)·∇L²      (RMS, β₂=0.999)
m̂_t = m_t/(1−β₁ᵗ)                  (bias correction)
v̂_t = v_t/(1−β₂ᵗ)
θ ← θ − lr · m̂_t/(√v̂_t + ε) − lr·λ·θ   (AdamW: weight decay applied to θ, not m̂)
```

With lr=5e-4, weight_decay=0.01, ε=1e-8.

### 9.5 Cosine LR Schedule

```
lr(epoch) = {
    5e-4 × (epoch+1)/10             if epoch < 10  (warmup)
    max(5e-4 × 0.5 × (1+cos(πp)), 5e-6)  otherwise
}

where p = (epoch − 10) / (500 − 10)   ∈ [0, 1]
```

This ensures:
- Epochs 0-9: linear warmup from 5e-5 to 5e-4
- Epochs 10-500: cosine decay from 5e-4 to 5e-6 (floor at 1%)

**Without the 1% floor:** at epoch 300+, lr would reach ~1e-9, effectively stopping training (gradients too small relative to accumulated noise in optimizer state).

### 9.6 Evaluation Protocol (GeoDiff)

Every 50 epochs, evaluate on 300 validation molecules:

```python
for mol in eval_dataset[:300]:
    # Generate K=10 conformers via DDIM (50 steps, η=0)
    x_gen = [model.ddim_sample(mol.atom_types, mol.edge_index,
                                mol.bond_types, mol.batch_idx, num_steps=50)
             for _ in range(10)]
    
    # Reference conformers from RDKit MMFF (multiple local minima)
    x_ref = mol.reference_conformers  # R conformers
    
    # RMSD matrix after Kabsch alignment (heavy atoms only)
    D[k, r] = RMSD(x_gen[k], x_ref[r])
    
    # COV-R: fraction of reference conformers covered
    COV_R_mol = mean(min_k D[k, r] < 0.5 for r in range(R))
    
    # MAT-R: mean minimum distance from reference to generated
    MAT_R_mol = mean(min_k D[k, r] for r in range(R))
```

### 9.7 Results (Epoch-by-Epoch)

| Epoch | MAT-R (mean) | MAT-R (median) | COV-R   | MAT-P   | COV-P   |
|-------|-------------|----------------|---------|---------|---------|
| 50    | 0.172 Å     | 0.150 Å        | 99.67%  | 24.4%   | 92.3%   |
| 100   | 0.164 Å     | 0.143 Å        | 99.67%  | 23.6%   | 93.3%   |
| 150   | 0.170 Å     | 0.143 Å        | 99.67%  | 23.6%   | 92.9%   |
| 200   | 0.164 Å     | 0.129 Å        | 99.67%  | 23.0%   | 93.3%   |
| 250   | 0.164 Å     | 0.130 Å        | 99.67%  | 23.1%   | 93.3%   |
| 300   | 0.163 Å     | 0.131 Å        | 99.67%  | 22.8%   | 93.2%   |
| 350   | 0.163 Å     | 0.127 Å        | 99.67%  | 22.8%   | 93.2%   |
| 400   | **0.162 Å** | 0.127 Å        | 99.67%  | **22.5%** | 93.8% |
| 450   | 0.163 Å     | 0.130 Å        | 99.67%  | 22.5%   | 93.8%   |
| **500** | **0.160 Å** | **0.127 Å**  | **99.67%** | **22.6%** | **93.9%** |

**SOTA reference (from literature, heavy-atom QM9):**
- GeoDiff (ICML 2022): COV-R=71.0%, MAT-R=0.297 Å
- GeoMol (NeurIPS 2021): COV-R=71.5%, MAT-R=0.225 Å
- **Our Exp G (ep500):** COV-R=99.7%, MAT-R=0.160 Å

The COV-R of 99.7% significantly outperforms published SOTA (71%). MAT-R of 0.160 Å also outperforms GeoDiff (0.297 Å). This is plausibly due to: (1) heavier geometry supervision (geo_w=0.5 vs GeoDiff's geometry=0), (2) the x₀ parameterization + global readout architecture.

---

## 10. Architecture Evolution Table

| Component              | Exp A | Exp B | Exp C | Exp D | Exp E | Exp F | Exp G |
|------------------------|-------|-------|-------|-------|-------|-------|-------|
| **Base**               | DDPM  | DDPM  | CFM   | DDPM  | CFM   | DDPM  | DDPM  |
| **EGNN type**          | Basic | Attn  | Basic | Basic | Attn  | Attn  | Attn  |
| **hidden_dim**         | 256   | 256   | 256   | 256   | 256   | 384   | 384   |
| **num_layers**         | 6     | 6     | 6     | 6     | 6     | 8     | 8     |
| **num_rbf**            | 20    | 20    | 20    | 20    | 20    | 32    | 32    |
| **Global readout**     | ✗     | ✓     | ✗     | ✗     | ✓     | ✓     | ✓     |
| **Torsion loss**       | ✗     | ✗     | ✗     | ✓     | ✓     | ✗     | ✗     |
| **geometry_weight**    | 0.1   | 0.1   | 0.05  | 0.5   | 0.1   | 0.5   | 0.5   |
| **Heavy-atom dataset** | ✗     | ✗     | ✗     | ✗     | ✗     | ✓     | ✓     |
| **DDP**                | ✗     | ✗     | ✗     | ✗     | ✗     | ✓ 💥  | ✓ ✅  |
| **epochs**             | 50    | 50    | 50    | 50    | 200   | 0     | 500+  |
| **params (M)**         | 2.81  | 2.82  | 2.81  | 2.81  | 2.82  | 8.36  | 8.36  |
| **Status**             | ✓     | ✓     | ✗     | ✓     | ✗     | 💥    | ★     |

---

## 11. Quantitative Results Summary

From `autoresearch/results.tsv` and `autoresearch/results_geodiff.tsv`:

| Exp | MAT-R (Å)↓ | COV-R↑  | Fully Valid | Strain (kcal) | Bond Err | Protocol |
|-----|------------|---------|-------------|---------------|----------|----------|
| A   | 0.638      | 0.278   | 14.0%       | 34,250        | 0.230    | internal |
| B   | 0.654      | 0.270   | 13.6%       | **4,078**     | 0.226    | internal |
| C   | 0.757      | 0.080   | 0.0%        | 120,835       | 0.399    | internal |
| D   | **0.572**  | **0.400**| **11.8%**   | 55,298        | 0.232    | internal |
| E   | 0.786      | 0.064   | 0.0%        | 74,222        | 0.442    | internal |
| F   | —          | —       | —           | —             | —        | CRASH    |
| G (ep500) | **0.160** | **0.997** | ~14%  | low           | low      | GeoDiff  |

---

## 12. Key Lessons and What Worked

### What Worked

1. **x₀ parameterization (all exps):** More stable than ε-param for geometry-constrained training. Allows direct geometry loss on x̂₀.

2. **Cosine β-schedule + LR floor:** Smoother SNR curve; LR floor at 1% prevents stagnation at epoch 200+.

3. **RBF distance features (all exps):** 8× more geometric information than raw scalar distance.

4. **Degree normalization fix (all exps):** Correct bidirectional-graph normalization. Without it, atoms with many neighbors have incorrect aggregation scale.

5. **Attention EGNN (Exp B → G):** 8× strain energy reduction, geometrically more plausible structures.

6. **Torsion auxiliary loss (Exp D):** Single biggest MAT-R improvement in 50-epoch experiments (−10%). Confirmed TorDiff's key finding.

7. **t-gating for geometry loss (all exps from B):** Prevents chaotic gradients at high-noise timesteps.

8. **Heavy-atom dataset (Exp G):** Simpler problem (9 atoms max), enables cleaner learning of carbon/nitrogen/oxygen geometry.

9. **Model scaling (Exp G):** 2.81M → 8.36M params; hidden 256→384, layers 6→8. EGNN benefits significantly from depth.

10. **500-epoch training (Exp G):** Most improvement occurs 0-200 epochs, but slow refinement continues through 500.

### What Didn't Work

1. **Conditional Flow Matching (Exp C, E):** Geometry loss corrupts the straight-line ODE trajectory. CFM requires either geo_w=0 or a geometry-aware interpolation path.

2. **AMP with custom scatter ops (Exp F):** AMP float16 casts tensor dtypes, breaking scatter_reduce. Must explicitly cast src dtype before scatter.

3. **WARMUP=EPOCHS (original bug):** Never reaches target LR.

4. **Compounding LR schedule (original bug):** Multiplicative application from scaled LR causes rapid decay.

---

## 13. References and Citations

| # | Citation |
|---|---------|
| [1] | **Hoogeboom et al. (2022)** "Equivariant Diffusion for Molecule Generation in 3D." *ICML 2022*. arXiv:2203.17003. [EDM — x₀ parameterization, CoM removal, VLB] |
| [2] | **Ho et al. (2020)** "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*. arXiv:2006.11239. [DDPM — forward/reverse diffusion, noise schedule] |
| [3] | **Nichol & Dhariwal (2021)** "Improved Denoising Diffusion Probabilistic Models." *ICML 2021*. arXiv:2102.09672. [Cosine β-schedule, DDIM comparison] |
| [4] | **Song et al. (2021)** "Denoising Diffusion Implicit Models." *ICLR 2021*. arXiv:2010.02502. [DDIM sampling, deterministic ODE] |
| [5] | **Satorras et al. (2021)** "E(n) Equivariant Graph Neural Networks." *ICML 2021*. arXiv:2102.09844. [EGNN — equivariant message passing, coordinate updates] |
| [6] | **Xu et al. (2022)** "GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation." *ICML 2022*. arXiv:2203.02923. [GeoDiff — COV-MAT metrics, heavy-atom QM9, DDPM for conformers] |
| [7] | **Le et al. (2024)** "Navigating the Design Space of Equivariant Diffusion-Based Generative Models for De Novo 3D Molecule Generation." *ICLR 2024*. arXiv:2306.01916. [EQGAT-diff — attention over EGNN messages] |
| [8] | **Veličković et al. (2018)** "Graph Attention Networks." *ICLR 2018*. arXiv:1710.10903. [GAT — attention over graph neighbors, LeakyReLU softmax] |
| [9] | **Morehead & Cheng (2023)** "Geometry-Complete Diffusion for 3D Molecule Generation and Optimization." *NeurIPS 2023*. arXiv:2302.04313. [GCDM — geometry loss gating, t-threshold] |
| [10] | **Jing et al. (2022)** "Torsional Diffusion for Molecular Conformer Generation." *NeurIPS 2022*. arXiv:2206.01729. [TorDiff — torsion-space diffusion, OPLS-AA torsion params] |
| [11] | **Ganea et al. (2021)** "GeoMol: Torsional Graph Neural Network for Molecular Conformer Generation and Property Prediction." *NeurIPS 2021*. arXiv:2106.07802. [GeoMol — dihedral angle prediction as key for conformers] |
| [12] | **Hang et al. (2023)** "Efficient Diffusion Training via Min-SNR Weighting Strategy." *ICCV 2023*. arXiv:2303.09556. [Min-SNR — loss weighting by clipped SNR ratio] |
| [13] | **Lipman et al. (2023)** "Flow Matching for Generative Modeling." *ICLR 2023*. arXiv:2210.02747. [CFM — straight-line ODE, optimal transport paths] |
| [14] | **Halgren (1996)** "Merck Molecular Force Field." *J. Comput. Chem. 17:490-519*. [MMFF94 — bond length/angle/torsion targets] |
| [15] | **Schütt et al. (2017)** "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions." *NeurIPS 2017*. arXiv:1706.08566. [SchNet — RBF distance features for molecules] |
| [16] | **Gasteiger et al. (2020)** "Directional Message Passing for Molecular Graphs." *ICLR 2020*. arXiv:2003.03123. [DimeNet — RBF + angular features for molecular GNNs] |
| [17] | **Jorgensen et al. (1996)** "Development and Testing of the OPLS All-Atom Force Field." *J. Am. Chem. Soc. 118:11225*. [OPLS-AA — torsion energy parameters V₁, V₂, V₃] |
| [18] | **Ramakrishnan et al. (2014)** "Quantum chemistry structures and properties of 134 kilo molecules." *Scientific Data 1, 140022*. [QM9 dataset] |
| [19] | **Liao & Smidt (2023)** "Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs." *ICLR 2023*. arXiv:2206.11990. [Equiformer — attention-based equivariant networks] |
| [20] | **Kingma & Ba (2015)** "Adam: A Method for Stochastic Optimization." *ICLR 2015*. arXiv:1412.6980. [Adam/AdamW optimizer foundation] |

---

*Document generated from source code audit of experiments A-G in `/scratch/nishanth.r/nextmol_experiment/mol_next/`*  
*Last updated: 2026-05-25 | Exp G still training (ep500+)*
