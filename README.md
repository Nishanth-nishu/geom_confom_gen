# geo-model

**E(3)-Equivariant Diffusion for 3D Molecular Conformer Generation**

A clean, research-grade implementation of an E(3)-equivariant diffusion model for generating chemically accurate 3D molecular conformers. Architecture aligned with [EDM (Hoogeboom et al., ICML 2022)](https://arxiv.org/abs/2203.17003) and evaluated on the GeoDiff/GeoMol benchmark protocol.

---

## Results (QM9, δ = 0.5 Å)

| Model | COV-R ↑ | MAT-R ↓ | Params |
|-------|---------|---------|--------|
| GeoDiff | 71.0% | 0.297 Å | ~6M |
| GeoMol | 71.5% | 0.225 Å | ~5M |
| TorDiff | 73.2% | 0.219 Å | ~8M |
| **geo-model** | **TBD** | **TBD** | **4.3M** |

---

## Architecture

- **Backbone**: E(3)-equivariant GNN ([EGNN, Satorras et al., ICML 2021](https://arxiv.org/abs/2102.09844))
- **Dimensions**: `hidden=256`, `layers=9`, `rbf=50`, `d_max=10Å` (EDM-aligned)
- **Parameterization**: Direct x₀ prediction (not ε-prediction) for stable geometry gradients
- **Noise schedule**: Cosine schedule ([Nichol & Dhariwal, ICML 2021](https://arxiv.org/abs/2102.09672))
- **Loss weighting**: Min-SNR-γ=5 ([Hang et al., ICCV 2023](https://arxiv.org/abs/2303.09556))
- **Geometry loss**: Bond length + angle constraints, gated to low-noise timesteps ([GCDM, Morehead 2023](https://arxiv.org/abs/2302.04313))
- **Sampling**: Deterministic DDIM ([Song et al., ICLR 2021](https://arxiv.org/abs/2010.02502))

---

## Project Structure

```
geo-model/
├── configs/               # YAML hyperparameter configs (no hardcoding)
│   ├── model/egnn_256.yaml
│   ├── training/qm9.yaml
│   └── training/geom_drugs.yaml
├── geomodel/              # pip-installable Python package
│   ├── models/            # noise_schedule, rbf, egnn, denoiser, diffusion
│   ├── training/          # geometry losses, DDP trainer
│   ├── data/              # dataset classes, collate_fn, prepare scripts
│   └── evaluation/        # Kabsch RMSD, COV/MAT metrics, reporter
├── scripts/               # Entry points: train.py, train_ddp.py, evaluate.py
├── slurm/                 # SLURM batch scripts for the cluster
└── tests/                 # 16 unit tests (equivariance, diffusion, metrics)
```

---

## Quick Start

### 1. Install

```bash
source /path/to/venv/bin/activate
cd geo-model
pip install -e .
```

### 2. Prepare Data

```bash
# QM9 — convert from GeoDiff PKL (fast, no download needed)
CUDA_VISIBLE_DEVICES="" python3 geomodel/data/prepare/convert_geodiff_pkl.py

# GEOM-Drugs
CUDA_VISIBLE_DEVICES="" python3 geomodel/data/prepare/prepare_geom.py
```

### 3. Train (Single GPU)

```bash
python3 scripts/train.py --config configs/training/qm9.yaml
```

### 4. Train (Multi-GPU DDP)

```bash
torchrun --nproc_per_node=2 scripts/train_ddp.py --config configs/training/qm9.yaml
```

### 5. Evaluate

```bash
python3 scripts/evaluate.py \
    --checkpoint checkpoints/qm9/best.pt \
    --config configs/training/qm9.yaml \
    --n-mols 200 --n-gen 10 --threshold 0.5
```

### 6. Run Tests

```bash
python3 -m pytest tests/ -v
# Expected: 16 passed
```

---

## Key References

1. **EDM**: Hoogeboom et al. "Equivariant Diffusion for Molecule Generation in 3D." ICML 2022.
2. **EGNN**: Satorras, Hoogeboom & Welling. "E(n) Equivariant Graph Neural Networks." ICML 2021.
3. **GeoDiff**: Xu et al. "GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation." ICML 2022.
4. **GCDM**: Morehead & Cheng. "Geometry-Complete Diffusion for 3D Molecule Generation." NeurIPS 2023.
5. **Min-SNR**: Hang et al. "Efficient Diffusion Training via Min-SNR Weighting Strategy." ICCV 2023.
6. **DDIM**: Song et al. "Denoising Diffusion Implicit Models." ICLR 2021.
7. **Cosine schedule**: Nichol & Dhariwal. "Improved Denoising Diffusion Probabilistic Models." ICML 2021.
