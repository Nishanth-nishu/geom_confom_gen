"""
geo-model: E(3)-Equivariant Diffusion for 3D Molecular Conformer Generation.

Installable package. Run: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="geomodel",
    version="0.1.0",
    description="E(3)-equivariant diffusion model for molecular conformer generation",
    author="Nishanth R",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "rdkit>=2022.03.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "wandb>=0.15.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
)
