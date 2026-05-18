"""
datasets.py — QM9 and GEOM-Drugs dataset classes.

Both datasets return a standardized dict per molecule:
    {
        "atom_types"  : LongTensor (N,)     — atomic numbers
        "edge_index"  : LongTensor (2, E)   — undirected bond pairs
        "bond_types"  : LongTensor (E,)     — bond orders (1–4)
        "coordinates" : FloatTensor (N, 3)  — ground-truth 3D positions (Å)
        "num_atoms"   : int                 — N (for inspection)
    }

The collate_fn in collate.py assembles these into batched tensors.
"""

import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from geomodel.models.noise_schedule import remove_com


class QM9Dataset(Dataset):
    """
    QM9 heavy-atom conformer dataset.

    Each molecule has exactly 1 DFT-optimized 3D geometry (B3LYP/6-31G*).
    Molecules are filtered to ≤ max_atoms heavy atoms and atom types
    in a standard vocabulary (C, N, O, F, S, Cl).

    The 90/10 train/val split uses a fixed random permutation (seed=42)
    for reproducibility across all experiments.

    Args:
        root     : Path to directory containing the processed JSONL file.
        filename : Name of the JSONL file (one molecule per line).
        split    : 'train' or 'val'.
        seed     : RNG seed for reproducible split (default 42).
        max_atoms: Discard molecules with more than this many heavy atoms.
    """

    VALID_ATOMIC_NUMS = {6, 7, 8, 9, 16, 17}   # C, N, O, F, S, Cl

    def __init__(
        self,
        root: str,
        filename: str = "qm9_heavy.jsonl",
        split: str = "train",
        seed: int = 42,
        max_atoms: int = 9,
    ) -> None:
        super().__init__()
        assert split in ("train", "val"), f"split must be 'train' or 'val', got '{split}'"

        self.split = split

        # Load and filter all molecules
        all_mols = self._load_jsonl(os.path.join(root, filename), max_atoms)

        # Reproducible 90/10 split
        N = len(all_mols)
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=generator).tolist()
        n_train = int(N * 0.9)

        indices = perm[:n_train] if split == "train" else perm[n_train:]
        self.mols: List[Dict] = [all_mols[i] for i in indices]

    def _load_jsonl(self, path: str, max_atoms: int) -> List[Dict]:
        """Load and filter molecules from JSONL file."""
        mols = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                if not item.get("coordinates"):
                    continue
                z = item["atom_types"]
                if len(z) > max_atoms:
                    continue
                if any(zi not in self.VALID_ATOMIC_NUMS for zi in z):
                    continue
                mols.append(item)
        return mols

    def __len__(self) -> int:
        return len(self.mols)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mol = self.mols[idx]
        coords = torch.tensor(mol["coordinates"], dtype=torch.float32)

        # Remove CoM from ground-truth coordinates
        coords = coords - coords.mean(0)

        return {
            "atom_types":  torch.tensor(mol["atom_types"], dtype=torch.long),
            "edge_index":  torch.tensor(mol["edge_index"], dtype=torch.long),
            "bond_types":  torch.tensor(mol["bond_types"], dtype=torch.long),
            "coordinates": coords,
            "num_atoms":   len(mol["atom_types"]),
        }


class GeomDrugsDataset(Dataset):
    """
    GEOM-Drugs conformer dataset.

    Each molecule has multiple DFT-optimized conformers (Boltzmann-weighted).
    During training, one conformer is randomly sampled per molecule per batch.
    During evaluation, all conformers serve as reference structures.

    The processed pickle contains a list of dicts, each with:
        - "atom_types"  : List[int]        — atomic numbers
        - "edge_index"  : List[List[int]]  — [[src...], [dst...]]
        - "bond_types"  : List[int]        — bond orders
        - "conformers"  : List[List[List[float]]] — (M, N, 3) conformer stack

    Args:
        root        : Path to directory containing the processed pickle.
        filename    : Name of the processed pickle file.
        split       : 'train' or 'test'.
        mode        : 'train' (sample 1 conformer) or 'eval' (return all conformers).
        max_conformers: Maximum number of conformers to keep per molecule.
    """

    def __init__(
        self,
        root: str,
        filename: str = "geom_drugs_processed.pkl",
        split: str = "train",
        mode: str = "train",
        max_conformers: int = 50,
    ) -> None:
        super().__init__()
        assert split in ("train", "test")
        assert mode in ("train", "eval")

        self.mode = mode
        self.max_conformers = max_conformers

        with open(os.path.join(root, filename), "rb") as f:
            self.mols: List[Dict] = pickle.load(f)

    def __len__(self) -> int:
        return len(self.mols)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mol = self.mols[idx]
        conformers = mol["conformers"][:self.max_conformers]   # (M, N, 3)

        if self.mode == "train":
            # Randomly sample one conformer for training
            i = np.random.randint(len(conformers))
            coords = torch.tensor(conformers[i], dtype=torch.float32)
            coords = coords - coords.mean(0)

            return {
                "atom_types":  torch.tensor(mol["atom_types"], dtype=torch.long),
                "edge_index":  torch.tensor(mol["edge_index"], dtype=torch.long),
                "bond_types":  torch.tensor(mol["bond_types"], dtype=torch.long),
                "coordinates": coords,
                "num_atoms":   len(mol["atom_types"]),
            }
        else:
            # Eval mode: return all reference conformers (for COV/MAT metrics)
            all_coords = []
            for conf in conformers:
                c = torch.tensor(conf, dtype=torch.float32)
                c = c - c.mean(0)
                all_coords.append(c)

            return {
                "atom_types":  torch.tensor(mol["atom_types"], dtype=torch.long),
                "edge_index":  torch.tensor(mol["edge_index"], dtype=torch.long),
                "bond_types":  torch.tensor(mol["bond_types"], dtype=torch.long),
                "all_conformers": all_coords,   # List of (N, 3) tensors
                "num_atoms":   len(mol["atom_types"]),
            }
