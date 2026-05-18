"""
collate.py — Custom collate function for variable-size molecular graphs.

PyTorch's default collate_fn cannot handle variable-length tensors (different
number of atoms, bonds per molecule). This custom collate:
    1. Concatenates all per-atom tensors along dim=0.
    2. Offsets edge indices so they point into the global atom tensor.
    3. Creates a batch_idx tensor mapping each atom to its molecule.
"""

from typing import Dict, List

import torch


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of molecule dicts into a single batched dict.

    Input (list of dicts from Dataset.__getitem__):
        Each dict has: atom_types (N,), edge_index (2,E), bond_types (E,),
                       coordinates (N,3), num_atoms (int).

    Output (batched dict):
        atom_types  : (N_total,)    — all atoms concatenated
        edge_index  : (2, E_total)  — edges with globally-offset atom indices
        bond_types  : (E_total,)    — all bond types concatenated
        coordinates : (N_total, 3)  — all coordinates concatenated
        batch_idx   : (N_total,)    — molecule index per atom (0-indexed)
        num_atoms   : (B,)          — atom count per molecule
    """
    atom_types_list  = []
    edge_index_list  = []
    bond_types_list  = []
    coordinates_list = []
    batch_idx_list   = []

    atom_offset = 0

    for mol_idx, mol in enumerate(batch):
        N = mol["num_atoms"]

        atom_types_list.append(mol["atom_types"])
        coordinates_list.append(mol["coordinates"])

        # Offset edge indices by current atom count
        edge_index_list.append(mol["edge_index"] + atom_offset)
        bond_types_list.append(mol["bond_types"])

        # Assign molecule index to each atom
        batch_idx_list.append(torch.full((N,), mol_idx, dtype=torch.long))

        atom_offset += N

    return {
        "atom_types":  torch.cat(atom_types_list,  dim=0),
        "edge_index":  torch.cat(edge_index_list,  dim=1),
        "bond_types":  torch.cat(bond_types_list,  dim=0),
        "coordinates": torch.cat(coordinates_list, dim=0),
        "batch_idx":   torch.cat(batch_idx_list,   dim=0),
        "num_atoms":   torch.tensor([m["num_atoms"] for m in batch], dtype=torch.long),
    }
