#!/usr/bin/env python3
"""
convert_geodiff_pkl.py — Convert GeoDiff QM9 PKL files to geo-model JSONL format.

GeoDiff PKL files contain torch_geometric.data.Data objects with:
    atom_type  : (N,)    long — atomic numbers
    pos        : (N, 3)  float — 3D coordinates
    edge_index : (2, E)  long — bond connectivity
    edge_type  : (E,)    long — bond orders (1=single, 2=double, ...)

This script converts all three splits (train 40k, val 5k, test 1k) into
a single qm9_heavy.jsonl with the geo-model molecule dict format.

Usage:
    CUDA_VISIBLE_DEVICES="" python3 convert_geodiff_pkl.py
"""

import json
import os
import pickle
import sys

import torch

GEODIFF_DATA = "/scratch/nishanth.r/nextmol_experiment/GeoDiff/data/GEOM/QM9"
OUT_DIR      = "/scratch/nishanth.r/nextmol_experiment/geo-model/data/qm9"
OUT_FILE     = os.path.join(OUT_DIR, "qm9_heavy.jsonl")

# Heavy atom types only (no H=1)
VALID_Z = {6, 7, 8, 9, 16, 17}

PKL_FILES = [
    "train_data_40k.pkl",
    "val_data_5k.pkl",
    "test_data_1k.pkl",
]


def data_to_dict(data) -> dict | None:
    """Convert a PyG Data object to geomodel dict. Returns None to skip."""
    try:
        # Access fields — PyG Data acts like a dict
        atom_type  = data.atom_type.tolist()   if hasattr(data, 'atom_type')  else data['atom_type'].tolist()
        pos        = data.pos.tolist()          if hasattr(data, 'pos')        else data['pos'].tolist()
        edge_index = data.edge_index.tolist()   if hasattr(data, 'edge_index') else data['edge_index'].tolist()
        edge_type  = data.edge_type.tolist()    if hasattr(data, 'edge_type')  else data['edge_type'].tolist()
    except Exception as e:
        return None

    # Filter: heavy atoms only (drop H=1)
    heavy_mask = [z != 1 for z in atom_type]
    if sum(heavy_mask) < 2:
        return None

    # Remap indices
    old2new = {}
    new_atom_types, new_coords = [], []
    for old_i, (keep, z, xyz) in enumerate(zip(heavy_mask, atom_type, pos)):
        if keep:
            if z not in VALID_Z:
                return None  # skip molecule with unsupported atoms
            old2new[old_i] = len(new_atom_types)
            new_atom_types.append(z)
            new_coords.append([round(c, 6) for c in xyz])

    # Remap edges — keep only heavy-heavy bonds
    src_list, dst_list, bo_list = [], [], []
    n_edges = len(edge_index[0]) if isinstance(edge_index[0], list) else edge_index[0]
    for k in range(len(edge_index[0])):
        i = edge_index[0][k]
        j = edge_index[1][k]
        if i in old2new and j in old2new:
            src_list.append(old2new[i])
            dst_list.append(old2new[j])
            bo_list.append(edge_type[k])

    return {
        "atom_types":  new_atom_types,
        "edge_index":  [src_list, dst_list],
        "bond_types":  bo_list,
        "coordinates": new_coords,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_total = n_ok = n_skip = 0

    print(f"Output: {OUT_FILE}")

    with open(OUT_FILE, "w") as f_out:
        for pkl_name in PKL_FILES:
            pkl_path = os.path.join(GEODIFF_DATA, pkl_name)
            print(f"Loading {pkl_name} ...", flush=True)

            with open(pkl_path, "rb") as f:
                dataset = pickle.load(f)

            print(f"  Loaded {len(dataset)} molecules. Converting...", flush=True)

            for i, item in enumerate(dataset):
                n_total += 1
                d = data_to_dict(item)
                if d is None:
                    n_skip += 1
                    continue
                f_out.write(json.dumps(d) + "\n")
                n_ok += 1

                if (i + 1) % 5000 == 0:
                    print(f"  [{pkl_name}] {i+1}/{len(dataset)} done ...", flush=True)

            # Free memory immediately
            del dataset
            print(f"  Done: {n_ok} written so far.", flush=True)

    print(f"\nFinished!")
    print(f"  Total molecules processed : {n_total:,}")
    print(f"  Written to JSONL           : {n_ok:,}")
    print(f"  Skipped (invalid)          : {n_skip:,}")
    print(f"  File: {OUT_FILE}")
    sz = os.path.getsize(OUT_FILE) / 1024 / 1024
    print(f"  Size: {sz:.1f} MB")


if __name__ == "__main__":
    main()
