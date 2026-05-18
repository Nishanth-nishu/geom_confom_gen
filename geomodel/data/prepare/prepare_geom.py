#!/usr/bin/env python3
"""
prepare_geom.py — Convert GeoDiff GEOM-Drugs PKL to geo-model format.

GeoDiff GEOM-Drugs PKL contains molecules with multiple conformers.
Each item is a PackedConformer-style object with:
    atom_type      : (N,)    long — atomic numbers
    edge_index     : (2, E)  long — bond connectivity
    edge_type      : (E,)    long — bond orders
    pos_ref        : (M*N, 3) float — M stacked conformers

Output:
    A single geom_drugs_processed.pkl with list of dicts:
    {
        "atom_types"  : List[int],
        "edge_index"  : List[List[int]],
        "bond_types"  : List[int],
        "conformers"  : List[List[List[float]]]  # (M, N, 3)
    }

Usage:
    CUDA_VISIBLE_DEVICES="" python3 prepare_geom.py
"""

import os
import pickle
import sys

GEODIFF_DATA = "/scratch/nishanth.r/nextmol_experiment/GeoDiff/data/GEOM/Drugs"
OUT_DIR      = "/scratch/nishanth.r/nextmol_experiment/geo-model/data/geom_drugs"
VALID_Z      = {6, 7, 8, 9, 16, 17, 35, 53}  # C, N, O, F, S, Cl, Br, I

PKL_FILES = {
    "train": "train_data_40k.pkl",
    "test":  "test_data_1k.pkl",
    "val":   "val_data_5k.pkl",
}


def data_to_dict(data) -> dict | None:
    """Convert a GeoDiff Data object to geomodel GEOM-Drugs dict."""
    try:
        atom_type  = data.atom_type.tolist()
        edge_index = data.edge_index.tolist()
        edge_type  = data.edge_type.tolist()
        pos_ref    = data.pos_ref  # (M*N, 3) tensor
        N = len(atom_type)
        M = pos_ref.shape[0] // N
    except Exception:
        return None

    if any(z not in VALID_Z for z in atom_type):
        return None
    if M < 1 or N < 3:
        return None

    conformers = []
    for m in range(M):
        conf = pos_ref[m * N:(m + 1) * N].tolist()
        conformers.append([[round(c, 6) for c in xyz] for xyz in conf])

    return {
        "atom_types":  atom_type,
        "edge_index":  edge_index,
        "bond_types":  edge_type,
        "conformers":  conformers,
    }


def convert_split(split_name: str, pkl_filename: str, out_dir: str):
    pkl_path = os.path.join(GEODIFF_DATA, pkl_filename)
    out_path = os.path.join(out_dir, f"geom_drugs_{split_name}.pkl")

    print(f"Loading {pkl_filename} ...", flush=True)
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"  Loaded {len(dataset)} molecules.", flush=True)

    results = []
    n_skip = 0
    for i, item in enumerate(dataset):
        d = data_to_dict(item)
        if d is None:
            n_skip += 1
            continue
        results.append(d)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(dataset)} done ...", flush=True)

    del dataset

    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"  Written {len(results)} molecules → {out_path}  (skipped {n_skip})", flush=True)
    return len(results)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for split, pkl_file in PKL_FILES.items():
        n = convert_split(split, pkl_file, OUT_DIR)
        print(f"  [{split}] {n:,} molecules converted.\n", flush=True)

    print("Done! GEOM-Drugs PKL files ready in:", OUT_DIR)


if __name__ == "__main__":
    main()
