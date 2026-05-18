#!/usr/bin/env python3
"""
prepare_qm9.py — Download and preprocess QM9 into the geomodel JSONL format.

Reference:
    Ramakrishnan et al., "Quantum chemistry structures and properties of 134
    kilo molecules." Scientific Data, 2014.
    Dataset: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904

Output (per molecule, one JSON per line):
    {
        "atom_types":  [int, ...],        # atomic numbers, heavy atoms only
        "edge_index":  [[src...],[dst...]], # undirected bonds
        "bond_types":  [int, ...],         # 1=single, 2=double, 3=triple, 4=aromatic
        "coordinates": [[x,y,z], ...],     # DFT-optimized 3D positions (Å)
    }

Usage:
    python geomodel/data/prepare/prepare_qm9.py --out-dir data/qm9
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request

# Try RDKit import
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    print("ERROR: RDKit not found. Activate the GeoDiff venv first:")
    print("  source /scratch/nishanth.r/nextmol_experiment/GeoDiff/venv/bin/activate")
    sys.exit(1)


# QM9 raw SDF download URL (official figshare mirror)
QM9_SDF_URL = "https://figshare.com/ndownloader/files/3195389"
QM9_SDF_FILENAME = "gdb9.sdf"

BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE:   1,
    Chem.rdchem.BondType.DOUBLE:   2,
    Chem.rdchem.BondType.TRIPLE:   3,
    Chem.rdchem.BondType.AROMATIC: 4,
}

VALID_ATOMIC_NUMS = {6, 7, 8, 9, 16, 17}   # C, N, O, F, S, Cl


def download_qm9(raw_dir: str) -> str:
    """Download QM9 SDF file if not already present."""
    os.makedirs(raw_dir, exist_ok=True)
    sdf_path = os.path.join(raw_dir, QM9_SDF_FILENAME)

    if os.path.exists(sdf_path) and os.path.getsize(sdf_path) > 10_000_000:
        print(f"  ✅ Raw SDF already exists: {sdf_path}")
        return sdf_path

    print(f"Downloading QM9 SDF from figshare (~80MB)...")
    print(f"  URL: {QM9_SDF_URL}")

    try:
        urllib.request.urlretrieve(QM9_SDF_URL, sdf_path)
        print(f"  ✅ Downloaded: {sdf_path} ({os.path.getsize(sdf_path)//1024//1024}MB)")
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        print("  Try manually: wget -O data/qm9/raw/gdb9.sdf 'https://figshare.com/ndownloader/files/3195389'")
        sys.exit(1)

    return sdf_path


def mol_to_dict(mol: Chem.Mol) -> dict:
    """
    Convert an RDKit Mol to the geomodel dict format.

    Heavy-atom-only: hydrogen atoms are stripped before processing.
    Only bonds between remaining heavy atoms are kept.
    """
    # Remove explicit H atoms
    mol = Chem.RemoveHs(mol)

    conf = mol.GetConformer()
    atom_types, coords = [], []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        pos = conf.GetAtomPosition(atom.GetIdx())
        atom_types.append(z)
        coords.append([round(pos.x, 6), round(pos.y, 6), round(pos.z, 6)])

    src_list, dst_list, bond_list = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bo = BOND_TYPE_MAP.get(bond.GetBondType(), 1)
        # Add both directions (undirected graph)
        src_list += [i, j]
        dst_list += [j, i]
        bond_list += [bo, bo]

    return {
        "atom_types":  atom_types,
        "edge_index":  [src_list, dst_list],
        "bond_types":  bond_list,
        "coordinates": coords,
    }


def process_qm9(sdf_path: str, out_dir: str, max_atoms: int = 9) -> int:
    """Parse QM9 SDF and write heavy-atom JSONL."""
    out_path = os.path.join(out_dir, "qm9_heavy.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)

    n_ok = n_skip = n_err = 0
    with open(out_path, "w") as f_out:
        for i, mol in enumerate(suppl):
            if mol is None:
                n_err += 1
                continue

            try:
                mol_noH = Chem.RemoveHs(mol)
            except Exception:
                n_err += 1
                continue

            # Filter by atom count and valid atom types
            heavy_atoms = [a.GetAtomicNum() for a in mol_noH.GetAtoms()]
            if len(heavy_atoms) > max_atoms:
                n_skip += 1
                continue
            if any(z not in VALID_ATOMIC_NUMS for z in heavy_atoms):
                n_skip += 1
                continue
            if not mol_noH.GetNumConformers():
                n_skip += 1
                continue

            try:
                d = mol_to_dict(mol)
            except Exception:
                n_err += 1
                continue

            f_out.write(json.dumps(d) + "\n")
            n_ok += 1

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1:,} molecules... ({n_ok:,} kept)")

    print(f"\n  Total processed: {n_ok:,} kept, {n_skip:,} skipped, {n_err:,} errors")
    print(f"  Output: {out_path}")
    return n_ok


def main():
    parser = argparse.ArgumentParser(description="Prepare QM9 dataset for geomodel.")
    parser.add_argument("--raw-dir", default="data/qm9/raw", help="Directory for raw SDF download.")
    parser.add_argument("--out-dir", default="data/qm9",     help="Output directory for JSONL.")
    parser.add_argument("--max-atoms", type=int, default=9,  help="Max heavy atoms per molecule.")
    parser.add_argument("--no-download", action="store_true", help="Skip download (SDF already in --raw-dir).")
    args = parser.parse_args()

    print("=" * 60)
    print("  geo-model QM9 Data Preparation")
    print("=" * 60)

    if not args.no_download:
        sdf_path = download_qm9(args.raw_dir)
    else:
        sdf_path = os.path.join(args.raw_dir, QM9_SDF_FILENAME)
        if not os.path.exists(sdf_path):
            print(f"ERROR: SDF not found at {sdf_path}")
            sys.exit(1)

    print(f"\nProcessing QM9 (max_atoms={args.max_atoms})...")
    n = process_qm9(sdf_path, args.out_dir, args.max_atoms)
    print(f"\n✅ Done. {n:,} molecules written to {args.out_dir}/qm9_heavy.jsonl")


if __name__ == "__main__":
    main()
