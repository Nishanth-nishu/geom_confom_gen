#!/usr/bin/env python3
"""
evaluate.py — Full evaluation script (COV-R/P, MAT-R/P, Diversity).

Usage:
    # QM9 evaluation (200 molecules, 10 conformers each):
    python scripts/evaluate.py \
        --checkpoint checkpoints/qm9/best.pt \
        --config configs/training/qm9.yaml \
        --n-mols 200 --n-gen 10 --threshold 0.5

    # GEOM-Drugs evaluation:
    python scripts/evaluate.py \
        --checkpoint checkpoints/geom_drugs/best.pt \
        --config configs/training/geom_drugs.yaml \
        --n-mols 200 --n-gen 10 --threshold 1.25
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geomodel.data.collate import collate_fn
from geomodel.data.datasets import GeomDrugsDataset, QM9Dataset
from geomodel.evaluation.metrics import evaluate_metrics
from geomodel.evaluation.reporter import print_results_table
from geomodel.models.diffusion import ConformerDiffusion
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ConformerDiffusion")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--config",     required=True, help="Path to training YAML config.")
    parser.add_argument("--n-mols",  type=int, default=200, help="Number of val molecules.")
    parser.add_argument("--n-gen",   type=int, default=10,  help="Generated conformers per mol.")
    parser.add_argument("--threshold", type=float, default=0.5, help="COV threshold (Å).")
    parser.add_argument("--num-steps", type=int, default=100, help="DDIM steps.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg_path = cfg.get("model_config")
    if model_cfg_path and os.path.exists(model_cfg_path):
        with open(model_cfg_path) as f:
            cfg["model"] = yaml.safe_load(f)

    device = torch.device(args.device)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    state = torch.load(args.checkpoint, map_location=device)
    model = ConformerDiffusion(
        model_cfg=cfg.get("model", {}),
        num_timesteps=cfg["diffusion"].get("num_timesteps", 1000),
    )
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    logging.info(f"Loaded checkpoint from {args.checkpoint} (epoch {state.get('epoch', '?')})")

    # ── Dataset ────────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    if data_cfg["name"] == "qm9":
        val_ds = QM9Dataset(root=data_cfg["root"],
                            filename=data_cfg.get("processed_file", "qm9_heavy.jsonl"),
                            split="val")
    else:
        val_ds = GeomDrugsDataset(root=data_cfg["root"], split="test", mode="eval")

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ── Generate & evaluate ────────────────────────────────────────────────────
    all_refs, all_gens = [], []
    count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", total=args.n_mols):
            if count >= args.n_mols:
                break

            atom_types = batch["atom_types"].to(device)
            edge_index  = batch["edge_index"].to(device)
            bond_types  = batch["bond_types"].to(device)
            batch_idx   = batch["batch_idx"].to(device)
            x0          = batch["coordinates"].to(device)

            # Reference conformer(s)
            refs = [x0.cpu().numpy()]

            # Generate n_gen conformers from noise
            gens = [
                model.sample(atom_types, edge_index, bond_types, batch_idx,
                             num_steps=args.num_steps).cpu().numpy()
                for _ in range(args.n_gen)
            ]

            all_refs.append(refs)
            all_gens.append(gens)
            count += 1

    results = evaluate_metrics(all_refs, all_gens, threshold=args.threshold)

    print_results_table(
        results,
        label="geo-model",
        threshold=args.threshold,
        dataset=data_cfg["name"].upper(),
        n_mols=count,
        n_gen=args.n_gen,
    )


if __name__ == "__main__":
    main()
