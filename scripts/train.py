#!/usr/bin/env python3
"""
train.py — Single-GPU training entry point.

Usage:
    python scripts/train.py --config configs/training/qm9.yaml
    python scripts/train.py --config configs/training/qm9.yaml --resume checkpoints/qm9/latest.pt
    python scripts/train.py --config configs/training/geom_drugs.yaml --wandb
"""

import argparse
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

# Allow running from repo root without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geomodel.data.collate import collate_fn
from geomodel.data.datasets import GeomDrugsDataset, QM9Dataset
from geomodel.models.diffusion import ConformerDiffusion
from geomodel.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConformerDiffusion (single GPU)")
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Load nested model config
    model_cfg_path = cfg.get("model_config")
    if model_cfg_path and os.path.exists(model_cfg_path):
        with open(model_cfg_path, "r") as f:
            cfg["model"] = yaml.safe_load(f)
    return cfg


def build_datasets(cfg: dict):
    data_cfg = cfg["data"]
    name = data_cfg["name"]

    if name == "qm9":
        train_ds = QM9Dataset(
            root=data_cfg["root"],
            filename=data_cfg.get("processed_file", "qm9_heavy.jsonl"),
            split="train",
            seed=data_cfg.get("seed", 42),
            max_atoms=data_cfg.get("max_atoms", 9),
        )
        val_ds = QM9Dataset(
            root=data_cfg["root"],
            filename=data_cfg.get("processed_file", "qm9_heavy.jsonl"),
            split="val",
            seed=data_cfg.get("seed", 42),
            max_atoms=data_cfg.get("max_atoms", 9),
        )
    elif name == "geom_drugs":
        train_ds = GeomDrugsDataset(
            root=data_cfg["root"],
            filename=data_cfg.get("processed_file", "geom_drugs_processed.pkl"),
            split="train",
            mode="train",
        )
        val_ds = GeomDrugsDataset(
            root=data_cfg["root"],
            filename=data_cfg.get("processed_file", "geom_drugs_processed.pkl"),
            split="test",
            mode="eval",
        )
    else:
        raise ValueError(f"Unknown dataset: '{name}'")

    return train_ds, val_ds


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("logging", {})["use_wandb"] = True

    device = torch.device(args.device)
    logging.info(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds, val_ds = build_datasets(cfg)
    logging.info(f"Train: {len(train_ds)} molecules | Val: {len(val_ds)} molecules")

    data_cfg = cfg["data"]
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.get("batch_size", 256),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 8),
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConformerDiffusion(
        model_cfg=cfg.get("model", {}),
        num_timesteps=cfg["diffusion"].get("num_timesteps", 1000),
        noise_schedule=cfg["diffusion"].get("noise_schedule", "cosine"),
        s=cfg["diffusion"].get("s", 0.008),
    )
    logging.info(f"Model parameters: {model.num_parameters:,}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
