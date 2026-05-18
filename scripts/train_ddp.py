#!/usr/bin/env python3
"""
train_ddp.py — Multi-GPU DDP training entry point.

Usage (torchrun, 2 GPUs):
    torchrun --nproc_per_node=2 scripts/train_ddp.py \
        --config configs/training/qm9.yaml

Usage (legacy torch.distributed.launch):
    python -m torch.distributed.launch --nproc_per_node=2 scripts/train_ddp.py \
        --config configs/training/qm9.yaml
"""

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geomodel.data.collate import collate_fn
from geomodel.data.datasets import GeomDrugsDataset, QM9Dataset
from geomodel.models.diffusion import ConformerDiffusion
from geomodel.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP training for ConformerDiffusion")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--wandb", action="store_true")
    # DDP args (set automatically by torchrun)
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg_path = cfg.get("model_config")
    if model_cfg_path and os.path.exists(model_cfg_path):
        with open(model_cfg_path, "r") as f:
            cfg["model"] = yaml.safe_load(f)
    return cfg


def main():
    args = parse_args()

    # ── Initialize process group ───────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = args.local_rank
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [rank0] %(message)s",
            datefmt="%H:%M:%S",
        )

    cfg = load_config(args.config)
    if args.wandb and rank == 0:
        cfg.setdefault("logging", {})["use_wandb"] = True

    # ── Datasets ──────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    name = data_cfg["name"]

    if name == "qm9":
        train_ds = QM9Dataset(root=data_cfg["root"],
                              filename=data_cfg.get("processed_file", "qm9_heavy.jsonl"),
                              split="train")
        val_ds   = QM9Dataset(root=data_cfg["root"],
                              filename=data_cfg.get("processed_file", "qm9_heavy.jsonl"),
                              split="val")
    elif name == "geom_drugs":
        train_ds = GeomDrugsDataset(root=data_cfg["root"], split="train", mode="train")
        val_ds   = GeomDrugsDataset(root=data_cfg["root"], split="test",  mode="eval")
    else:
        raise ValueError(f"Unknown dataset: '{name}'")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader  = DataLoader(
        train_ds,
        batch_size=data_cfg.get("batch_size", 128) // world_size,
        sampler=train_sampler,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    # Val runs on rank 0 only
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=collate_fn,
    ) if rank == 0 else None

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConformerDiffusion(
        model_cfg=cfg.get("model", {}),
        num_timesteps=cfg["diffusion"].get("num_timesteps", 1000),
        noise_schedule=cfg["diffusion"].get("noise_schedule", "cosine"),
    ).to(device)

    if rank == 0:
        logging.info(f"Model parameters: {model.num_parameters:,}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
