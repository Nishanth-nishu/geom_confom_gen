"""
trainer.py — DDP-aware training loop for ConformerDiffusion.

Features:
    - Single-GPU and multi-GPU (DDP) training via a unified interface.
    - Best-checkpoint tracking on MAT-R (lower = better).
    - Optional WandB logging.
    - Cosine LR schedule with linear warmup.
    - Gradient clipping.
    - Periodic MAT-R validation on a fixed val subset.
"""

import logging
import math
import os
import time
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from geomodel.models.diffusion import ConformerDiffusion
from geomodel.training.geometry import GeometryLoss
from geomodel.evaluation.metrics import evaluate_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learning rate schedule: cosine with linear warmup
# ---------------------------------------------------------------------------

def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    max_epochs: int,
    min_lr_ratio: float = 1e-3,
) -> LambdaLR:
    """
    Cosine decay schedule with linear warmup (BERT-style).

    lr(epoch) = lr_max · [warmup_linear | cosine_decay]
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(warmup_epochs, 1))
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Training manager for ConformerDiffusion.

    Handles the full training loop including:
        - DDP wrapping (optional, auto-detected from distributed env)
        - AdamW optimizer + cosine-warmup scheduler
        - Geometry loss via GeometryLoss
        - Periodic MAT-R validation
        - Checkpoint save/load (best + latest)
        - WandB logging (optional)

    Args:
        model      : ConformerDiffusion instance (unwrapped).
        train_loader, val_loader: DataLoaders.
        cfg        : Training configuration dict (from YAML).
        device     : Target torch device.
        rank       : DDP rank (0 for single-GPU).
        world_size : Total DDP processes (1 for single-GPU).
    """

    def __init__(
        self,
        model: ConformerDiffusion,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)

        # Move model to device and wrap with DDP if needed
        self.model_raw = model.to(device)
        if world_size > 1:
            self.model = DDP(self.model_raw, device_ids=[rank])
        else:
            self.model = self.model_raw

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Geometry loss (used inside compute_loss via closure)
        self.geo_loss_fn = GeometryLoss().to(device)

        # Optimizer
        opt_cfg = cfg.get("optimizer", {})
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=opt_cfg.get("lr", 5e-4),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        # Scheduler
        sched_cfg = cfg.get("scheduler", {})
        self.scheduler = build_cosine_warmup_scheduler(
            self.optimizer,
            warmup_epochs=sched_cfg.get("warmup_epochs", 10),
            max_epochs=cfg["training"]["max_epochs"],
            min_lr_ratio=sched_cfg.get("min_lr", 1e-6) / opt_cfg.get("lr", 5e-4),
        )

        # Training state
        self.epoch = 0
        self.best_matr = float("inf")

        # WandB (optional)
        self.use_wandb = cfg.get("logging", {}).get("use_wandb", False)
        if self.use_wandb and self.is_main:
            import wandb
            wandb.init(
                project=cfg["logging"].get("project", "geo-model"),
                name=cfg["logging"].get("run_name", "run"),
                config=cfg,
            )

    def _geo_loss_callable(self, pos, atom_types, edge_index, bond_types, batch_idx):
        """Callable wrapper for geometry loss passed into compute_loss."""
        include_angles = self.cfg["loss"].get("include_angles", True)
        return self.geo_loss_fn(
            pos, atom_types, edge_index, bond_types, batch_idx,
            include_angles=include_angles,
        )

    def _train_one_epoch(self) -> Dict[str, float]:
        """Run one full training epoch. Returns mean loss dict."""
        self.model.train()
        loss_cfg = self.cfg.get("loss", {})
        train_cfg = self.cfg.get("training", {})

        total_loss_sum = total_mse_sum = total_geo_sum = 0.0
        n_batches = 0

        for batch in self.train_loader:
            # Move to device
            x0         = batch["coordinates"].to(self.device)
            atom_types = batch["atom_types"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            bond_types = batch["bond_types"].to(self.device)
            batch_idx  = batch["batch_idx"].to(self.device)

            self.optimizer.zero_grad()

            # Unwrap DDP for compute_loss (passes geo_loss_fn via closure)
            model_core = self.model.module if self.world_size > 1 else self.model
            loss_dict = model_core.compute_loss(
                x0=x0,
                atom_types=atom_types,
                edge_index=edge_index,
                bond_types=bond_types,
                batch_idx=batch_idx,
                geo_loss_fn=self._geo_loss_callable,
                geo_weight=loss_cfg.get("geo_weight", 0.5),
                geo_t_fraction=loss_cfg.get("geo_t_fraction", 0.3),
                min_snr_gamma=loss_cfg.get("min_snr_gamma", 5.0),
            )

            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                train_cfg.get("gradient_clip", 1.0)
            )
            self.optimizer.step()

            total_loss_sum += loss_dict["total"].item()
            total_mse_sum  += loss_dict["mse"].item()
            total_geo_sum  += loss_dict["geo"].item()
            n_batches += 1

        n = max(n_batches, 1)
        return {
            "loss/total": total_loss_sum / n,
            "loss/mse":   total_mse_sum / n,
            "loss/geo":   total_geo_sum / n,
            "lr":         self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run COV-R / MAT-R evaluation on a fixed val subset."""
        self.model.eval()
        train_cfg = self.cfg.get("training", {})
        n_mols = train_cfg.get("val_n_mols", 300)
        n_gen  = train_cfg.get("val_n_gen", 10)

        model_core = self.model.module if self.world_size > 1 else self.model

        refs, gens = [], []
        count = 0
        for batch in self.val_loader:
            if count >= n_mols:
                break

            x0         = batch["coordinates"].to(self.device)
            atom_types = batch["atom_types"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            bond_types = batch["bond_types"].to(self.device)
            batch_idx  = batch["batch_idx"].to(self.device)

            B = int(batch_idx.max()) + 1
            for mol_i in range(B):
                if count >= n_mols:
                    break
                mask = batch_idx == mol_i
                x0_i   = x0[mask]
                at_i   = atom_types[mask]
                bi_i   = torch.zeros(mask.sum(), dtype=torch.long, device=self.device)

                # Find edges belonging to this molecule
                row, col = edge_index
                emask = mask[row] & mask[col]
                local = torch.full((mask.size(0),), -1, dtype=torch.long, device=self.device)
                local[mask] = torch.arange(mask.sum(), device=self.device)
                ei_i  = local[edge_index[:, emask]]
                bt_i  = bond_types[emask]

                # Generate n_gen conformers
                gen_set = [
                    model_core.sample(
                        at_i, ei_i, bt_i, bi_i,
                        num_steps=train_cfg.get("val_num_steps", 20)
                    ).cpu().numpy()
                    for _ in range(n_gen)
                ]
                refs.append([x0_i.cpu().numpy()])
                gens.append(gen_set)
                count += 1

        metrics = evaluate_metrics(refs, gens, threshold=0.5)
        return {
            "val/cov_r": metrics["cov_r"],
            "val/mat_r": metrics["mat_r"],
            "val/cov_p": metrics["cov_p"],
            "val/mat_p": metrics["mat_p"],
        }

    def save_checkpoint(self, filename: str, extra: Optional[Dict] = None) -> None:
        """Save model, optimizer, scheduler, and training state."""
        if not self.is_main:
            return
        save_dir = self.cfg["checkpointing"]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        state = {
            "epoch": self.epoch,
            "best_matr": self.best_matr,
            "model_state": self.model_raw.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "cfg": self.cfg,
        }
        if extra:
            state.update(extra)
        torch.save(state, os.path.join(save_dir, filename))

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a saved checkpoint."""
        state = torch.load(path, map_location=self.device)
        self.model_raw.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.epoch = state.get("epoch", 0)
        self.best_matr = state.get("best_matr", float("inf"))
        logger.info(f"Resumed from epoch {self.epoch} (best MAT-R: {self.best_matr:.4f})")

    def train(self) -> None:
        """Run the full training loop."""
        train_cfg = self.cfg.get("training", {})
        max_epochs  = train_cfg.get("max_epochs", 1000)
        log_every   = train_cfg.get("log_every", 50)
        val_every   = train_cfg.get("val_every", 50)
        save_every  = self.cfg["checkpointing"].get("save_every", 50)

        if self.is_main:
            logger.info(
                f"Training for {max_epochs} epochs | "
                f"params={self.model_raw.num_parameters:,}"
            )

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            t0 = time.time()

            if self.world_size > 1:
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self._train_one_epoch()
            self.scheduler.step()

            log_data = {**train_metrics, "epoch": epoch}

            # Validation
            if epoch % val_every == 0 or epoch == max_epochs - 1:
                val_metrics = self._validate()
                log_data.update(val_metrics)

                matr = val_metrics["val/mat_r"]
                if matr < self.best_matr and self.is_main:
                    self.best_matr = matr
                    self.save_checkpoint("best.pt", {"val_metrics": val_metrics})
                    logger.info(f"  → New best MAT-R: {matr:.4f} Å")

            # Periodic checkpoint
            if epoch % save_every == 0 and self.is_main:
                self.save_checkpoint(f"epoch_{epoch:04d}.pt")

            # Logging
            if epoch % log_every == 0 and self.is_main:
                elapsed = time.time() - t0
                logger.info(
                    f"Epoch {epoch:4d}/{max_epochs} | "
                    f"loss={train_metrics['loss/total']:.4f} "
                    f"mse={train_metrics['loss/mse']:.4f} "
                    f"geo={train_metrics['loss/geo']:.4f} "
                    f"lr={train_metrics['lr']:.2e} | "
                    f"{elapsed:.1f}s"
                )

            if self.use_wandb and self.is_main:
                import wandb
                wandb.log(log_data)

        # Save final checkpoint
        self.save_checkpoint("final.pt")
        if self.is_main:
            logger.info(f"Training complete. Best MAT-R: {self.best_matr:.4f} Å")
