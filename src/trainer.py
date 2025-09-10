from src.models import GNN
from src.utils import (
    discrete_simulation,
    ParticleDataset,
)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
import wandb

import numpy as np
from tqdm import tqdm

import os
import json
import random
from glob import glob
from omegaconf import OmegaConf


class Trainer:
    def __init__(self, config):
        config_path = os.path.abspath(config)
        with open(config_path, "r") as f:
            self.cfg = OmegaConf.load(f)
        if self.cfg.seed is None:
            self.seed = random.randint(0, 2**31)
        self.seed = self.cfg.seed
        self.set_seed(self.seed)
        self.dtype = torch.double

        self.device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
        self.script_dir = os.path.dirname(config_path)
        self.checkpoint_dir = os.path.join(self.script_dir, "ckp")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.load_data(self.cfg.dataset)
        self.model = self.create_model()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.scheduler = None
        self.criterion = nn.MSELoss(reduction="none")
        self.metric = nn.L1Loss(reduction="none")

        self.history = {
            "train_loss": [],
            "train_metric": [],
            "val_loss": [],
            "val_metric": [],
            "best_val_loss": float("inf"),
        }
        self.initial_epoch = 0
        if self.cfg.train.base_model_path is not None:
            self.load_trained_model(self.cfg.train.base_model_path)

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def load_data(self, dataset):
        train_glob = sorted(
            glob(f"{self.script_dir}/../../datasets/{dataset}/data/simulation_train_*")
        )
        test_glob = sorted(
            glob(f"{self.script_dir}/../../datasets/{dataset}/data/simulation_test_*")
        )
        with open(f"{self.script_dir}/../../datasets/{dataset}/metadata.json") as f:
            self.metadata = json.load(f)

        self.train_simulations = [
            torch.tensor(np.load(f), device=self.device, dtype=self.dtype)
            for f in train_glob
        ]
        self.test_simulations = [
            torch.tensor(np.load(f), device=self.device, dtype=self.dtype)
            for f in test_glob
        ]

        data_pairs_train = discrete_simulation(
            self.train_simulations,
            subset=self.cfg.data.subset,
            subset_samples=self.cfg.data.subset_samples,
            cluster_method=self.cfg.data.cluster.method,
            p=self.cfg.data.cluster.parameter,
            use_distance=self.cfg.data.features.use_distance,
            use_rel_pos=self.cfg.data.features.use_rel_pos,
            use_pos=self.cfg.data.features.use_pos,
            target_vel=self.cfg.data.features.target_vel,
            stats=self.metadata,
            boundary_type=self.cfg.data.boundary_type,
            dtype=self.dtype,
            device=self.device,
        )
        data_pairs_test = discrete_simulation(
            self.test_simulations,
            subset=self.cfg.data.subset,
            subset_samples=self.cfg.data.subset_samples,
            cluster_method=self.cfg.data.cluster.method,
            p=self.cfg.data.cluster.parameter,
            use_distance=self.cfg.data.features.use_distance,
            use_rel_pos=self.cfg.data.features.use_rel_pos,
            use_pos=self.cfg.data.features.use_pos,
            target_vel=self.cfg.data.features.target_vel,
            stats=self.metadata,
            boundary_type=self.cfg.data.boundary_type,
            dtype=self.dtype,
            device=self.device,
        )

        train_dataset = ParticleDataset(data_pairs_train)
        test_dataset = ParticleDataset(data_pairs_test)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
        )

        self.val_loader = DataLoader(
            test_dataset,
            batch_size=len(test_dataset),
        )

    def create_model(self):
        model_type = self.cfg.model.name
        if model_type == "GNN":
            model = GNN(
                n_layers=self.cfg.model.n_layers,
                in_node_nf=self.cfg.model.in_node_nf,
                out_node_nf=self.cfg.model.out_node_nf,
                in_edge_nf=self.cfg.model.in_edge_nf,
                hidden_nf=self.cfg.model.hidden_nf,
                device=self.device,
                dropout=self.cfg.model.dropout,
                norm=self.cfg.model.norm,
            ).to(dtype=self.dtype).to(device=self.device)
        else:
            print("Model name not valid.")
        return model

    def load_trained_model(self, path):
        params = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(params["model_state_dict"])
        self.optimizer.load_state_dict(params["optimizer_state_dict"])
        if self.scheduler and params["scheduler_state_dict"]:
            self.scheduler.load_state_dict(params["scheduler_state_dict"])
        self.initial_epoch = params["epoch"]

    def train_step(self, batch):
        batch = batch.to(self.device)
        y = batch.y
        pred = self.model(batch)
        loss = self.criterion(pred, y).mean()
        metric_train = self.metric(pred, y).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        wandb.log(
            {
                "train/batch_loss": loss.item(),
                "train/batch_metric": metric_train.item(),
            }
        )
        return loss, metric_train

    def val_step(self, batch):
        batch = batch.to(self.device)
        y = batch.y
        pred = self.model(batch)
        loss = self.criterion(pred, y).mean()
        metric_val = self.metric(pred, y).mean()
        return loss, metric_val

    def save_ckpt(self, epoch, train_loss, val_loss, path):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler
                else None,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            path,
        )

    def train(self):
        wandb.init(project="ABP_GNN", name=self.cfg.wandb.name, config=OmegaConf.to_container(self.cfg, resolve=True))
        if self.cfg.train.track_gradients:
            wandb.watch(self.model, log="all", log_freq=125)
        for epoch in range(
            self.initial_epoch, self.cfg.train.n_epochs + self.initial_epoch
        ):
            self.model.train()
            train_losses = []
            train_metrics = []
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.cfg.train.n_epochs + self.initial_epoch}",
            )
            for batch_idx, batch in enumerate(pbar):
                loss, metric_train = self.train_step(batch)
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})
                train_losses.append(loss.item())
                train_metrics.append(metric_train.item())

            self.model.eval()
            val_losses = []
            val_metrics = []
            with torch.no_grad():
                for batch in self.val_loader:
                    loss, metric_val = self.val_step(batch)
                    val_losses.append(loss.item())
                    val_metrics.append(metric_val.item())

            avg_train_loss = np.mean(train_losses)
            std_train_loss = np.std(train_losses)
            avg_val_loss = np.mean(val_losses)
            std_val_loss = np.std(val_losses)

            avg_train_metric = np.mean(train_metrics)
            std_train_metric = np.std(train_metrics)
            avg_val_metric = np.mean(val_metrics)
            std_val_metric = np.std(val_metrics)

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["train_metric"].append(avg_train_metric)
            self.history["val_metric"].append(avg_val_metric)

            if self.scheduler is not None:
                self.scheduler.step()

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": avg_train_loss,
                    "train/std_loss": std_train_loss,
                    "train/metric": avg_train_metric,
                    "train/std_metric": std_train_metric,
                    "val/loss": avg_val_loss,
                    "val/std_loss": std_val_loss,
                    "val/metric": avg_val_metric,
                    "val/std_metric": std_val_metric,
                    "lr": self.scheduler.get_last_lr()[0]
                    if self.scheduler
                    else self.cfg.train.lr,
                }
            )

            if avg_val_loss < self.history["best_val_loss"]:
                self.history["best_val_loss"] = avg_val_loss
                checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                self.save_ckpt(epoch, avg_train_loss, avg_val_loss, checkpoint_path)

            if (epoch + 1) % self.cfg.train.checkpoint_every == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"model_epoch_{epoch + 1}.pt"
                )
                self.save_ckpt(epoch, avg_train_loss, avg_val_loss, checkpoint_path)

            print(f"\nEpoch {epoch + 1}/{self.cfg.train.n_epochs + self.initial_epoch}")
            print(f"Train Loss: {avg_train_loss:.8f}")
            print(f"Val Loss: {avg_val_loss:.8f}")
            print("-" * 30)

        wandb.finish()
