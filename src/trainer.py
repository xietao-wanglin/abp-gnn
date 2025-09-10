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
        else:
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

        self.initial_step = 0
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
            batch_size=self.cfg.val.batch_size,
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
            raise ValueError(f"Unknown model type: {model_type}")
        return model

    def load_trained_model(self, path):
        params = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(params["model_state_dict"])
        self.optimizer.load_state_dict(params["optimizer_state_dict"])
        if self.scheduler and params["scheduler_state_dict"]:
            self.scheduler.load_state_dict(params["scheduler_state_dict"])
        self.initial_step = params["step"]

    def logs(self, loss, metric, type):
        metrics = {
            f"{type}/loss": loss.mean(),
            f"{type}/std_loss": loss.std(),
            f"{type}/metric": metric.mean(),
            f"{type}/std_metric": metric.std(),
        }
        return metrics

    def train_step(self, batch):
        batch = batch.to(self.device)
        y = batch.y
        pred = self.model(batch)
        loss = self.criterion(pred, y)
        metric = self.metric(pred, y)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss, metric

    def val_step(self, batch):
        batch = batch.to(self.device)
        y = batch.y
        pred = self.model(batch)
        loss = self.criterion(pred, y)
        metric = self.metric(pred, y)
        return loss, metric

    def save_ckpt(self, step, path):
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler
                else None,
            },
            path,
        )

    def train(self):
        wandb.init(project="ABP_GNN", name=self.cfg.wandb.name, config=OmegaConf.to_container(self.cfg, resolve=True))
        if self.cfg.train.track_gradients:
            wandb.watch(self.model, log="all")
        
        step = self.initial_step
        while step < self.cfg.train.n_steps + self.initial_step:
            self.model.train()
            for batch in self.train_loader:
                step += 1
                loss, metric = self.train_step(batch)
                if (step % self.cfg.train.log_steps) == 0:
                    train_logs = self.logs(loss, metric, type="train")
                    wandb.log(train_logs, step=step)

                if (step % self.cfg.val.log_steps) == 0:
                    self.model.eval()
                    all_losses, all_metrics = [], []
                    with torch.no_grad():
                        for batch in self.val_loader:
                            loss, metric = self.val_step(batch)
                            all_losses.append(loss)
                            all_metrics.append(metric)
                    combined_loss = torch.cat(all_losses, dim=0)
                    combined_metric = torch.cat(all_metrics, dim=0)
                    val_logs = self.logs(combined_loss, combined_metric, type="val")
                    wandb.log(val_logs, step=step)
                    self.model.train()
                    print(f"Step: {step} --- Validation loss: {val_logs['val/loss']}")

                if (step % self.cfg.train.checkpoint_every) == 0:
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir, f"model_step_{step}.pt"
                    )
                    self.save_ckpt(step, checkpoint_path)

            if self.scheduler is not None:
                self.scheduler.step()

        wandb.finish()
