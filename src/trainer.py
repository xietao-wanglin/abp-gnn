from src.models.gnn import GNN
from src.models.gns import GNS, StochasticGNS
from src.utils import (
    discrete_simulation,
    ParticleDataset,
    apply_periodic_boundary,
    compute_graph,
)


import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import wandb

import numpy as np

import os
import json
import random
from glob import glob


class Trainer:
    def __init__(self, config, config_path):
        self.cfg = config
        if self.cfg.seed is None:
            self.seed = random.randint(0, 2**31)
        else:
            self.seed = self.cfg.seed
        self.set_seed(self.seed)
        self.dtype = torch.float

        self.device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
        self.script_dir = os.path.dirname(config_path)
        self.checkpoint_dir = os.path.join(self.script_dir, "ckp")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.load_data(self.cfg.dataset)
        if self.cfg.test.active:
            self.prepare_test(self.cfg.data.subset_samples)
        self.model = self.create_model()

        self.stochastic = getattr(self.model, "stochastic", False)

        num_params = sum(p.numel() for p in self.model.parameters())
        wandb.config.update(
            {"model": {"num_parameters": num_params}}, allow_val_change=True
        )

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
        val_glob = sorted(
            glob(f"{self.script_dir}/../../datasets/{dataset}/data/simulation_test_*")
        )
        with open(f"{self.script_dir}/../../datasets/{dataset}/metadata.json") as f:
            self.metadata = json.load(f)

        self.train_simulations = [
            torch.tensor(np.load(f), device=self.device, dtype=self.dtype)
            for f in train_glob
        ]
        self.val_simulations = [
            torch.tensor(np.load(f), device=self.device, dtype=self.dtype)
            for f in val_glob
        ]

        particle_type_train_glob = sorted(
            glob(f"{self.script_dir}/../../datasets/{dataset}/data/particle_train_*")
        )
        particle_type_test_glob = sorted(
            glob(f"{self.script_dir}/../../datasets/{dataset}/data/particle_test_*")
        )
        if particle_type_train_glob:
            self.train_particle_type = [
                np.load(particle_type) for particle_type in particle_type_train_glob
            ]
            self.test_particle_type = [
                np.load(particle_type) for particle_type in particle_type_test_glob
            ]
        else:
            self.train_particle_type = None
            self.test_particle_type = None

        data_pairs_train = discrete_simulation(
            self.train_simulations,
            particle_type_list=self.train_particle_type,
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
        data_pairs_val = discrete_simulation(
            self.val_simulations,
            particle_type_list=self.test_particle_type,
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
        val_dataset = ParticleDataset(data_pairs_val)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.val.batch_size,
        )

    def get_activation(self, name):
        if name == "silu":
            return nn.SiLU()
        elif name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "linear":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def create_model(self):
        model_type = self.cfg.model.name
        if model_type == "GNN":
            model = (
                GNN(
                    n_layers=self.cfg.model.n_layers,
                    in_node_nf=self.cfg.model.in_node_nf,
                    out_node_nf=self.cfg.model.out_node_nf,
                    in_edge_nf=self.cfg.model.in_edge_nf,
                    hidden_nf=self.cfg.model.hidden_nf,
                    encoder_depth=self.cfg.model.encoder_depth,
                    decoder_depth=self.cfg.model.decoder_depth,
                    edge_mlp_depth=self.cfg.model.edge_mlp_depth,
                    node_mlp_depth=self.cfg.model.node_mlp_depth,
                    device=self.device,
                    dropout=self.cfg.model.dropout,
                    norm=self.cfg.model.norm,
                    activation=self.get_activation(self.cfg.model.activation),
                )
                .to(dtype=self.dtype)
                .to(device=self.device)
            )
        elif model_type == "GNS":
            model = (
                GNS(
                    n_layers=self.cfg.model.n_layers,
                    in_node_nf=self.cfg.model.in_node_nf,
                    out_node_nf=self.cfg.model.out_node_nf,
                    in_edge_nf=self.cfg.model.in_edge_nf,
                    hidden_nf=self.cfg.model.hidden_nf,
                    encoder_depth=self.cfg.model.encoder_depth,
                    decoder_depth=self.cfg.model.decoder_depth,
                    edge_mlp_depth=self.cfg.model.edge_mlp_depth,
                    node_mlp_depth=self.cfg.model.node_mlp_depth,
                    device=self.device,
                    dropout=self.cfg.model.dropout,
                    norm=self.cfg.model.norm,
                    num_particle_types=self.cfg.model.n_particle_types,
                    particle_type_embedding_size=self.cfg.model.particle_embedding,
                    activation=self.get_activation(self.cfg.model.activation),
                )
                .to(dtype=self.dtype)
                .to(device=self.device)
            )
        elif model_type == "S-GNS":
            model = (
                StochasticGNS(
                    n_layers=self.cfg.model.n_layers,
                    in_node_nf=self.cfg.model.in_node_nf,
                    out_node_nf=self.cfg.model.out_node_nf,
                    in_edge_nf=self.cfg.model.in_edge_nf,
                    hidden_nf=self.cfg.model.hidden_nf,
                    encoder_depth=self.cfg.model.encoder_depth,
                    decoder_depth=self.cfg.model.decoder_depth,
                    edge_mlp_depth=self.cfg.model.edge_mlp_depth,
                    node_mlp_depth=self.cfg.model.node_mlp_depth,
                    device=self.device,
                    dropout=self.cfg.model.dropout,
                    norm=self.cfg.model.norm,
                    num_particle_types=self.cfg.model.n_particle_types,
                    particle_type_embedding_size=self.cfg.model.particle_embedding,
                    activation=self.get_activation(self.cfg.model.activation),
                )
                .to(dtype=self.dtype)
                .to(device=self.device)
            )
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

    def get_grad_norm(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5

    def train_step(self, batch):
        batch = batch.to(self.device)
        y = batch.y
        particle_type = getattr(batch, "particle_type", None)
        context = self.model(batch, particle_type)
        if not self.stochastic:
            pred = context
            loss = self.criterion(pred, y)
            metric = self.metric(pred, y)
        else:
            loss = self.model.compute_nll(pred, y)
            metric = loss
        self.optimizer.zero_grad()
        loss.mean().backward()

        grad_norm = self.get_grad_norm()
        self.optimizer.step()

        return loss, metric, grad_norm

    def val_step(self, batch):
        batch = batch.to(self.device)
        y = batch.y
        particle_type = getattr(batch, "particle_type", None)
        context = self.model(batch, particle_type)
        if not self.stochastic:
            pred = context
            loss = self.criterion(pred, y)
            metric = self.metric(pred, y)
        else:
            loss = self.model.compute_nll(pred, y)
            metric = loss
        return loss, metric

    def prepare_test(self, timesteps):
        initial_states = []
        ground_truths = []
        particle_types = []
        for sim_idx, sim in enumerate(self.val_simulations):
            if self.test_particle_type is not None:
                p_type = torch.tensor(
                    self.test_particle_type[sim_idx],
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                p_type = None
            for t in timesteps:
                x_init = sim[t]
                x_bounded = apply_periodic_boundary(x_init)
                initial_states.append(x_bounded)

                gt_trajectory = apply_periodic_boundary(sim[t + 1 : t + 21])
                ground_truths.append(gt_trajectory)
                particle_types.append(p_type)

        self.initial_states = initial_states
        self.ground_truths = ground_truths
        self.particle_types = particle_types
        self.rollout_length = len(ground_truths[0])

        return len(initial_states)

    def compute_rollout(self):
        num_trajectories = len(self.initial_states)

        predictions = []
        for i in range(num_trajectories):
            N_i = self.initial_states[i].shape[1]
            pred_trajectory = torch.zeros(
                self.rollout_length + 1,
                3,
                N_i,
                dtype=self.dtype,
                device=self.device,
            )
            pred_trajectory[0] = self.initial_states[i]
            predictions.append(pred_trajectory)

        for t in range(self.rollout_length):
            batch_data_list = []
            trajectory_sizes = []
            particle_types = []

            for traj_idx in range(num_trajectories):
                x = predictions[traj_idx][t].clone()
                N_i = x.shape[1]
                trajectory_sizes.append(N_i)

                x_bounded = apply_periodic_boundary(x)

                edge_index, edge_attr = compute_graph(
                    x_bounded,
                    method=self.cfg.data.cluster.method,
                    p=self.cfg.data.cluster.parameter,
                    use_distance=self.cfg.data.features.use_distance,
                    use_rel_pos=self.cfg.data.features.use_rel_pos,
                    boundary_type=self.cfg.data.boundary_type,
                    device=self.device,
                )
                features = []

                if self.cfg.data.features.use_pos:
                    features.append(x_bounded[:2].T)
                if self.cfg.data.features.use_angle:
                    features.append(x_bounded[2].unsqueeze(0).T)
                if features:
                    data_input = torch.cat(features, dim=1)
                else:
                    batch_size = x_bounded.shape[1]
                    data_input = torch.ones(
                        batch_size, 1, device=self.device, dtype=self.dtype
                    )
                data = Data(x=data_input, edge_index=edge_index, edge_attr=edge_attr)
                batch_data_list.append(data)

                if self.particle_types[traj_idx] is not None:
                    particle_types.append(self.particle_types[traj_idx])
                else:
                    particle_types.append(None)

            batched_data = Batch.from_data_list(batch_data_list).to(self.device)
            batched_particle_type = (
                torch.cat([pt for pt in particle_types if pt is not None])
                if particle_types[0] is not None
                else None
            )

            with torch.no_grad():
                if batched_particle_type is not None:
                    forward_pass = self.model(batched_data, batched_particle_type)
                else:
                    forward_pass = self.model(batched_data)
                if self.stochastic:
                    forward_pass = self.model.sample(forward_pass).squeeze(1)
                if not self.cfg.data.features.target_vel:
                    batched_pred = forward_pass
                else:
                    if self.metadata["angular_std"] > 0:
                        batched_vel_pred = (
                            forward_pass[:, :2] * self.metadata["vel_std"]
                            + self.metadata["vel_mean"]
                        )
                        batched_theta_vel_pred = (
                            forward_pass[:, 2] * self.metadata["angular_std"]
                            + self.metadata["angular_mean"]
                        )
                        batched_pred = torch.cat(
                            [batched_vel_pred, batched_theta_vel_pred.unsqueeze(1)],
                            dim=1,
                        )
                    else:
                        batched_pred = (
                            forward_pass * self.metadata["vel_std"]
                            + self.metadata["vel_mean"]
                        )

            start_idx = 0
            for traj_idx in range(num_trajectories):
                N_i = trajectory_sizes[traj_idx]
                end_idx = start_idx + N_i

                traj_pred = batched_pred[start_idx:end_idx]
                current_state = predictions[traj_idx][t].clone()
                if not (self.metadata["angular_std"] > 0):
                    theta_vel = (
                        torch.ones(N_i, 1, device=self.device, dtype=self.dtype)
                        * self.metadata["angular_mean"]
                    )
                    if not self.cfg.data.features.target_vel:
                        theta_vel = theta_vel + current_state[2].unsqueeze(0).T
                    full_vel_pred = torch.cat([traj_pred, theta_vel], dim=-1)
                else:
                    full_vel_pred = traj_pred
                next_state = current_state + full_vel_pred.T
                if not self.cfg.data.features.target_vel:
                    next_state = full_vel_pred.T

                predictions[traj_idx][t + 1] = apply_periodic_boundary(next_state)

                start_idx = end_idx

        self.predictions = predictions

    def test_metrics(self):
        mse_all = []

        for pred, gt in zip(self.predictions, self.ground_truths):
            pred_rollout = pred[1:]

            mse_1 = torch.mean((pred_rollout[0, :2] - gt[0, :2]) ** 2)
            mse_5 = torch.mean((pred_rollout[:5, :2] - gt[:5, :2]) ** 2)
            mse_10 = torch.mean((pred_rollout[:10, :2] - gt[:10, :2]) ** 2)
            mse_20 = torch.mean((pred_rollout[:20, :2] - gt[:20, :2]) ** 2)
            mse_all.append([mse_1.item(), mse_5.item(), mse_10.item(), mse_20.item()])

        mse_all = np.array(mse_all)
        metrics = {
            "test/mse_1": mse_all[:, 0].mean(),
            "test/mse_5": mse_all[:, 1].mean(),
            "test/mse_10": mse_all[:, 2].mean(),
            "test/mse_20": mse_all[:, 3].mean(),
        }

        return metrics

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
        if self.cfg.wandb.track_gradients:
            wandb.watch(self.model, log="all", log_freq=self.cfg.wandb.gradients_every)

        step = self.initial_step
        while step < self.cfg.train.n_steps + self.initial_step:
            self.model.train()
            for batch in self.train_loader:
                step += 1
                loss, metric, grad_norm = self.train_step(batch)
                if (step % self.cfg.train.log_steps) == 0:
                    train_logs = self.logs(loss, metric, type="train")
                    wandb.log(train_logs, step=step)
                    wandb.log({"gradients/total": grad_norm}, step=step)

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

                if ((step % self.cfg.test.log_steps) == 0) and self.cfg.test.active:
                    self.model.eval()
                    self.compute_rollout()
                    metrics = self.test_metrics()
                    wandb.log(metrics, step=step)
                    self.model.train()

            if self.scheduler is not None:
                self.scheduler.step()

        wandb.finish()
