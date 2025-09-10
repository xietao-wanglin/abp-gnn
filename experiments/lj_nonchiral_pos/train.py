from src.models import GNN
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
from torch_geometric.data import Data
import wandb

import numpy as np
from tqdm import tqdm

import os
import json
from glob import glob
from typing import Optional, List, Dict

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")
dtype = torch.double
seed = 0
torch.manual_seed(seed)
np.random.seed(0)
wandb.login()


def train(
    model: nn.Module,
    cluster_method: str,
    cluster_parameter: float | int,
    batch_size: Optional[int] = 1,
    n_epochs: Optional[int] = 10000,
    lr: Optional[float] = 1e-4,
    weight_decay: Optional[float] = 1e-8,
    device: Optional[str | torch.device] = "cpu",
    checkpoint_dir: Optional[str] = "checkpoints",
    checkpoint_every: Optional[int] = 200,
    hist_filename: Optional[str] = "training_history",
    dataset: Optional[str] = "nonchiral_lj",
    subset: Optional[bool] = False,
    subset_samples: Optional[List] = None,
    base_model_path: Optional[str] = None,
) -> Dict:
    """
    Train the GNN model.

    Parameters
    ----------
    model: nn.Module
        Model instance.
    cluster_method: str
        Graph creation method, either 'radius' or 'knn'.
    cluster_parameter: float or int
        Parameter used in `cluster_method`.
    batch_size: int, optional
        Mini batch size.
    n_epochs: int, optional
        Number of epochs to train, default is 100.
    lr: float, optional
        Learning rate of AdamW optimiser, default is 5e-4.
    weight_decay: float, optional
        Weight decay of AdamW optimiser, default is 1e-4.
    device: str or torch.device, optional
        Either 'cpu', 'cuda' or torch.device instance, default is 'cpu'.
    checkpoint_dir: str, optional
        Directory to save checkpoints, default is 'checkpoints'.
    checkpoint_every: int, optional
        Save checkpoint every N epochs, default is 10.
    hist_filename: str, optional
        Name of training history JSON file, defualt is 'training_history'.
    subset: bool, optional
        If True, use a subset of the trajectories instead of full simulations, default is False.
    subset_samples: List, optional
        If provided, samples to choose from trajectories if `subset=True`, otherwise random, default is None.
    base_model_path: str, optional
        If provided, use path model to load weights, default is None.

    Returns
    -------
    history: Dict
        Training history.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, checkpoint_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_glob = sorted(
        glob(f"{script_dir}/../../datasets/{dataset}/data/simulation_train_*")
    )
    test_glob = sorted(
        glob(f"{script_dir}/../../datasets/{dataset}/data/simulation_test_*")
    )
    with open(f"{script_dir}/../../datasets/{dataset}/metadata.json") as f:
        metadata = json.load(f)

    train_simulations = [np.load(sim) for sim in train_glob]
    test_simulations = [np.load(sim) for sim in test_glob]

    data_pairs_train = discrete_simulation(
        train_simulations,
        subset=subset,
        subset_samples=subset_samples,
        cluster_method=cluster_method,
        p=cluster_parameter,
        use_distance=False,
        use_rel_pos=False,
        use_pos=True,
        target_vel=False,
        stats=metadata,
        boundary_type=(1, 1, 1),
        dtype=dtype,
        device=device,
    )
    data_pairs_test = discrete_simulation(
        test_simulations,
        subset=subset,
        subset_samples=subset_samples,
        cluster_method=cluster_method,
        p=cluster_parameter,
        use_distance=False,
        use_rel_pos=False,
        use_pos=True,
        target_vel=False,
        boundary_type=(1, 1, 1),
        stats=metadata,
        dtype=dtype,
        device=device,
    )

    train_dataset = ParticleDataset(data_pairs_train)
    test_dataset = ParticleDataset(data_pairs_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=200,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    criterion = nn.MSELoss(reduction="none")
    metric = nn.L1Loss(reduction="none")

    activation = getattr(model, "activation", None)
    train_details = {
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__ if scheduler else None,
        "criterion": criterion.__class__.__name__,
        "metric": metric.__class__.__name__,
        "model": getattr(model, "name", None),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "lr": lr,
        "n_epochs": n_epochs,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "n_layers": getattr(model, "n_layers", None),
        "in_node_nf": getattr(model, "in_node_nf", None),
        "out_node_nf": getattr(model, "out_node_nf", None),
        "latent_nf": getattr(model, "latent_nf", None),
        "in_edge_nf": getattr(model, "in_edge_nf", None),
        "hidden_nf": getattr(model, "hidden_nf", None),
        "activation": activation.__class__.__name__ if activation is not None else None,
        "dropout": getattr(model, "dropout", None),
        "norm": getattr(model, "norm", None),
        "cluster_method": cluster_method,
        "cluster_parameter": cluster_parameter,
        "train_samples": len(train_glob),
        "test_samples": len(test_glob),
    }
    wandb.init(project="ABP_GNN", name="nonchiral_lr", config=train_details)

    details_path = os.path.join(checkpoint_dir, "details.json")
    with open(details_path, "w") as f:
        json.dump(train_details, f, indent=4)

    print(f"Training details saved to {details_path}")

    history = {
        "train_loss": [],
        "train_metric": [],
        "val_loss": [],
        "val_metric": [],
        "best_val_loss": float("inf"),
    }

    initial_epoch = 0
    if base_model_path is not None:
        params = torch.load(base_model_path, map_location=device, weights_only=False)
        model.load_state_dict(params["model_state_dict"])
        optimizer.load_state_dict(params["optimizer_state_dict"])
        if scheduler and params["scheduler_state_dict"]:
            scheduler.load_state_dict(params["scheduler_state_dict"])
        initial_epoch = params["epoch"]

    for epoch in range(initial_epoch, n_epochs + initial_epoch):
        model.train()
        train_losses = []
        train_metrics = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs + initial_epoch}")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            y = batch.y
            pred = model(batch)
            loss = criterion(pred, y).mean()
            metric_train = metric(pred, y).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log(
                {
                    "train/batch_loss": loss.item(),
                    "train/batch_metric": metric_train.item(),
                    "epoch": epoch,
                }
            )
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            train_losses.append(loss.item())
            train_metrics.append(metric_train.item())

        model.eval()
        val_losses = []
        val_metrics = []

        if False:
            mse_1 = []
            mse_5 = []
            mse_10 = []
            mse_20 = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y = batch.y
                pred = model(batch)
                loss = criterion(pred, y).mean()
                metric_val = metric(pred, y).mean()
                if False:
                    gt_trajectory = batch.trajectory.permute(1, 2, 0)
                    rollout = torch.zeros_like(gt_trajectory)
                    vel_pred = pred * metadata["vel_std"] + metadata["vel_mean"]
                    theta_pred = (
                        torch.ones(pred.shape[0]).unsqueeze(1)
                        * metadata["angular_mean"]
                    )
                    pred = torch.cat([vel_pred, theta_pred], dim=-1)
                    rollout[0] = apply_periodic_boundary(
                        (pred + batch.full_x).T, wrap_dims=[0, 1, 2]
                    )  # Rollout manually
                    for roll in range(19):
                        x = rollout[roll]
                        edge_index, edge_attr = compute_graph(
                            x,
                            method="radius",
                            p=0.1,
                            use_distance=False,
                            use_rel_pos=False,
                            boundary_type=(1, 1, 1),
                        )
                        data = Data(
                            x=x[:2].T,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                        )
                        pred = model(data)
                        vel_pred = pred * metadata["vel_std"] + metadata["vel_mean"]
                        theta_pred = (
                            torch.ones(pred.shape[0]).unsqueeze(1)
                            * metadata["angular_mean"]
                        )
                        pred = torch.cat([vel_pred, theta_pred], dim=-1)
                        rollout[roll + 1] = apply_periodic_boundary(
                            (pred + rollout[roll].T).T, wrap_dims=[0, 1, 2]
                        )
                    mse_trajectory = (rollout[:, :2] - gt_trajectory[:, :2]).pow(2)
                    mse_1.append(mse_trajectory[0].mean().item())
                    mse_5.append(mse_trajectory[:5].mean().item())
                    mse_10.append(mse_trajectory[:10].mean().item())
                    mse_20.append(mse_trajectory[:20].mean().item())
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

        if False:
            avg_mse_1 = np.mean(mse_1)
            avg_mse_5 = np.mean(mse_5)
            avg_mse_10 = np.mean(mse_10)
            avg_mse_20 = np.mean(mse_20)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_metric"].append(avg_train_metric)
        history["val_metric"].append(avg_val_metric)
        if scheduler is not None:
            scheduler.step()

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
                "lr": scheduler.get_last_lr()[0] if scheduler else lr,
            }
        )
        if False:
            wandb.log(
                {
                    "val/mse_1": avg_mse_1,
                    "val/mse_5": avg_mse_5,
                    "val/mse_10": avg_mse_10,
                    "val/mse_20": avg_mse_20,
                    "lr": scheduler.get_last_lr()[0] if scheduler else lr,
                }
            )

        if avg_val_loss < history["best_val_loss"]:
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            history["best_val_loss"] = avg_val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                    if scheduler
                    else None,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                checkpoint_path,
            )
            wandb.save(checkpoint_path, base_path=script_dir)

        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"model_epoch_{epoch + 1}.pt"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                    if scheduler
                    else None,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                checkpoint_path,
            )
            wandb.save(checkpoint_path, base_path=script_dir)

        print(f"\nEpoch {epoch + 1}/{n_epochs + initial_epoch}")
        print(f"Train Loss: {avg_train_loss:.8f}")
        print(f"Val Loss: {avg_val_loss:.8f}")
        print("-" * 30)

        history_path = os.path.join(checkpoint_dir, f"{hist_filename}.json")
        with open(history_path, "w") as f:
            json.dump(
                {
                    "train_loss": history["train_loss"],
                    "val_loss": history["val_loss"],
                    "best_val_loss": history["best_val_loss"],
                },
                f,
                indent=4,
            )

    wandb.finish()
    return history


if __name__ == "__main__":
    model = (
        GNN(
            n_layers=4,
            in_node_nf=2,
            out_node_nf=2,
            in_edge_nf=0,
            hidden_nf=64,
            device=device,
            norm=False,
            activation=nn.SiLU(),
        )
        .to(dtype=dtype)
        .to(device=device)
    )

    history = train(
        model=model,
        cluster_method="radius",
        cluster_parameter=10.0,
        batch_size=32,
        checkpoint_every=2,
        n_epochs=10000,
        lr=1e-4,
        weight_decay=1e-8,
        device=device,
        subset=True,
        subset_samples=[0, 20, 40, 60],
        dataset="nonchiral_low_repulsion",
        base_model_path=None,
    )
