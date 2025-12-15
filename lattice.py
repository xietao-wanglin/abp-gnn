import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from src.models.gnn import GNN
from src.models.gns import GNS, StochasticGNS
from src.utils import (
    apply_periodic_boundary,
    compute_graph,
)
from src.simulation import ParticleType
from torch_geometric.data import Data
import json
import argparse


def get_activation(name):
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


def create_model(cfg):
    model_type = cfg.model.name
    if model_type == "GNN":
        model = (
            GNN(
                n_layers=cfg.model.n_layers,
                in_node_nf=cfg.model.in_node_nf,
                out_node_nf=cfg.model.out_node_nf,
                in_edge_nf=cfg.model.in_edge_nf,
                hidden_nf=cfg.model.hidden_nf,
                encoder_depth=cfg.model.encoder_depth,
                decoder_depth=cfg.model.decoder_depth,
                edge_mlp_depth=cfg.model.edge_mlp_depth,
                node_mlp_depth=cfg.model.node_mlp_depth,
                device=device,
                dropout=cfg.model.dropout,
                norm=cfg.model.norm,
                activation=get_activation(cfg.model.activation),
            )
            .to(dtype=dtype)
            .to(device=device)
        )
    elif model_type == "GNS":
        model = (
            GNS(
                n_layers=cfg.model.n_layers,
                in_node_nf=cfg.model.in_node_nf,
                out_node_nf=cfg.model.out_node_nf,
                in_edge_nf=cfg.model.in_edge_nf,
                hidden_nf=cfg.model.hidden_nf,
                encoder_depth=cfg.model.encoder_depth,
                decoder_depth=cfg.model.decoder_depth,
                edge_mlp_depth=cfg.model.edge_mlp_depth,
                node_mlp_depth=cfg.model.node_mlp_depth,
                device=device,
                dropout=cfg.model.dropout,
                norm=cfg.model.norm,
                activation=get_activation(cfg.model.activation),
                num_particle_types=cfg.model.n_particle_types,
                particle_type_embedding_size=cfg.model.particle_embedding,
            )
            .to(dtype=dtype)
            .to(device=device)
        )
    elif model_type == "S-GNS":
        model = (
            StochasticGNS(
                n_layers=cfg.model.n_layers,
                in_node_nf=cfg.model.in_node_nf,
                out_node_nf=cfg.model.out_node_nf,
                in_edge_nf=cfg.model.in_edge_nf,
                hidden_nf=cfg.model.hidden_nf,
                encoder_depth=cfg.model.encoder_depth,
                decoder_depth=cfg.model.decoder_depth,
                edge_mlp_depth=cfg.model.edge_mlp_depth,
                node_mlp_depth=cfg.model.node_mlp_depth,
                device=device,
                dropout=cfg.model.dropout,
                norm=cfg.model.norm,
                num_particle_types=cfg.model.n_particle_types,
                particle_type_embedding_size=cfg.model.particle_embedding,
                activation=get_activation(cfg.model.activation),
            )
            .to(dtype=dtype)
            .to(device=device)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def generate_state_with_grid_boundary(lc, n_boundary=400, sigma=0.04):
    n_side = int(np.sqrt(n_boundary))
    box_length = n_side * lc

    x = np.linspace(0, box_length - lc, n_side) + lc / 2
    y = np.linspace(0, box_length - lc, n_side) + lc / 2
    X, Y = np.meshgrid(x, y)
    X, Y = X.flatten()[:n_boundary], Y.flatten()[:n_boundary]

    while True:
        active_x = np.random.rand() * box_length
        active_y = np.random.rand() * box_length
        d = np.sqrt((X - active_x) ** 2 + (Y - active_y) ** 2)
        if np.all(d > 1 * sigma):
            break
    active_theta = np.random.rand() * 2 * np.pi

    all_x = np.concatenate([X, [active_x]])
    all_y = np.concatenate([Y, [active_y]])
    all_theta = np.concatenate([np.zeros(n_boundary), [active_theta]])

    n_total = n_boundary + 1
    particle_type = np.zeros(n_total, dtype=int)
    particle_type[-1] = 1

    initial_state = np.vstack([all_x, all_y, all_theta])
    return (
        torch.tensor(particle_type, dtype=torch.int),
        torch.tensor(initial_state, dtype=torch.float),
        box_length,
    )


def run_sim(particles, initial_state, box_length, n_sims=200):
    all_sims = torch.zeros(n_sims, timesteps)
    for j in range(n_sims):
        init = initial_state
        predictions = torch.zeros(size=(timesteps, 3, init.shape[1]), dtype=torch.float)
        predictions[0] = init

        boundary_mask = (particles > ParticleType.BOUNDARY).unsqueeze(1)
        model.eval()
        for i in range(timesteps - 1):
            x = predictions[i].clone()
            x_bounded = apply_periodic_boundary(
                x, dims=[box_length, box_length, 2 * torch.pi]
            )
            N_i = x.shape[1]
            edge_index, edge_attr = compute_graph(
                x_bounded,
                method=cfg.data.cluster.method,
                p=cfg.data.cluster.parameter,
                use_distance=cfg.data.features.use_distance,
                use_rel_pos=cfg.data.features.use_rel_pos,
                box_length=box_length,
                boundary_type=cfg.data.boundary_type,
                device=device,
            )
            features = []
            if cfg.data.features.use_pos:
                features.append(x_bounded[:2].T)
            if cfg.data.features.use_angle:
                features.append(x_bounded[2].unsqueeze(0).T)
            if features:
                data_input = torch.cat(features, dim=1)
            else:
                batch_size = x_bounded.shape[1]
                data_input = torch.ones(batch_size, 1, device=device, dtype=dtype)
            data = Data(x=data_input, edge_index=edge_index, edge_attr=edge_attr)
            with torch.no_grad():
                forward_pass = model(data, particles)
                if stochastic:
                    forward_pass = model.sample_mean(forward_pass, n_samples=20)
                if not cfg.data.features.target_vel:
                    pred = forward_pass
                else:
                    if metadata["angular_std"] > 0:
                        vel_pred = (
                            forward_pass[:, :2] * metadata["vel_std"]
                            + metadata["vel_mean"]
                        )
                        theta_vel_pred = (
                            forward_pass[:, 2] * metadata["angular_std"]
                            + metadata["angular_mean"]
                        )
                        pred = torch.cat(
                            [vel_pred, theta_vel_pred.unsqueeze(1)],
                            dim=1,
                        )
                    else:
                        pred = forward_pass * metadata["vel_std"] + metadata["vel_mean"]

            if not (metadata["angular_std"] > 0):
                theta_vel = (
                    torch.ones(N_i, 1, device=device, dtype=dtype)
                    * metadata["angular_mean"]
                )
                if not cfg.data.features.target_vel:
                    theta_vel = theta_vel + x[2].unsqueeze(0).T
                full_vel_pred = torch.cat([pred, theta_vel], dim=-1)
            else:
                full_vel_pred = pred
            next_state = x + (full_vel_pred * boundary_mask).T
            if not cfg.data.features.target_vel:
                next_state = full_vel_pred.T
            predictions[i + 1] = apply_periodic_boundary(
                next_state, dims=[box_length, box_length, 2 * torch.pi]
            )

        disp = predictions[:, :2, -1] - predictions[0, :2, -1].unsqueeze(0)
        squared_displacement = torch.sum(disp**2, dim=1)
        all_sims[j] = squared_displacement
    return all_sims


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index", help="index")
    args = parser.parse_args()

    lcs = [0.2, 0.18, 0.1253, 0.115, 0.11098, 0.09, 0.15, 0.1, 0.095]
    index = int(args.index)
    lc = lcs[index]

    experiment = "chiral_boundary"
    cfg = OmegaConf.load(f"./experiments/{experiment}/cfg.yaml")
    device = "cpu"
    dtype = torch.float
    model_step = 120_000
    timesteps = 16000
    record_every = 10
    start_record = 12000

    total_records = max(0, (timesteps - start_record) // record_every + 1)

    with open(f"./datasets/{cfg.dataset}/metadata.json") as f:
        metadata = json.load(f)
    model = create_model(cfg)
    stochastic = getattr(model, "stochastic", False)
    data = torch.load(
        f"./experiments/{experiment}/ckp/model_step_{model_step}.pt", map_location="cpu"
    )
    model.load_state_dict(data["model_state_dict"])

    n_replications = 1
    model.eval()
    for replic in range(n_replications):
        particles, initial_state, box_length = generate_state_with_grid_boundary(lc=lc)
        density = 100 * 100 * torch.pi * (0.04) ** 2 / (box_length) ** 2
        init = initial_state

        predictions = torch.zeros(size=(total_records, 3, init.shape[1]), dtype=dtype)

        if start_record == 0:
            predictions[0] = init
        boundary_mask = (particles > ParticleType.BOUNDARY).unsqueeze(1)
        current_state = init
        save_idx = 0
        for i in range(timesteps - 1):
            x_bounded = apply_periodic_boundary(
                current_state, dims=[box_length, box_length, 2 * torch.pi]
            )
            N_i = current_state.shape[1]
            edge_index, edge_attr = compute_graph(
                x_bounded,
                method=cfg.data.cluster.method,
                p=cfg.data.cluster.parameter,
                use_distance=cfg.data.features.use_distance,
                use_rel_pos=cfg.data.features.use_rel_pos,
                use_rel_theta=cfg.data.features.use_rel_angle,
                box_length=box_length,
                boundary_type=cfg.data.boundary_type,
                device=device,
            )
            features = []
            if cfg.data.features.use_pos:
                features.append(x_bounded[:2].T)
            if cfg.data.features.use_angle:
                features.append(x_bounded[2].unsqueeze(0).T)
            if features:
                data_input = torch.cat(features, dim=1)
            else:
                batch_size = x_bounded.shape[1]
                data_input = torch.ones(batch_size, 1, device=device, dtype=dtype)
            data = Data(x=data_input, edge_index=edge_index, edge_attr=edge_attr)
            with torch.no_grad():
                forward_pass = model(data, particles)
                if stochastic:
                    forward_pass = model.sample_mean(forward_pass, n_samples=20)
                if not cfg.data.features.target_vel:
                    pred = forward_pass
                else:
                    if metadata["angular_std"] > 0:
                        vel_pred = (
                            forward_pass[:, :2] * metadata["vel_std"]
                            + metadata["vel_mean"]
                        )
                        theta_vel_pred = (
                            forward_pass[:, 2] * metadata["angular_std"]
                            + metadata["angular_mean"]
                        )
                        pred = torch.cat(
                            [vel_pred, theta_vel_pred.unsqueeze(1)],
                            dim=1,
                        )
                    else:
                        pred = forward_pass * metadata["vel_std"] + metadata["vel_mean"]

            if not (metadata["angular_std"] > 0):
                theta_vel = (
                    torch.ones(N_i, 1, device=device, dtype=dtype)
                    * metadata["angular_mean"]
                )
                if not cfg.data.features.target_vel:
                    theta_vel = theta_vel + current_state[2].unsqueeze(0).T
                full_vel_pred = torch.cat([pred, theta_vel], dim=-1)
            else:
                full_vel_pred = pred
            next_state = current_state + (full_vel_pred * boundary_mask).T
            if not cfg.data.features.target_vel:
                next_state = full_vel_pred.T
            next_state = apply_periodic_boundary(
                next_state, dims=[box_length, box_length, 2 * torch.pi]
            )
            if i >= start_record and (i - start_record) % record_every == 0:
                if not (i == 0 and start_record > 0):
                    predictions[save_idx] = next_state
                save_idx += 1
            current_state = next_state
        np.savez(
            f"./lattice_ml/density-{density:.2f}-{replic}.npz",
            predictions=predictions.numpy(),
            box_length=box_length,
            initial_state=initial_state.numpy(),
        )
