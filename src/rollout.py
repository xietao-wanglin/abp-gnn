from src.simulation import ParticleType
from src.utils import (
    apply_periodic_boundary,
    compute_graph,
    create_model,
)
import torch
from torch_geometric.data import Data
import json


def model_rollout(cfg, experiment_name, model_step, timesteps, initial_state, box_length, particles, particle_features=None, 
                  record_every = 10,
    start_record = 0, src_root = "..", dtype=torch.float, device="cpu"):
    with open(f"{src_root}/datasets/{cfg.dataset}/metadata.json") as f:
        metadata = json.load(f)
    model = create_model(cfg)
    checkpoint_path = f"{src_root}/experiments/{experiment_name}/ckp/model_step_{model_step}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    stochastic = getattr(model, "stochastic", False)
    model.eval()
    init = initial_state
    total_records = max(0, (timesteps - start_record) // record_every + 1)

    predictions = torch.zeros(size=(total_records, 3, init.shape[1]), dtype=dtype)

    if start_record == 0:
        predictions[0] = init
    boundary_mask = (particles > ParticleType.BOUNDARY).unsqueeze(1)
    current_state = init
    save_idx = 0
    if start_record == 0:
        predictions[save_idx] = current_state
        save_idx += 1
    for i in range(1, timesteps):
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
        if particle_features is not None:
            features.append(particle_features.T)
        if cfg.data.features.use_pos:
            features.append(x_bounded[:2].T)
        if cfg.data.features.use_angle:
            features.append(x_bounded[2].unsqueeze(0).T)
        if features:
            data_input = torch.cat(features, dim=1)
        else:
            batch_size = x_bounded.shape[1]
            data_input = torch.ones(batch_size, 1, device=device, dtype=dtype)
        if cfg.data.features.separate_coords:
            data = Data(
                x=x_bounded[:2].T,
                theta=x_bounded[2].unsqueeze(0).T,
                h=None,
                edge_index=edge_index,
                edge_attr=edge_attr,
                box_length=torch.tensor(box_length).unsqueeze(0),
                )
        else:
            
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
            if particles is not None:
                mask = particles.unsqueeze(0)
                next_state = (mask * next_state) + ((1 - mask) * current_state)
        next_state = apply_periodic_boundary(
            next_state, dims=[box_length, box_length, 2 * torch.pi]
        )
        if i >= start_record and (i - start_record) % record_every == 0:
            if save_idx < total_records:
                predictions[save_idx] = current_state
                save_idx += 1
        current_state = next_state

    res = predictions.detach().cpu().numpy()
    return res