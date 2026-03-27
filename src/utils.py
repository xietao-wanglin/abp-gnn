from src.simulation import BoundaryType, ParticleType

import math
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Dataset, Data

from typing import Optional, List


def apply_periodic_boundary(
    positions: torch.Tensor,
    dims: Optional[List[float]] = None,
    wrap_dims: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Applies periodic conditions in three dimensions.

    Parameters
    ----------
    positions: torch.Tensor
        Position Tensor.
    dims: list[float], optional
        The dimensions of the periodic box, default is None.
    wrap_dims: list[int], optional.
        Specifies which dimensions to wrap, default is None.

    Returns
    -------
    positions: torch.Tensor
        Position Tensor after applying boundary conditions.
    """
    if dims is None:
        dims = [1.0, 1.0, 2 * torch.pi]
    dims_tensor = torch.tensor(dims, dtype=positions.dtype, device=positions.device)
    dims_tensor = dims_tensor.view(3, 1)
    if wrap_dims is None:
        wrap_mask = torch.ones(3, dtype=torch.bool, device=positions.device)
    else:
        wrap_mask = torch.zeros(3, dtype=torch.bool, device=positions.device)
        wrap_mask[wrap_dims] = True
    out = positions.clone()
    if positions.ndim == 3:
        for i in range(3):
            if wrap_mask[i]:
                out[:, i, :] = torch.remainder(out[:, i, :], dims_tensor[i])
    elif positions.ndim == 2:
        for i in range(3):
            if wrap_mask[i]:
                out[i, :] = torch.remainder(out[i, :], dims_tensor[i])
    return out


def discrete_simulation(
    simulation_list,
    particle_type_list=None,
    features_list=None,
    subset=False,
    subset_samples=None,
    n_samples=4,
    cluster_method="radius",
    p=0.1,
    noise_std=0.0,
    use_distance=False,
    use_rel_pos=False,
    target_vel=True,
    use_pos=False,
    use_angle=True,
    use_rel_angle=False,
    separate_coords=False,
    boundary_type=None,
    box_length=1,
    stats=None,
    dtype=torch.double,
    device="cpu",
):
    if boundary_type is None:
        boundary_type = (
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
        )
    data_pairs = []
    wrap_dims = []
    for j, type in enumerate(boundary_type):
        if type == BoundaryType.PERIODIC:
            wrap_dims.append(j)

    for idx in range(len(simulation_list)):
        sim = simulation_list[idx]
        if particle_type_list is not None:
            particle_type = particle_type_list[idx]
            particle_type = torch.tensor(particle_type, dtype=torch.int, device=device)
        else:
            particle_type = None

        if features_list is not None:
            particle_features = features_list[idx]
            particle_features = torch.tensor(
                particle_features, dtype=dtype, device=device
            )

        sim = torch.tensor(sim, dtype=dtype)

        num_timesteps = sim.shape[0] - 1

        if subset:
            if subset_samples is None:
                timesteps = torch.randint(
                    0, num_timesteps, size=(min(n_samples, num_timesteps),)
                )
            else:
                timesteps = subset_samples
        else:
            timesteps = range(0, num_timesteps)

        for t in timesteps:
            x = sim[t]
            y = sim[t + 1]

            if noise_std > 0:
                noise = torch.randn_like(x) * noise_std
                noise[2] *= 2 * torch.pi
                if particle_type is not None:
                    noise_mask = (
                        (particle_type == ParticleType.BOUNDARY)
                        .to(dtype=dtype)
                        .view(-1, 1)
                    )
                    noise = noise * (1 - noise_mask)
                x_copy = x.clone()
                x = x_copy + noise
                y_copy = y.clone()
                y = y_copy + noise

            x_bounded = apply_periodic_boundary(
                x, dims=[box_length, box_length, 2 * torch.pi], wrap_dims=wrap_dims
            )

            edge_index, edge_attr = compute_graph(
                x_bounded,
                method=cluster_method,
                p=p,
                use_distance=use_distance,
                use_rel_pos=use_rel_pos,
                use_rel_theta=use_rel_angle,
                boundary_type=boundary_type,
                box_length=box_length,
                device=device,
            )
            if target_vel:
                if stats is not None:
                    vel = y - x_bounded
                    vel[:2] = (vel[:2] - stats["vel_mean"]) / stats["vel_std"]
                    label = vel[:2].T.to(device).to(dtype=dtype)
                    if stats["angular_std"] > 0:
                        vel[2] = (vel[2] - stats["angular_mean"]) / stats["angular_std"]
                        label = vel.T.to(device).to(dtype=dtype)
                else:
                    label = (y - x_bounded).T.to(device).to(dtype=dtype)
            else:
                label = y[:2].T

            features = []

            if use_pos:
                features.append(x_bounded[:2].T)
            if use_angle:
                features.append(x_bounded[2].unsqueeze(0).T)
            if features_list is not None:
                features.append(particle_features.T)
            if features:
                data_input = torch.cat(features, dim=1)
            else:
                batch_size = x_bounded.shape[1]
                data_input = torch.ones(batch_size, 1, device=device, dtype=dtype)

            if not (boundary_type[0] == BoundaryType.PERIODIC):
                bl = None
            else:
                bl = box_length

            if separate_coords:
                data = Data(
                    x=x_bounded[:2].T,
                    theta=x_bounded[2].unsqueeze(0).T,
                    h=particle_features.T if features_list is not None else None,
                    y=label,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    particle_type=particle_type,
                    box_length=bl,
                )
            else:
                data = Data(
                    x=data_input,
                    y=label,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    particle_type=particle_type,
                )

            data_pairs.append(data)

    return data_pairs


def compute_graph(
    x,
    method,
    p,
    use_distance=False,
    use_rel_pos=False,
    use_rel_theta=False,
    box_length=1,
    boundary_type=None,
    device="cpu",
):
    if boundary_type is None:
        boundary_type = (
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
        )

    xy, theta = x[:-1], x[-1]
    xy = xy.transpose(0, 1)
    if method == "radius":
        x_coords = xy[:, 0]
        y_coords = xy[:, 1]

        dx = x_coords.unsqueeze(1) - x_coords.unsqueeze(0)
        dy = y_coords.unsqueeze(1) - y_coords.unsqueeze(0)

        if boundary_type[0] == BoundaryType.PERIODIC:
            dx = dx - box_length * torch.round(dx / box_length)
        if boundary_type[1] == BoundaryType.PERIODIC:
            dy = dy - box_length * torch.round(dy / box_length)

        distances = torch.sqrt(dx.pow(2) + dy.pow(2))
        edges = torch.where(distances < p)
        mask = edges[0] != edges[1]
        edge_index = torch.stack([edges[0][mask], edges[1][mask]])  # Remove self-loops
    elif method == "knn":
        x_coords = xy[:, 0]
        y_coords = xy[:, 1]

        dx = x_coords.unsqueeze(1) - x_coords.unsqueeze(0)
        dy = y_coords.unsqueeze(1) - y_coords.unsqueeze(0)

        if boundary_type[0] == BoundaryType.PERIODIC:
            dx = dx - box_length * torch.round(dx / box_length)
        if boundary_type[1] == BoundaryType.PERIODIC:
            dy = dy - box_length * torch.round(dy / box_length)

        distances = torch.sqrt(dx.pow(2) + dy.pow(2))
        k = int(p)
        distances = distances.fill_diagonal_(float("inf"))  # Avoid self-loops
        _, indices = torch.topk(distances, k=k, dim=1, largest=False)
        row_indices = torch.arange(xy.shape[0], device=device).repeat_interleave(k)
        col_indices = indices.flatten()
        edge_index = torch.stack([row_indices, col_indices])
    else:
        raise ValueError("Invalid method, must be either 'radius' or 'knn'")

    row, col = edge_index
    rel_pos = xy[row] - xy[col]
    theta = theta.unsqueeze(1)
    rel_theta = theta[row] - theta[col]
    if boundary_type[0] == BoundaryType.PERIODIC:
        rel_pos[:, 0] = rel_pos[:, 0] - box_length * torch.round(
            rel_pos[:, 0] / box_length
        )
    if boundary_type[1] == BoundaryType.PERIODIC:
        rel_pos[:, 1] = rel_pos[:, 1] - box_length * torch.round(
            rel_pos[:, 1] / box_length
        )
    rel_theta = rel_theta % (2 * torch.pi)
    rel_dist = torch.sum(rel_pos**2, dim=-1, keepdim=True)

    features = []
    if use_distance:
        features.append(rel_dist)
    if use_rel_pos:
        features.append(rel_pos)
    if use_rel_theta:
        features.append(rel_theta)
    edge_attr = (
        torch.cat(features, dim=-1)
        if features
        else torch.zeros(edge_index.shape[1], 0).to(device)
    )

    return edge_index, edge_attr


class ParticleDataset(Dataset):
    def __init__(self, data_pairs, transform=None, pre_transform=None):
        super(ParticleDataset, self).__init__(None, transform, pre_transform)
        self.data_pairs = data_pairs

    def len(self):
        return len(self.data_pairs)

    def get(self, idx):
        return self.data_pairs[idx]


class ExponentialDecayScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        alpha_start=1e-4,
        alpha_final=1e-6,
        decay_steps=5e6,
        last_epoch=-1,
    ):
        self.alpha_start = alpha_start
        self.alpha_final = alpha_final
        self.decay_steps = decay_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        factor = math.pow(0.1, step / self.decay_steps)
        lr = self.alpha_final + (self.alpha_start - self.alpha_final) * factor
        return [lr for _ in self.optimizer.param_groups]
