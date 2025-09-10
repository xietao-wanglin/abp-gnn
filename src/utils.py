from src.simulation import BoundaryType
import torch
from torch_geometric.data import Dataset, Data

from typing import Optional, List, Tuple, Dict


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
    simulation_list: List,
    particle_type_list: Optional[List] = None,
    subset: Optional[bool] = False,
    subset_samples: Optional[List] = None,
    n_samples: Optional[int] = 4,
    cluster_method: Optional[str] = "radius",
    p: Optional[int] = 0.1,
    use_distance: Optional[bool] = False,
    use_rel_pos: Optional[bool] = False,
    target_vel: Optional[bool] = True,
    use_pos: Optional[bool] = False,
    boundary_type: Optional[Tuple] = None,
    stats: Optional[Dict] = None,
    dtype: Optional[torch.dtype] = torch.float,
    device: Optional[str | torch.device] = "cpu",
) -> List:
    """
    Process multiple simulations for training.
    Returns list of (input, target) pairs.

    Parameters
    ----------
    simulation_list: List
        List of simulation arrays, each of shape (timesteps, 3, N).
    subset: bool, optional
        If True, selects `n_samples` random timesteps instead of all.
    subset_samples: List, optional
        If provided, samples to choose from trajectories if `subset=True`, otherwise random, default is None.
    n_samples: int, optional
        Number of random samples to pick when `subset=True`.
    cluster_method: str, optional
        Method used to create edges in graph, either 'radius' or 'knn', default is 'radius'.
    p: float or int, optional
        Parameter of `cluster_method`, default is 0.1.
    dtype: torch.dtype
        Data type for conversion, default is torch.float.
    device: str or torch.device, optional
        Either 'cpu', 'cuda' or torch.device instance, default is 'cpu'.

    Returns
    -------
    data_pairs: List
    """
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
            if not torch.is_tensor(particle_type):
                particle_type = torch.tensor(
                    particle_type, dtype=torch.int, device=device
                )
        else:
            particle_type = None

        if not torch.is_tensor(sim):
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

            x_bounded = apply_periodic_boundary(x, wrap_dims=wrap_dims)

            edge_index, edge_attr = compute_graph(
                x_bounded,
                method=cluster_method,
                p=p,
                use_distance=use_distance,
                use_rel_pos=use_rel_pos,
                boundary_type=boundary_type,
                device=device,
            )
            if target_vel:
                if stats is not None:
                    vel = y - x_bounded
                    vel[:2] = (vel[:2] - stats["vel_mean"]) / stats["vel_std"]
                    label = vel[:2].T.to(device).to(dtype=dtype)
                else:
                    label = (y - x_bounded).T.to(device).to(dtype=dtype)
            else:
                label = y[:2].T

            if use_pos:
                data_input = x_bounded[:2].T.to(device).to(dtype=dtype)
            else:
                data_input = x_bounded[2].unsqueeze(0).T.to(device).to(dtype=dtype)

            data = Data(
                x=data_input,
                y=label,
                edge_index=edge_index.to(device),
                edge_attr=edge_attr.to(device).to(dtype=dtype),
                pos=x_bounded[:2].T.to(device).to(dtype=dtype),
                trajectory=apply_periodic_boundary(sim[t + 1 : t + 21]).permute(
                    2, 0, 1
                ),
                full_x=x_bounded.T.to(device).to(dtype=dtype),
                particle_type=particle_type,
            )

            data_pairs.append(data)

    return data_pairs


def continuous_simulation(
    simulation_list: List,
    times_list: List,
    angle: bool,
    cluster_method: Optional[str] = "radius",
    p: Optional[int] = 0.1,
    use_distance: Optional[bool] = False,
    dtype: Optional[torch.dtype] = torch.float,
    device: Optional[str | torch.device] = "cpu",
) -> List:
    """
    Process multiple simulations for training.
    Returns list of (input, target) pairs instead of concatenated tensors.

    Parameters
    ----------
    simulation_list: List
        List of simulation arrays, each of shape (timesteps, 3, N) where N can vary between simulations.
    n_samples: int, optional
        Number of random samples to pick when `subset=True`.
    cluster_method: str, optional
        Method used to create edges in graph, either 'radius' or 'knn', default is 'radius'.
    p: float or int, optional
        Parameter of `cluster_method`, default is 0.1.
    dtype : torch.dtype
        Data type for conversion, default is torch.float.
    device: str or torch.device, optional
        Either 'cpu', 'cuda' or torch.device instance, default is 'cpu'.

    Returns
    -------
    data_pairs: List
        [(x1, y1, edge_index1, edge_attr1), (x2, y2, edge_index2, edge_attr2), ...]
        where each x and y represents one timestep pair from any simulation.
    """
    data_pairs = []
    assert len(simulation_list) == len(times_list), (
        "Must have equal amounts of simulations and times"
    )

    for idx in range(len(simulation_list)):
        sim = simulation_list[idx]
        times = times_list[idx]

        if not torch.is_tensor(sim):
            sim = torch.tensor(sim, dtype=dtype, device=device)
        if not torch.is_tensor(times):
            times = torch.tensor(times, dtype=dtype, device=device)

        x = sim[0]  # Features
        # Labels
        if angle:
            y = sim[1:]
        else:
            y = sim[1:][:, :2, :]
        t = times[1:]  # Collocation times

        edge_index, edge_attr = compute_graph(
            x, method=cluster_method, p=p, device=device, use_distance=use_distance
        )
        data = Data(
            x=x.T.to(device),
            y=y.permute(0, 2, 1).to(device),
            t=t.to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device),
        )
        data_pairs.append(data)

    return data_pairs


def compute_graph(
    x: torch.Tensor,
    method: str,
    p: float | int,
    use_distance: Optional[bool] = False,
    use_rel_pos: Optional[bool] = False,
    box_length: Optional[float] = 1,
    boundary_type: Optional[Tuple] = None,
    device: Optional[str | torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute graph for a given set of node features.

    Parameters
    ----------
    x: torch.Tensor
        Postions (x, y, theta) of shape (3, N).
    method: str
        Either 'radius' for radial graph or 'knn' for knn graph.
    p: float or int, optional
        Parameter of 'method'.
    device: str or torch.device, optional
        Either 'cpu', 'cuda' or torch.device instance, default is 'cpu'.

    Returns
    -------
    edge_index: torch.Tensor
        Tensor of shape (2, E) containing source and target node indices.
    edge_attr: torch.Tensor
        Edge features, shape (E, 1).
    """
    if boundary_type is None:
        boundary_type = (
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
            BoundaryType.PERIODIC,
        )

    xy, _theta = x[:-1], x[-1]
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
    if boundary_type[0] == BoundaryType.PERIODIC:
        rel_pos[:, 0] = rel_pos[:, 0] - box_length * torch.round(
            rel_pos[:, 0] / box_length
        )
    if boundary_type[1] == BoundaryType.PERIODIC:
        rel_pos[:, 1] = rel_pos[:, 1] - box_length * torch.round(
            rel_pos[:, 1] / box_length
        )
    rel_dist = torch.sum(rel_pos**2, dim=-1, keepdim=True)

    features = []
    if use_distance:
        features.append(rel_dist)
    if use_rel_pos:
        features.append(rel_pos)
    edge_attr = (
        torch.cat(features, dim=-1)
        if features
        else torch.zeros(edge_index.shape[1], 0).to(device)
    )

    return edge_index, edge_attr


class ParticleDataset(Dataset):
    """
    Dataset for particle simulations.

    Parameters
    ----------
    data_pairs: List
        List of PyG Data objects.
    """

    def __init__(self, data_pairs, transform=None, pre_transform=None):
        super(ParticleDataset, self).__init__(None, transform, pre_transform)
        self.data_pairs = data_pairs

    def len(self):
        return len(self.data_pairs)

    def get(self, idx):
        return self.data_pairs[idx]
