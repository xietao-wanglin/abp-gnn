import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_cluster import knn_graph, radius_graph

from typing import Optional, List, Tuple
class TorusMSELoss(nn.Module):
    """
    Custom loss function for the torus.
    
    Parameters
    ----------
    torus_dims: array-like
        The maximum values for each coordinate in the torus.
    device: str, optional
        Either 'cpu', 'cuda' or torch.device instance, default is 'cpu'.
    """
    def __init__(self, torus_dims, 
                 device: Optional[str | torch.device] = 'cpu'):
        super(TorusMSELoss, self).__init__()
        self.torus_dims = torch.tensor(torus_dims, dtype=torch.double, device=device)

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        wrapped_diff = torch.minimum(diff, self.torus_dims - diff)
        loss = torch.mean(wrapped_diff.pow(2))
        return loss

class NormalizeOutput(nn.Module):
    """Normalize pairs of outputs to unit length"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to (3, 2, N) to handle pairs
        x = x.transpose(1, 0)
        x = x.reshape(3, 2, x.shape[-1])
        # Normalize each pair
        norms = torch.norm(x, dim=-2, keepdim=True)
        x = x / norms
        x = x.reshape(6, x.shape[-1])
        x = x.transpose(1, 0)
        return x

def apply_periodic_boundary(positions):
    pos = positions.clone()
    pos[0] = pos[0] % 1.0
    pos[1] = pos[1] % 1.0
    pos[2] = pos[2] % (2*torch.pi)
    return pos

def to_periodic(coords: torch.Tensor) -> torch.Tensor:
    """
    Transform coordinates from [x, y, theta] to periodic representation

    Parameters
    ----------
    coords: torch.Tensor
        Input coordinates of shape (..., 3, N) containing [x, y, theta]
        (x, y) expected in [0, 1]

    Returns
    -------
    periodic: torch.Tensor
        Transformed coordinates of shape (..., 4, N)
    """
    periodic = torch.empty((*coords.shape[:-2], 5, coords.shape[-1]), 
                        dtype=coords.dtype, device=coords.device)

    periodic[..., 0, :] = torch.sin(2 * torch.pi * coords[..., 0, :])
    periodic[..., 1, :] = torch.cos(2 * torch.pi * coords[..., 0, :])

    periodic[..., 2, :] = torch.sin(2 * torch.pi * coords[..., 1, :])
    periodic[..., 3, :] = torch.cos(2 * torch.pi * coords[..., 1, :])

    periodic[..., 4, :] = coords[..., 2, :]

    return periodic

def from_periodic(periodic: torch.Tensor) -> torch.Tensor:
    """
    Transform coordinates from periodic representation back to [x,y,theta]

    Parameters
    ----------
    periodic: torch.Tensor
        Input periodic coordinates of shape (..., 6, N)

    Returns
    -------
    coords: torch.Tensor
        Original coordinates of shape (..., 3, N) 
        with (x, y) in [0, 1] and theta in [0, 2pi]
    """
    coords = torch.empty((*periodic.shape[:-2], 2, periodic.shape[-1]), 
                        dtype=periodic.dtype, device=periodic.device)

    # Recover x coordinate
    coords[..., 0, :] = torch.atan2(periodic[..., 0, :], 
                                    periodic[..., 1, :]) / (2 * torch.pi)
    coords[..., 0, :] = torch.where(coords[..., 0, :] < 0, 
                                    coords[..., 0, :] + 1, 
                                    coords[..., 0, :])

    # Recover y coordinate
    coords[..., 1, :] = torch.atan2(periodic[..., 2, :], 
                                    periodic[..., 3, :]) / (2 * torch.pi)
    coords[..., 1, :] = torch.where(coords[..., 1, :] < 0, 
                                    coords[..., 1, :] + 1, 
                                    coords[..., 1, :])

    return coords

def process_simulation_data(simulation_list: List, 
                            subset: Optional[bool] = False,
                            n_samples: Optional[int] = 4,
                            cluster_method: Optional[str] = 'radius',
                            p: Optional[int] = 4, 
                            dtype: Optional[torch.dtype] = torch.float,
                            device: Optional[str | torch.device] = 'cpu') -> List:
    """
    Process multiple simulations for training.
    Returns list of (input, target) pairs instead of concatenated tensors.
    
    Parameters
    ----------
    simulation_list: List
        List of simulation arrays, each of shape (timesteps, 3, N) where N can vary between simulations.
    subset: bool
        If True, selects `n_samples` random timesteps instead of all.
    num_samples : int
        Number of random samples to pick when `subset=True`.
    k: int
        Number of nearest neighbors to connect each node to.
    dtype : torch.dtype
        Data type for conversion (default: torch.float).
    device: str, optional
        Either 'cpu', 'cuda' or torch.device instance, default is 'cpu'.
    
    Returns
    -------
    data_pairs: List 
        [(x1, y1, edge_index1, edge_attr1), (x2, y2, edge_index2, edge_attr2), ...]
        where each x and y represents one timestep pair from any simulation.
    """
    data_pairs = []
    
    for sim in simulation_list:

        if not torch.is_tensor(sim):
            sim = torch.tensor(sim, dtype=dtype)
        transformed_sim = to_periodic(sim) # Transform to periodic

        num_timesteps = transformed_sim.shape[0]-1
        
        # Choose steps pairs
        if subset:
            timesteps = torch.randint(0, num_timesteps, size=(min(n_samples, num_timesteps),))
        else:
            timesteps = range(num_timesteps)

        for t in timesteps:
            x = sim[t]    # (3, N)
            y = sim[t+1]  # (3, N)

            x = apply_periodic_boundary(x) # Ensure [0, 1] x [0, 1] x [0, 2pi]
            
            x[2] = x[2] / (2*torch.pi)
            y[2] = y[2] / (2*torch.pi)

            x_periodic = transformed_sim[t] # (6, N)
            edge_index, edge_attr = compute_graph(x_periodic, method=cluster_method, p=p, device=device)

            data_pairs.append((x, y, (y-x), edge_index, edge_attr))
    
    return data_pairs

def compute_graph(x: torch.Tensor, 
                      method: str,
                      p: Optional[float | int] = 4, 
                      device: Optional[str | torch.device] = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute k-NN graph for a given set of node features.

    Parameters
    ----------
    x: torch.Tensor
        Node features of shape (6, N).
    k: int, optional
        Number of nearest neighbors.
    device: str, optional
        Either 'cpu', 'cuda' or torch.device instance, default is 'cpu'.

    Returns
    -------
    edge_index : torch.Tensor
        Tensor of shape (2, E) containing source and target node indices.
    edge_attr : torch.Tensor
        Edge features (L^2 distances), shape (E, 1).
    """

    xy, theta = x[:-1], x[-1]
    xy = xy.transpose(0, 1)
    if method == 'radius':
        edge_index = radius_graph(xy, r=p).to(device)
    elif method == 'knn':
        edge_index = knn_graph(xy, k=p).to(device)  # Shape (2, E)
    else:
        raise ValueError('Invalid method')

    # Compute L^2 distances
    row, col = edge_index
    distances = torch.norm(xy[row] - xy[col], dim=1, keepdim=True)  # Shape (E, 1)

    # Compute angle differences
    angle_diff = theta[col] - theta[row]  # Shape (E,)

    # Compute cosine and sine of angle differences
    cos_diff = torch.cos(2*torch.pi*angle_diff).unsqueeze(1)  # Shape (E, 1)
    sin_diff = torch.sin(2*torch.pi*angle_diff).unsqueeze(1)  # Shape (E, 1)

    # Concatenate all features
    edge_attr = torch.cat([distances, cos_diff, sin_diff], dim=1)  # Shape (E, 3)

    return edge_index, edge_attr

def collate_fn(batch):
    """
    Collate function that doesn't try to stack the varying-size tensors
    """
    # Each element in batch is already a (x, y) pair
    return batch

class ParticleDataset(Dataset):
    """
    Dataset for particle simulations with variable N.

    Parameters
    ----------
    data_pairs : List
        List of (x, y, edge_index, edge_attr) samples.
    """
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        x, y, res, edge_index, edge_attr = self.data_pairs[idx]
        return x, y, res, edge_index, edge_attr
