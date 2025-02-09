import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_cluster import knn_graph

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


def to_periodic(coords: torch.Tensor) -> torch.Tensor:
    """
    Transform coordinates from [x, y, theta] to periodic representation

    Parameters
    ----------
    coords: torch.Tensor
        Input coordinates of shape (..., 3, N) containing [x, y, theta]
        (x, y) expected in [0, 1], theta in [0, 2pi]

    Returns
    -------
    periodic: torch.Tensor
        Transformed coordinates of shape (..., 6, N)
    """
    periodic = torch.empty((*coords.shape[:-2], 6, coords.shape[-1]), 
                        dtype=coords.dtype, device=coords.device)

    periodic[..., 0, :] = torch.sin(2 * torch.pi * coords[..., 0, :])
    periodic[..., 1, :] = torch.cos(2 * torch.pi * coords[..., 0, :])

    periodic[..., 2, :] = torch.sin(2 * torch.pi * coords[..., 1, :])
    periodic[..., 3, :] = torch.cos(2 * torch.pi * coords[..., 1, :])

    periodic[..., 4, :] = torch.sin(coords[..., 2, :])
    periodic[..., 5, :] = torch.cos(coords[..., 2, :])

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
    coords = torch.empty((*periodic.shape[:-2], 3, periodic.shape[-1]), 
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

    # Recover theta coordinate
    coords[..., 2, :] = torch.atan2(periodic[..., 4, :], 
                                    periodic[..., 5, :])
    coords[..., 2, :] = torch.where(coords[..., 2, :] < 0, 
                                    coords[..., 2, :] + 2*torch.pi, 
                                    coords[..., 2, :])

    return coords

def process_simulation_data(simulation_list: List, 
                            subset: Optional[bool] = False,
                            n_samples: Optional[int] = 4,
                            k: Optional[int] = 4, 
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
            x = transformed_sim[t]    # (6, N)
            y = transformed_sim[t+1]  # (6, N)
            edge_index, edge_attr = compute_knn_graph(x, k=k, device=device)

            data_pairs.append((x, y, 10*(y-x), edge_index, edge_attr))
    
    return data_pairs

def compute_knn_graph(x: torch.Tensor, 
                      k: Optional[int] = 4, 
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

    x = x.T
    edge_index = knn_graph(x, k=k).to(device)  # Shape (2, E)

    # Compute L^2 distances
    row, col = edge_index
    edge_attr = torch.norm(x[row] - x[col], dim=1, keepdim=True)  # Shape (E, 1)

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
