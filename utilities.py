import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse

from typing import Optional, List, Tuple

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

def prepare_graph_data(positions: torch.Tensor, 
                       device: Optional[str | torch.device] = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare graph data from particle positions.
    
    Parameters
    ----------
    positions: torch.Tensor
        Position tensor of shape (batch_size, 3, N) or (3, N).
    device: str, optional
        Either 'cpu', 'cuda' or torch.device instance, default is 'cpu'.
    
    Returns
    -------
    h: torch.Tensor
        Node features tensor
    edge_index: torch.Tensor
        Edge index tensor for fully connected graph
    edge_attr: torch.Tensor
        Edge features tensor
    """
    # Handle both batched and unbatched inputs
    if len(positions.shape) == 3:
        batch_size, dims, n_particles = positions.shape
        # Reshape to (batch_size * N, 3)
        h = positions.transpose(1, 2).reshape(-1, dims)
    else:
        dims, n_particles = positions.shape
        h = positions.T  # (N, 3)
        batch_size = 1
    
    # Create fully connected edge indices
    # First create adjacency matrix then convert to edge indices
    adj = torch.ones(n_particles, n_particles) - torch.eye(n_particles)
    edge_index, _ = dense_to_sparse(adj)
    edge_attr = torch.zeros(edge_index.shape[1], 0)

    h = h.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    
    return h, edge_index, edge_attr

def process_simulation_data(simulation_list: List, long: Optional[bool] = False) -> List:
    """
    Process multiple simulations for training.
    Returns list of (input, target) pairs instead of concatenated tensors.
    
    Parameters
    ----------
    simulation_list: List
        List of simulation arrays, each of shape (timesteps, 3, N) where N can vary between simulations.
    long: bool
        Set to True for long datasets, chooses some steps at random.
    
    Returns
    -------
    data_pair: List 
        [(x1, y1), (x2, y2), ...] where each x and y represents one timestep pair from any simulation.
    """
    data_pairs = []
    
    for sim in simulation_list:

        if not torch.is_tensor(sim):
            sim = torch.tensor(sim, dtype=torch.double)
        transformed_sim = to_periodic(sim) # Transform to periodic
        
        # Create pairs of consecutive timesteps
        if long:
            timesteps = torch.randint(0, 98, size=(4,))
        else:
            timesteps = range(len(sim)-1)
        for t in timesteps:
            x = transformed_sim[t]    # (6, N)
            y = transformed_sim[t+1]  # (6, N)
            data_pairs.append((x, y))
    
    return data_pairs

class ParticleDataset(Dataset):
    """
    Dataset for particle simulations with variable N.
    """
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        return self.data_pairs[idx]

def collate_fn(batch):
    """
    Collate function that doesn't try to stack the varying-size tensors
    """
    # Each element in batch is already a (x, y) pair
    return batch

class TorusMSELoss(nn.Module):
    def __init__(self, torus_dims, device):
        """
        Custom loss function for the torus.
        
        Parameters
        ----------
        torus_dims array-like
            The maximum values for each coordinate in the torus.
        """
        super(TorusMSELoss, self).__init__()
        self.torus_dims = torch.tensor(torus_dims, dtype=torch.double, device=device)

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        wrapped_diff = torch.minimum(diff, self.torus_dims - diff)
        loss = torch.mean(wrapped_diff.pow(2))
        return loss
