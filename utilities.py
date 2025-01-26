import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse

from typing import Optional, List, Tuple

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
    
    # Repeat edge indices for each batch if necessary
    if batch_size > 1:
        edge_index_batched = []
        for i in range(batch_size):
            edge_index_batch = edge_index + (i * n_particles)
            edge_index_batched.append(edge_index_batch)
        edge_index = torch.cat(edge_index_batched, dim=1)

    xy_coords = positions[:2, :].T  # (N, 2)

    row, col = edge_index
    diff = xy_coords[row] - xy_coords[col]  # (num_edges, 2)
    wrapped_diff = torch.minimum(torch.abs(diff), 1.0 - torch.abs(diff))  # Wrap distances on the torus
    edge_attr = torch.norm(wrapped_diff, dim=1).unsqueeze(1)  # Compute the toroidal L^2 norm

    h = h.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    
    return h, edge_index, edge_attr

def process_simulation_data(simulation_list: List) -> List:
    """
    Process multiple simulations for training.
    Returns list of (input, target) pairs instead of concatenated tensors.
    
    Parameters
    ----------
    simulation_list: List
        List of simulation arrays, each of shape (timesteps, 3, N) where N can vary between simulations.
    
    Returns
    -------
    data_pair: List 
        [(x1, y1), (x2, y2), ...] where each x and y represents one timestep pair from any simulation.
    """
    data_pairs = []
    
    for sim in simulation_list:
        # Convert to torch tensor if not already
        if not torch.is_tensor(sim):
            sim = torch.tensor(sim, dtype=torch.double)
        
        # Create pairs of consecutive timesteps
        for t in range(len(sim)-1):
            x = sim[t]    # (3, N)
            y = sim[t+1]  # (3, N)
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
    def __init__(self, torus_dims):
        """
        Custom loss function for the torus.
        
        Parameters
        ----------
        torus_dims array-like
            The maximum values for each coordinate in the torus.
        """
        super(TorusMSELoss, self).__init__()
        self.torus_dims = torch.tensor(torus_dims, dtype=torch.double)

    def forward(self, pred, target):
        assert pred.shape == target.shape, "Pred and target tensors must have the same shape"

        # Compute the difference modulo the torus dimensions
        diff = torch.abs(pred - target)
        wrapped_diff = torch.minimum(diff, self.torus_dims - diff)

        # Compute the MSE loss with the wrapped differences
        loss = torch.mean(wrapped_diff.pow(2))
        return loss
