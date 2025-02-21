import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_cluster import knn_graph

from typing import Optional, List, Tuple

def apply_periodic_boundary(positions):
    pos = positions.clone()
    pos[0] = pos[0] % 1.0
    pos[1] = pos[1] % 1.0
    pos[2] = pos[2] % (2*torch.pi)
    return pos

def process_simulation_data(simulation_list: List, 
                            subset: Optional[bool] = False,
                            n_samples: Optional[int] = 4,
                            cluster_method: Optional[str] = 'radius',
                            p: Optional[int] = 0.1, 
                            dtype: Optional[torch.dtype] = torch.float,
                            device: Optional[str | torch.device] = 'cpu') -> List:
    """
    Process multiple simulations for training.
    Returns list of (input, target) pairs instead of concatenated tensors.
    
    Parameters
    ----------
    simulation_list: List
        List of simulation arrays, each of shape (timesteps, 3, N) where N can vary between simulations.
    subset: bool, optional
        If True, selects `n_samples` random timesteps instead of all.
    n_samples: int, optional
        Number of random samples to pick when `subset=True`.
    cluster_method: str, optional
        Method used to create edges in graph, either 'radius' or 'knn'.
    p: float or int, optional
        Parameter of `cluster_method`.
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
    
    for sim in simulation_list:

        if not torch.is_tensor(sim):
            sim = torch.tensor(sim, dtype=dtype)

        num_timesteps = sim.shape[0]-1
        
        # Choose steps pairs
        if subset:
            timesteps = torch.randint(0, num_timesteps, size=(min(n_samples, num_timesteps),))
        else:
            timesteps = range(num_timesteps)

        for t in timesteps:
            x = sim[t]
            y = sim[t+1]

            x = apply_periodic_boundary(x) # Ensure [0, 1] x [0, 1] x [0, 2pi]

            edge_index, edge_attr = compute_graph(x, method=cluster_method, p=p, device=device)

            data_pairs.append((x, t, (y-x)/0.1, edge_index, edge_attr))
    
    return data_pairs

def compute_graph(x: torch.Tensor, 
                      method: str,
                      p: Optional[float | int] = 4, 
                      device: Optional[str | torch.device] = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute graph for a given set of node features.

    Parameters
    ----------
    x: torch.Tensor
        Node features of shape (3, N).
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

    xy, theta = x[:-1], x[-1]
    xy = xy.transpose(0, 1)
    if method == 'radius':
        x_coords = xy[:, 0]
        y_coords = xy[:, 1]
        
        dx = x_coords.unsqueeze(1) - x_coords.unsqueeze(0)
        dy = y_coords.unsqueeze(1) - y_coords.unsqueeze(0)
        
        dx = dx - torch.round(dx)
        dy = dy - torch.round(dy)
        
        distances = torch.sqrt(dx.pow(2) + dy.pow(2))
        edges = torch.where(distances < p)
        mask = edges[0] != edges[1]
        edge_index = torch.stack([edges[0][mask], edges[1][mask]]) # Remove self-loops
    elif method == 'knn':
        edge_index = knn_graph(xy, k=p).to(device)
    else:
        raise ValueError("Invalid method, must be either 'radius' or 'knn'")

    row, col = edge_index
    angle_diff = theta[row] - theta[col] 
    sin_diff = torch.sin(angle_diff).unsqueeze(1)
    edge_attr = sin_diff

    return edge_index, edge_attr

def collate_fn(batch):
    """
    Collate function that doesn't try to stack the varying-size tensors
    """
    return batch

class ParticleDataset(Dataset):
    """
    Dataset for particle simulations with variable N.

    Parameters
    ----------
    data_pairs: List
        List of (x, y, edge_index, edge_attr) samples.
    """
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        x, y, res, edge_index, edge_attr = self.data_pairs[idx]
        return x, y, res, edge_index, edge_attr
