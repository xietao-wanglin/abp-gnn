import numpy as np
import torch
from torch import nn
from torch_geometric.data import Dataset, Data
from torch_cluster import radius_graph

from typing import Optional, List, Tuple

from src.simulation import Simulation, StiffSimulation

def apply_periodic_boundary(positions: torch.Tensor, dims: Optional[List] = None) -> torch.Tensor:
    """
    Applies periodic conditions in three dimensions.

    Parameters
    ----------
    positions: torch.Tensor
        Position Tensor.
    dims: List, optional
        The dimensions of the periodic box, default is None.

    Returns
    -------
    positions: torch.Tensor
        Position Tensor after applying boundary conditions.
    """
    if dims is None:
        dims = [1.0, 1.0, 2*torch.pi]
    pos = positions.clone()
    pos[0] = pos[0] % dims[0]
    pos[1] = pos[1] % dims[1]
    pos[2] = pos[2] % dims[2]
    return pos

def process_simulation_data(simulation_list: List, 
                            times_list: List,
                            angle: bool,
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
    assert len(simulation_list) == len(times_list), 'Must have equal amounts of simulations and times'
    
    for idx in range(len(simulation_list)):

        sim = simulation_list[idx]
        times = times_list[idx]

        if not torch.is_tensor(sim):
            sim = torch.tensor(sim, dtype=dtype, device=device)
        if not torch.is_tensor(times):
            times = torch.tensor(times, dtype=dtype, device=device)

        x = sim[0] # Features
        # Labels
        if angle:
            y = sim[1:]
        else: 
            y = sim[1:][:, :2, :]
        t = times[1:] # Collocation times

        edge_index, edge_attr = compute_graph(x, method=cluster_method, p=p, device=device)
        data = Data(
            x=x.T.to(device),
            y=y.permute(0, 2, 1).to(device),
            t=t.to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device)
        )
        data_pairs.append(data)
    
    return data_pairs

def compute_graph(x: torch.Tensor, 
                    method: str,
                    p: float | int, 
                    device: Optional[str | torch.device] = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute graph for a given set of node features.

    Parameters
    ----------
    x: torch.Tensor
        Postions (x, y, theta) of shape (3, N).
    method: str
        Either 'radius' for radial graph or 'knn' for knn graph.
    p: float or int
        Parameter of `method`.
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
        x_coords = xy[:, 0]
        y_coords = xy[:, 1]
        
        dx = x_coords.unsqueeze(1) - x_coords.unsqueeze(0)
        dy = y_coords.unsqueeze(1) - y_coords.unsqueeze(0)
        
        dx = dx - torch.round(dx)
        dy = dy - torch.round(dy)
        
        distances = torch.sqrt(dx.pow(2) + dy.pow(2))
        k = int(p)
        distances = distances.fill_diagonal_(float('inf')) # Avoid self-loops
        _, indices = torch.topk(distances, k=k, dim=1, largest=False)
        row_indices = torch.arange(xy.shape[0], device=device).repeat_interleave(k)
        col_indices = indices.flatten()
        edge_index = torch.stack([row_indices, col_indices])
    elif method == 'np_radius':
        edge_index = radius_graph(x, r=p)
    else:
        raise ValueError("Invalid method, must be either 'radius', 'knn' or 'np_radius'")
    
    edge_attr = torch.zeros(edge_index.shape[1], 0).to(device) # Empty edges
 
    return edge_index, edge_attr
    
class ParticleDataset(Dataset):
    """
    Dataset for particle simulations.
    
    Parameters
    ----------
    data_pairs: List
        List of (x, y, t, edge_index, edge_attr) samples.
    """
    def __init__(self, data_pairs, transform=None, pre_transform=None):
        super(ParticleDataset, self).__init__(None, transform, pre_transform)
        self.data_pairs = data_pairs
    
    def len(self):
        return len(self.data_pairs)
    
    def get(self, idx):
        return self.data_pairs[idx]

class RelativeL2Loss(nn.Module):
    def __init__(self, epsilon=1e-8, reduction='mean', weights=None):
        super(RelativeL2Loss, self).__init__()
        self.epsilon = epsilon
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction
        if weights is None:
            self.weights = 1
        else:
            self.weights = torch.tensor(weights)

    def forward(self, y_pred, y_true):
        numerator = (self.weights*y_pred - self.weights*y_true) ** 2
        denominator = (self.weights*y_true) ** 2 + self.epsilon
        rel_l2 = numerator / denominator

        if self.reduction == 'mean':
            return torch.mean(rel_l2)
        elif self.reduction == 'sum':
            return torch.sum(rel_l2)
        else:  # 'none'
            return rel_l2
