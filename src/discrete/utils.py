import torch
from torch import nn
import torch
from torch_geometric.data import Dataset, Data
from torch_cluster import radius_graph, knn_graph

from typing import Optional, List, Tuple

from src.simulation import Simulation, StiffSimulation

def apply_periodic_boundary(positions: torch.Tensor, dims: Optional[List] = None) -> torch.Tensor:
    """
    Applies periodic conditions in three dimensions.

    Parameters
    ----------
    positions: torch.Tensor
        Position Tensor.
    dims: list, optional
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
                            subset: Optional[bool] = False,
                            subset_samples: Optional[List] = None,
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
    subset_samples: List, optional
        If provided, samples to choose from trajectories if `subset=True`, otherwise random, default is None.
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
    
    for sim in simulation_list:

        if not torch.is_tensor(sim):
            sim = torch.tensor(sim, dtype=dtype)

        num_timesteps = sim.shape[0]-1
        
        # Choose steps pairs
        if subset:
            if subset_samples is None:
                timesteps = torch.randint(0, num_timesteps, size=(min(n_samples, num_timesteps),))
            else:
                timesteps = subset_samples
        else:
            timesteps = range(0, num_timesteps)

        for t in timesteps:
            x = sim[t]
            y = sim[t+1]

            x_bounded = apply_periodic_boundary(x) # Ensure [0, 1] x [0, 1] x [0, 2pi]

            N = x[0].shape[0]

            edge_index, edge_attr = compute_graph(x_bounded, method=cluster_method, p=p, device=device)

            simulation = StiffSimulation(N=N, v0=0.1, L_box=1.0, delta_t=0.1, rot_couple=0, sigma=0.025, rot_rate=1,)

            derivatives = torch.tensor(simulation.particle_system(t, y.numpy().T.reshape(N*3)).reshape(N, 3).T, dtype=dtype)

            data_pairs.append((x_bounded.T, derivatives.T, edge_index, edge_attr))
    
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
    elif method == 'np_knn':
        edge_index = knn_graph(x, k=p)
    else:
        raise ValueError("Invalid method, must be either 'radius', 'knn', 'np_radius' or 'np_knn'.")

    edge_attr = torch.zeros(edge_index.shape[1], 0).to(device) # Empty edges
 
    return edge_index, edge_attr
    
class ParticleDataset(Dataset):
    """
    Dataset for particle simulations.
    
    Parameters
    ----------
    data_pairs: List
        List of (x, y, edge_index, edge_attr) samples.
    """
    def __init__(self, data_pairs, transform=None, pre_transform=None):
        super(ParticleDataset, self).__init__(None, transform, pre_transform)
        self.data_pairs = data_pairs
    
    def len(self):
        return len(self.data_pairs)
    
    def get(self, idx):
        x, y, edge_index, edge_attr = self.data_pairs[idx]
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y 
        )
        
        return data

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
        numerator = torch.sum((self.weights*y_pred - self.weights*y_true) ** 2, dim=-1)
        denominator = torch.sum((self.weights*y_true) ** 2, dim=-1) + self.epsilon
        rel_l2 = numerator / denominator

        if self.reduction == 'mean':
            return torch.mean(rel_l2)
        elif self.reduction == 'sum':
            return torch.sum(rel_l2)
        else:  # 'none'
            return rel_l2