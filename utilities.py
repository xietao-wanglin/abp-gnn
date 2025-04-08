import torch
from torch import nn
from torch.utils.data import Dataset
from torch_cluster import knn_graph

from typing import Optional, List, Tuple

from simulation import Simulation, StiffSimulation

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
            timesteps = torch.randint(1, num_timesteps, size=(min(n_samples, num_timesteps),))
        else:
            timesteps = range(1, num_timesteps) # Avoid step 1 as it is usually complete non-sense

        for t in timesteps:
            x = sim[t]
            y = sim[t+1]

            x_bounded = apply_periodic_boundary(x) # Ensure [0, 1] x [0, 1] x [0, 2pi]

            N = x[0].shape[0]

            edge_index, edge_attr = compute_graph(x_bounded, method=cluster_method, p=p, device=device)

            #simulation = StiffSimulation(N=N, v0=0.1, L_box=1.0, delta_t=0.1, rot_couple=0, sigma=0.025, rot_rate=1,)

            #derivatives = simulation.particle_system(t, x_bounded.numpy().T.reshape(N*3)).reshape(N, 3).T

            data_pairs.append((x_bounded, t, y, edge_index, edge_attr))
    
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
        edge_index = knn_graph(xy, k=p).to(device)
    else:
        raise ValueError("Invalid method, must be either 'radius' or 'knn'")

    row, col = edge_index
    angle_diff = theta[row] - theta[col] 
    sin_diff = torch.sin(angle_diff).unsqueeze(1)
    #rel_pos_raw = xy[row] - xy[col]
    #rel_pos = rel_pos_raw - torch.round(rel_pos_raw)
    #rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True)
    #rel_encoding = torch.cat([rel_pos, rel_dist], dim=-1)
    #edge_attr = torch.cat([sin_diff, rel_encoding], dim=-1) # Relative encoding
    #edge_attr = sin_diff # Only use phase difference
    edge_attr = torch.zeros(edge_index.shape[1], 0).to(device) # Empty edges
 
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

class RelativeL2Loss(nn.Module):
    def __init__(self, epsilon=1e-8, reduction='mean'):
        super(RelativeL2Loss, self).__init__()
        self.epsilon = epsilon
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        numerator = torch.sum((y_pred - y_true) ** 2, dim=1)
        denominator = torch.sum(y_true ** 2, dim=1) + self.epsilon
        rel_l2 = numerator / denominator

        if self.reduction == 'mean':
            return torch.mean(rel_l2)
        elif self.reduction == 'sum':
            return torch.sum(rel_l2)
        else:  # 'none'
            return rel_l2