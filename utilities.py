import torch
from torch_geometric.utils import dense_to_sparse

def prepare_graph_data(positions, device='cpu'):
    """
    Prepare graph data from particle positions.
    
    Args:
        positions: tensor of shape (batch_size, 3, N) or (3, N)
        device: torch device
    
    Returns:
        h: Node features tensor
        edge_index: Edge index tensor for fully connected graph
        edge_attr: Edge features tensor (empty for now)
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
    
    # Move to device
    edge_index = edge_index.to(device)
    h = h.to(device)
    
    # For now, we'll use empty edge features
    edge_attr = torch.zeros(edge_index.shape[1], 0).to(device)
    
    return h, edge_index, edge_attr

def process_simulation_data(simulation_list):
    """
    Process multiple simulations for GNN training.
    Returns list of (input, target) pairs instead of concatenated tensors.
    
    Args:
        simulation_list: List of simulation arrays, each of shape (timesteps, 3, N)
                        where N can vary between simulations
    
    Returns:
        list of tuples: [(x1, y1), (x2, y2), ...] where each x and y 
                       represents one timestep pair from any simulation
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

class ParticleDataset(torch.utils.data.Dataset):
    """
    Custom dataset for particle simulations with variable N
    """
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        return self.data_pairs[idx]

def collate_fn(batch):
    """
    Custom collate function that doesn't try to stack the varying-size tensors
    """
    # Each element in batch is already a (x, y) pair
    return batch