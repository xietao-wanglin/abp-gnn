import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, LayerNorm, GATConv
from torch_geometric.data import Data

class GNN_Layer(MessagePassing):
    def __init__(self, 
                 hidden_nf, 
                 edge_nf, 
                 activation=nn.SiLU(), 
                 dropout=0.0, 
                 norm=True):
        super(GNN_Layer, self).__init__(aggr='sum')
        self.hidden_nf = hidden_nf
        self.edge_nf = edge_nf
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = LayerNorm(hidden_nf) if norm else None
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf + edge_nf, hidden_nf),
            self.dropout,
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + hidden_nf, hidden_nf),
            self.dropout,
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation
        )
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        # x has shape [N, hidden_nf]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, edge_nf]
        
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, batch=batch)
    
    def message(self, x_i, x_j, edge_attr) -> torch.Tensor:
        # x_i has shape [E, hidden_nf]
        # x_j has shape [E, hidden_nf]
        # edge_attr has shape [E, edge_nf]
        
        edge_features = torch.cat([x_i, x_j, edge_attr], dim=-1)
        m_ij = self.edge_mlp(edge_features)
        return m_ij
    
    def update(self, aggr_out, x):
        # aggr_out has shape [N, hidden_nf]
        # x has shape [N, hidden_nf]
        if self.norm is not None:
            x_norm = self.norm(x)
        else:
            x_norm = x
        
        node_features = torch.cat([x_norm, aggr_out], dim=-1)
        x_new = self.node_mlp(node_features)
        
        # Apply skip connection
        x_new = x + x_new
            
        return x_new


class GNN(nn.Module):
    def __init__(self, 
                 n_layers, 
                 in_node_nf,
                 out_node_nf, 
                 in_edge_nf, 
                 hidden_nf,
                 activation=nn.SiLU(), 
                 device='cpu', 
                 dropout=0.0, 
                 norm=True):
        super(GNN, self).__init__()
        
        self.n_layers = n_layers
        self.encoder = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
        )
        self.layers = nn.ModuleList()

        for _ in range(self.n_layers):
            self.layers.append(
                GNN_Layer(hidden_nf, in_edge_nf, activation, dropout, norm)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, out_node_nf),
        )
        
        self.to(device)
    
    def forward(self, data, t):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        T = t.size(0)
        N = x.size(0)
        x_repeated = x.unsqueeze(0).repeat(T, 1, 1)
        t_expanded = t.unsqueeze(1).unsqueeze(2).expand(T, N, 1)
        x = torch.cat([x_repeated, torch.log(t_expanded)], dim=-1)
        x = x.view(T*N, -1)
        cumsum = torch.arange(0, T).to(x.device) * N
        num_edges = edge_index[0].shape[0]
        cumsum_edges = cumsum.repeat_interleave(num_edges, dim=0)
        edges_0 = edge_index[0].repeat(T) + cumsum_edges
        edges_1 = edge_index[1].repeat(T) + cumsum_edges
        edge_index = torch.stack([edges_0, edges_1], dim=0)
        edge_attr = edge_attr.repeat(T, 1)

        if batch is not None:
            batch_repeated = batch.unsqueeze(0).repeat(T, 1) # Shape: [T, N_total_nodes]
            shift_values = torch.arange(T, device=x.device).unsqueeze(1) * data.num_graphs # Shape: [T, 1]
            node_batch_indices = (batch_repeated + shift_values).view(-1) # Shape: [T * N_total_nodes]
        else:
            node_batch_indices = None
        x = self.encoder(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, node_batch_indices)
        
        x = self.decoder(x)
        x = x.view(T, N, -1) 
        return x

class GAT_Layer(nn.Module):
    def __init__(self,
                hidden_nf,
                edge_nf,
                heads=1,
                activation=nn.SiLU(),
                dropout=0.0,
                norm=True):
        super(GAT_Layer, self).__init__()
        self.hidden_nf = hidden_nf
        self.edge_nf = edge_nf
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = LayerNorm(hidden_nf) if norm else None
        
        # Edge features processing
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_nf, hidden_nf),
            self.dropout,
            activation
        )
        
        # GAT convolution
        self.conv = GATConv(
            hidden_nf, 
            hidden_nf, 
            heads=heads, 
            dropout=dropout,
            concat=False  # Average heads instead of concatenating
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + hidden_nf, hidden_nf),
            self.dropout,
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation
        )
        
    def forward(self, x, edge_index, edge_attr, batch=None):

        edge_features = self.edge_mlp(edge_attr)
        
        if self.norm is not None:
            x_norm = self.norm(x)
        else:
            x_norm = x
        gat_out = self.conv(x, edge_index, edge_attr=edge_features)
        gat_out = self.activation(gat_out)
        gat_out = self.dropout(gat_out)
        node_features = torch.cat([x_norm, gat_out], dim=-1)
        x_new = self.node_mlp(node_features)
        x_new = x + x_new
        return x_new

class GAT(nn.Module):
    def __init__(self,
                n_layers,
                in_node_nf,
                out_node_nf,
                in_edge_nf,
                hidden_nf,
                heads=4,
                activation=nn.SiLU(),
                device='cpu',
                dropout=0.0,
                norm=True):
        super(GAT, self).__init__()
        self.n_layers = n_layers
        
        self.encoder = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
        )
        
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                GAT_Layer(hidden_nf, in_edge_nf, heads, activation, dropout, norm)
            )
            
        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, out_node_nf),
        )
        
        self.to(device)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.encoder(x)
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)
            
        x = self.decoder(x)
        return x
    
class GraphEncoder(nn.Module):
    def __init__(self, 
                 n_layers, 
                 in_node_nf,
                 out_node_nf, 
                 in_edge_nf, 
                 hidden_nf,
                 activation=nn.SiLU(), 
                 device='cpu', 
                 dropout=0.0, 
                 norm=True):
        super(GraphEncoder, self).__init__()

        self.n_layers = n_layers
        self.encoder = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
        )
        self.layers = nn.ModuleList()

        for _ in range(self.n_layers):
            self.layers.append(
                GNN_Layer(hidden_nf, in_edge_nf, activation, dropout, norm)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, out_node_nf),
        )
        
        self.to(device)
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.encoder(x)
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)
            
        x = self.decoder(x)
        return x

class TimeEncoder(nn.Module):
    def __init__(self, 
                 n_layers, 
                 in_node_nf,
                 out_node_nf, 
                 in_edge_nf, 
                 hidden_nf,
                 activation=nn.SiLU(), 
                 device='cpu', 
                 dropout=0.0, 
                 norm=True):
        super(TimeEncoder, self).__init__()

        self.n_layers = n_layers
        self.encoder = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
        )
        self.layers = nn.ModuleList()

        for _ in range(self.n_layers):
            self.layers.append(
                GNN_Layer(hidden_nf, in_edge_nf, activation, dropout, norm)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, out_node_nf),
        )
        
        self.to(device)
    def forward(self, data, t):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        T = t.size(0)
        N = x.size(0)
        x_repeated = x.unsqueeze(0).repeat(T, 1, 1)
        t_expanded = t.unsqueeze(1).unsqueeze(2).expand(T, N, 1)
        x = torch.cat([x_repeated, torch.log(t_expanded)], dim=-1)
        x = x.view(T*N, -1)
        cumsum = torch.arange(0, T).to(x.device) * N
        num_edges = edge_index[0].shape[0]
        cumsum_edges = cumsum.repeat_interleave(num_edges, dim=0)
        edges_0 = edge_index[0].repeat(T) + cumsum_edges
        edges_1 = edge_index[1].repeat(T) + cumsum_edges
        edge_index = torch.stack([edges_0, edges_1], dim=0)
        edge_attr = edge_attr.repeat(T, 1)

        if batch is not None:
            batch_repeated = batch.unsqueeze(0).repeat(T, 1) # Shape: [T, N_total_nodes]
            shift_values = torch.arange(T, device=x.device).unsqueeze(1) * data.num_graphs # Shape: [T, 1]
            node_batch_indices = (batch_repeated + shift_values).view(-1) # Shape: [T * N_total_nodes]
        else:
            node_batch_indices = None
        x = self.encoder(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, node_batch_indices)
        
        x = self.decoder(x)
        x = x.view(T, N, -1) 
        return x


class LatentGNN(nn.Module):
    def __init__(self, 
                 n_layers, 
                 in_node_nf,
                 out_node_nf, 
                 latent_nf,
                 in_edge_nf, 
                 hidden_nf,
                 activation=nn.SiLU(), 
                 device='cpu', 
                 dropout=0.0, 
                 norm=True):
        super(LatentGNN, self).__init__()
        # latent_dim => x_dim
        
        self.n_layers = n_layers
        self.ic_encoder = GraphEncoder(n_layers, in_node_nf, latent_nf, in_edge_nf, hidden_nf, activation, device, dropout, norm)

        self.vel_encoder = GraphEncoder(n_layers, in_node_nf, latent_nf, in_edge_nf, hidden_nf, activation, device, dropout, norm)
        
        self.time_encoder = TimeEncoder(n_layers, 1+in_node_nf, latent_nf, in_edge_nf, hidden_nf, activation, device, dropout, norm)
        
        self.decoder = GraphEncoder(n_layers, 
                                    latent_nf, # Input feature dimension for the decoder GNN
                                    out_node_nf, # Output feature dimension for the decoder GNN
                                    in_edge_nf, 
                                    hidden_nf, 
                                    activation, 
                                    device, 
                                    dropout, 
                                    norm)
        
        self.to(device)
    
    def forward(self, data, t):
        original_edge_index, original_edge_attr = data.edge_index, data.edge_attr
        original_batch = data.batch
        original_num_graphs = getattr(data, 'num_graphs', 1) # Default to 1 if not explicitly given (e.g., single graph input)
        e = self.ic_encoder(data) # (x_dim,) -> (latent_dim,)
        c = self.vel_encoder(data) # (x_dim,) -> (latent_dim,) 
        tau = self.time_encoder(data, t) # (x_dim, T) -> (latent_dim, T)

        y = e.unsqueeze(0) + tau * c.unsqueeze(0)  # (latent_dim,) + (latent_dim, T)*(latent_dim,) -> (latent_dim, T)

        T = t.size(0) # Number of time steps
        N = data.x.size(0) # Total number of nodes in the original input batch (sum of nodes across all graphs)

        # Flatten y for GNN input: each node feature at each time step becomes a "node" in a larger graph batch
        y_flat = y.view(T * N, -1) # Shape: (T * N, latent_nf)

        # Adjust edge_index for the flattened batch of (T * original_N) nodes
        # This replicates the graph structure for each time step and shifts node indices
        num_original_edges = original_edge_index[0].shape[0]
        cumsum = torch.arange(0, T, device=y.device) * N # Shift for node indices
        cumsum_edges = cumsum.repeat_interleave(num_original_edges, dim=0) # Apply shift to each edge

        edge_index_decoder = torch.stack([
            original_edge_index[0].repeat(T) + cumsum_edges,
            original_edge_index[1].repeat(T) + cumsum_edges
        ], dim=0)
        
        # Edge attributes are also repeated for each time step's graph copy
        edge_attr_decoder = original_edge_attr.repeat(T, 1)

        # Adjust batch vector for the flattened batch
        batch_decoder = None
        if original_batch is not None:
            batch_repeated = original_batch.unsqueeze(0).repeat(T, 1) # (T, N_total_original_batch)
            shift_values = torch.arange(T, device=y.device).unsqueeze(1) * original_num_graphs # (T, 1)
            batch_decoder = (batch_repeated + shift_values).view(-1) # (T * N_total_original_batch)
        else:
            # If original data had no batch, assume it was a single graph.
            # Now we have T copies of that graph, so create a batch vector assigning each node to its graph copy.
            batch_decoder = torch.arange(T, device=y.device).repeat_interleave(N) # (T*N)

        # Create a temporary PyTorch Geometric Data object for the GNN decoder
        # This allows the GraphEncoder to operate on the expanded data
        num_graphs_decoder = T * original_num_graphs # Total number of graphs in the decoder's batch
        
        temp_data_for_decoder = Data(
            x=y_flat, 
            edge_index=edge_index_decoder, 
            edge_attr=edge_attr_decoder, 
            batch=batch_decoder,
            num_graphs=num_graphs_decoder 
        )

        # Pass through the GNN decoder
        decoded_y_flat = self.decoder(temp_data_for_decoder) # Shape: (T * N, out_node_nf)

        # Reshape the output back to (T, N, out_node_nf)
        x = decoded_y_flat.view(T, N, -1)  # (latent_dim, T) -> (x_dim, T)

        return x
