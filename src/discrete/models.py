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
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
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
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_nf, hidden_nf),
            self.dropout,
            activation
        )
        
        self.conv = GATConv(
            hidden_nf, 
            hidden_nf, 
            heads=heads, 
            dropout=dropout,
            concat=False
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
