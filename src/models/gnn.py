import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, LayerNorm

from src.models.basic import MLP


class GNN_Layer(MessagePassing):
    def __init__(
        self,
        hidden_nf,
        edge_nf,
        activation=nn.SiLU(),
        dropout=0.0,
        norm=True,
        edge_mlp_depth=2,
        node_mlp_depth=2,
    ):
        super().__init__(aggr="sum")
        self.hidden_nf = hidden_nf
        self.edge_nf = edge_nf
        self.activation = activation
        self.norm = LayerNorm(hidden_nf) if norm else None

        self.edge_mlp = MLP(
            in_dim=2 * hidden_nf + edge_nf,
            out_dim=hidden_nf,
            hidden_dim=hidden_nf,
            n_layers=edge_mlp_depth,
            activation=activation,
            dropout=dropout,
            out_activation=True,
        )

        self.node_mlp = MLP(
            in_dim=2 * hidden_nf,
            out_dim=hidden_nf,
            hidden_dim=hidden_nf,
            n_layers=node_mlp_depth,
            activation=activation,
            dropout=dropout,
            out_activation=True,
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        # x: [N, hidden_nf]
        # edge_index: [2, E]
        # edge_attr: [E, edge_nf]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, batch=batch)

    def message(self, x_i, x_j, edge_attr):
        edge_features = torch.cat(
            [x_i, x_j, edge_attr], dim=-1
        )  # Shape: [E, hidden_nf + hidden_nf + edge_nf]
        m_ij = self.edge_mlp(edge_features)
        return m_ij

    def update(self, aggr_out, x):
        if self.norm is not None:
            x_norm = self.norm(x)
        else:
            x_norm = x

        node_features = torch.cat(
            [x_norm, aggr_out], dim=-1
        )  # Shape: [N, hidden_nf + hidden_nf]
        x_new = self.node_mlp(node_features)
        x_new = x + x_new

        return x_new


class GNN(nn.Module):
    def __init__(
        self,
        n_layers,
        in_node_nf,
        out_node_nf,
        in_edge_nf,
        hidden_nf,
        encoder_depth=2,
        decoder_depth=2,
        edge_mlp_depth=2,
        node_mlp_depth=2,
        activation=nn.SiLU(),
        device="cpu",
        dropout=0.0,
        norm=True,
    ):
        super().__init__()

        self.encoder = MLP(
            in_dim=in_node_nf,
            out_dim=hidden_nf,
            hidden_dim=hidden_nf,
            n_layers=encoder_depth,
            activation=activation,
            dropout=dropout,
            out_activation=True,
        )

        self.layers = nn.ModuleList(
            [
                GNN_Layer(
                    hidden_nf,
                    in_edge_nf,
                    activation,
                    dropout,
                    norm,
                    edge_mlp_depth,
                    node_mlp_depth,
                )
                for _ in range(n_layers)
            ]
        )

        self.decoder = MLP(
            in_dim=hidden_nf,
            out_dim=out_node_nf,
            hidden_dim=hidden_nf,
            n_layers=decoder_depth,
            activation=activation,
            dropout=dropout,
            out_activation=False,
        )

        self.to(device)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)
        return self.decoder(x)
