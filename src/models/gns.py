import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, LayerNorm

from src.models.basic import MLP, ConditionalMAF


class GNS_Layer(MessagePassing):
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
        super(GNS_Layer, self).__init__(aggr="sum")
        self.hidden_nf = hidden_nf
        self.edge_nf = edge_nf
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
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
        # edge_attr: [E, hidden_nf]
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        edge_features = torch.cat(
            [x_i, x_j, edge_attr], dim=-1
        )  # [E, hidden_nf + hidden_nf + hidden_nf]

        edge_attr_new = self.edge_mlp(edge_features)

        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr_new, batch=batch)

        return x_new, edge_attr_new

    def message(self, x_i, x_j, edge_attr):
        return edge_attr

    def update(self, aggr_out, x):
        if self.norm is not None:
            x_norm = self.norm(x)
        else:
            x_norm = x

        node_features = torch.cat(
            [x_norm, aggr_out], dim=-1
        )  # [N, hidden_nf + hidden_nf]
        x_new = self.node_mlp(node_features)

        return x_new


class AbsoluteGNS(nn.Module):
    def __init__(
        self,
        n_layers,
        in_node_nf,
        out_node_nf,
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
        super(AbsoluteGNS, self).__init__()

        self.node_encoder = MLP(
            in_dim=in_node_nf,
            out_dim=hidden_nf,
            hidden_dim=hidden_nf,
            n_layers=encoder_depth,
            activation=activation,
            dropout=dropout,
            out_activation=True,
        )

        self.edge_encoder = MLP(
            in_dim=hidden_nf,
            out_dim=hidden_nf,
            hidden_dim=hidden_nf,
            n_layers=encoder_depth,
            activation=activation,
            dropout=dropout,
            out_activation=True,
        )

        self.edge_bias = nn.Parameter(torch.randn(hidden_nf))

        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                GNS_Layer(
                    hidden_nf,
                    hidden_nf,
                    activation,
                    dropout,
                    norm,
                    edge_mlp_depth,
                    node_mlp_depth,
                )
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

        x = self.node_encoder(x)
        num_edges = edge_attr.shape[0]
        bias_input = self.edge_bias.expand(num_edges, -1)
        edge_attr = self.edge_encoder(bias_input)

        for layer in self.layers:
            x_new, edge_attr_new = layer(x, edge_index, edge_attr, batch)
            x = x + x_new
            edge_attr = edge_attr + edge_attr_new

        x = self.decoder(x)

        return x


class GNS(nn.Module):
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
        num_particle_types=1,
        particle_type_embedding_size=16,
    ):
        super(GNS, self).__init__()

        if num_particle_types > 1:
            self.particle_embedding = nn.Embedding(
                num_particle_types, particle_type_embedding_size
            )
            node_input_size = in_node_nf + particle_type_embedding_size
        else:
            self.particle_embedding = None
            node_input_size = in_node_nf

        self.node_encoder = MLP(
            in_dim=node_input_size,
            out_dim=hidden_nf,
            hidden_dim=hidden_nf,
            n_layers=encoder_depth,
            activation=activation,
            dropout=dropout,
            out_activation=True,
        )

        self.edge_encoder = MLP(
            in_dim=in_edge_nf,
            out_dim=hidden_nf,
            hidden_dim=hidden_nf,
            n_layers=encoder_depth,
            activation=activation,
            dropout=dropout,
            out_activation=True,
        )

        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                GNS_Layer(
                    hidden_nf,
                    hidden_nf,
                    activation,
                    dropout,
                    norm,
                    edge_mlp_depth,
                    node_mlp_depth,
                )
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

    def forward(self, data, particle_types=None):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        if self.particle_embedding is not None and particle_types is not None:
            type_embeddings = self.particle_embedding(particle_types)
            x = torch.cat([x, type_embeddings], dim=-1)

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for layer in self.layers:
            x_new, edge_attr_new = layer(x, edge_index, edge_attr, batch)
            x = x + x_new
            edge_attr = edge_attr + edge_attr_new

        x = self.decoder(x)

        return x


class StochasticGNS(nn.Module):
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
        num_particle_types=1,
        particle_type_embedding_size=16,
    ):
        super(StochasticGNS, self).__init__()

        self.stochastic = True

        if num_particle_types > 1:
            self.particle_embedding = nn.Embedding(
                num_particle_types, particle_type_embedding_size
            )
            node_input_size = in_node_nf + particle_type_embedding_size
        else:
            self.particle_embedding = None
            node_input_size = in_node_nf

        self.node_encoder = MLP(
            in_dim=node_input_size,
            out_dim=hidden_nf,
            hidden_dim=hidden_nf,
            n_layers=encoder_depth,
            activation=activation,
            dropout=dropout,
            out_activation=True,
        )

        self.edge_encoder = MLP(
            in_dim=in_edge_nf,
            out_dim=hidden_nf,
            hidden_dim=hidden_nf,
            n_layers=encoder_depth,
            activation=activation,
            dropout=dropout,
            out_activation=True,
        )

        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                GNS_Layer(
                    hidden_nf,
                    hidden_nf,
                    activation,
                    dropout,
                    norm,
                    edge_mlp_depth,
                    node_mlp_depth,
                )
            )

        self.decoder = ConditionalMAF(
            input_dim=out_node_nf,
            context_dim=hidden_nf,
            hidden_dim=hidden_nf,
            n_flows=decoder_depth,
        )

        self.to(device)

    def compute_nll(self, x, y_true):
        return self.decoder(context=x, y=y_true)

    def sample(self, x, n_samples=1):
        return self.decoder.sample(context=x, n_samples=n_samples)

    def sample_mean(self, x, n_samples=None):
        return self.decoder.sample_mean(context=x, n_samples=n_samples)

    def forward(self, data, particle_types=None):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        if self.particle_embedding is not None and particle_types is not None:
            type_embeddings = self.particle_embedding(particle_types)
            x = torch.cat([x, type_embeddings], dim=-1)

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for layer in self.layers:
            x_new, edge_attr_new = layer(x, edge_index, edge_attr, batch)
            x = x + x_new
            edge_attr = edge_attr + edge_attr_new

        return x
