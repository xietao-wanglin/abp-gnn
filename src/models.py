import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, LayerNorm, GATConv


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        n_layers,
        activation=nn.SiLU(),
        dropout=0.0,
        out_activation=False,
    ):
        super().__init__()

        layers = []
        dim = in_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(activation)
            dim = hidden_dim

        layers.append(nn.Linear(dim, out_dim))
        if out_activation:
            layers.append(activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
        self.norm = nn.LayerNorm(hidden_nf) if norm else None

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


class GNS_Layer(MessagePassing):
    def __init__(
        self, hidden_nf, edge_nf, activation=nn.SiLU(), dropout=0.0, norm=True
    ):
        super(GNS_Layer, self).__init__(aggr="sum")
        self.hidden_nf = hidden_nf
        self.edge_nf = edge_nf
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = LayerNorm(hidden_nf) if norm else None

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf + hidden_nf, hidden_nf),
            self.dropout,
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + hidden_nf, hidden_nf),
            self.dropout,
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
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


class GNS(nn.Module):
    def __init__(
        self,
        n_layers,
        in_node_nf,
        out_node_nf,
        in_edge_nf,
        hidden_nf,
        activation=nn.SiLU(),
        device="cpu",
        dropout=0.0,
        norm=True,
    ):
        super(GNS, self).__init__()

        self.name = "GNS"

        self.n_layers = n_layers
        self.in_node_nf = in_node_nf
        self.out_node_nf = out_node_nf
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.activation = activation
        self.dropout = dropout
        self.norm = norm

        self.node_encoder = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(in_edge_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
        )

        self.layers = nn.ModuleList()

        for _ in range(self.n_layers):
            self.layers.append(
                GNS_Layer(hidden_nf, hidden_nf, activation, dropout, norm)
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
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for layer in self.layers:
            x_new, edge_attr_new = layer(x, edge_index, edge_attr, batch)
            x = x + x_new
            edge_attr = edge_attr + edge_attr_new

        x = self.decoder(x)

        return x


class AbsoluteGNS(nn.Module):
    def __init__(
        self,
        n_layers,
        in_node_nf,
        out_node_nf,
        hidden_nf,
        activation=nn.SiLU(),
        device="cpu",
        dropout=0.0,
        norm=True,
    ):
        super(AbsoluteGNS, self).__init__()

        self.name = "AbsoluteGNS"

        self.n_layers = n_layers
        self.in_node_nf = in_node_nf
        self.out_node_nf = out_node_nf
        self.hidden_nf = hidden_nf
        self.activation = activation
        self.dropout = dropout
        self.norm = norm

        self.node_encoder = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
        )

        self.edge_bias = nn.Parameter(torch.randn(hidden_nf))

        self.layers = nn.ModuleList()

        for _ in range(self.n_layers):
            self.layers.append(
                GNS_Layer(hidden_nf, hidden_nf, activation, dropout, norm)
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


class GNSPlus(nn.Module):
    def __init__(
        self,
        n_layers,
        in_node_nf,
        out_node_nf,
        in_edge_nf,
        hidden_nf,
        activation=nn.SiLU(),
        device="cpu",
        dropout=0.0,
        norm=True,
        num_particle_types=1,
        particle_type_embedding_size=16,
    ):
        super(GNSPlus, self).__init__()

        self.name = "GNSPlus"

        self.n_layers = n_layers
        self.in_node_nf = in_node_nf
        self.out_node_nf = out_node_nf
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.activation = activation
        self.dropout = dropout
        self.norm = norm

        if num_particle_types > 1:
            self.particle_embedding = nn.Embedding(
                num_particle_types, particle_type_embedding_size
            )
            node_input_size = in_node_nf + particle_type_embedding_size
        else:
            self.particle_embedding = None
            node_input_size = in_node_nf

        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_size, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(in_edge_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
        )

        self.layers = nn.ModuleList()

        for _ in range(self.n_layers):
            self.layers.append(
                GNS_Layer(hidden_nf, hidden_nf, activation, dropout, norm)
            )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, out_node_nf),
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


class GAT_Layer(nn.Module):
    def __init__(
        self, hidden_nf, edge_nf, heads=1, activation=nn.SiLU(), dropout=0.0, norm=True
    ):
        super(GAT_Layer, self).__init__()
        self.hidden_nf = hidden_nf
        self.edge_nf = edge_nf
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = LayerNorm(hidden_nf) if norm else None

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_nf, hidden_nf), self.dropout, activation
        )

        self.conv = GATConv(
            hidden_nf, hidden_nf, heads=heads, dropout=dropout, concat=False
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + hidden_nf, hidden_nf),
            self.dropout,
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
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
    def __init__(
        self,
        n_layers,
        in_node_nf,
        out_node_nf,
        in_edge_nf,
        hidden_nf,
        heads=4,
        activation=nn.SiLU(),
        device="cpu",
        dropout=0.0,
        norm=True,
    ):
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
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = self.encoder(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)

        x = self.decoder(x)
        return x


class ContinuousGNN(nn.Module):
    def __init__(
        self,
        n_layers,
        in_node_nf,
        out_node_nf,
        in_edge_nf,
        hidden_nf,
        activation=nn.SiLU(),
        device="cpu",
        dropout=0.0,
        norm=True,
    ):
        super(ContinuousGNN, self).__init__()

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
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        T = t.size(0)
        N = x.size(0)
        x_repeated = x.unsqueeze(0).repeat(T, 1, 1)
        t_expanded = t.unsqueeze(1).unsqueeze(2).expand(T, N, 1)
        x = torch.cat([x_repeated, torch.log(t_expanded)], dim=-1)
        x = x.view(T * N, -1)
        cumsum = torch.arange(0, T).to(x.device) * N
        num_edges = edge_index[0].shape[0]
        cumsum_edges = cumsum.repeat_interleave(num_edges, dim=0)
        edges_0 = edge_index[0].repeat(T) + cumsum_edges
        edges_1 = edge_index[1].repeat(T) + cumsum_edges
        edge_index = torch.stack([edges_0, edges_1], dim=0)
        edge_attr = edge_attr.repeat(T, 1)

        if batch is not None:
            batch_repeated = batch.unsqueeze(0).repeat(
                T, 1
            )  # Shape: [T, N_total_nodes]
            shift_values = (
                torch.arange(T, device=x.device).unsqueeze(1) * data.num_graphs
            )  # Shape: [T, 1]
            node_batch_indices = (batch_repeated + shift_values).view(
                -1
            )  # Shape: [T * N_total_nodes]
        else:
            node_batch_indices = None
        x = self.encoder(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, node_batch_indices)

        x = self.decoder(x)
        x = x.view(T, N, -1)
        return x


class GraphEncoder(nn.Module):
    def __init__(
        self,
        n_layers,
        in_node_nf,
        out_node_nf,
        in_edge_nf,
        hidden_nf,
        activation=nn.SiLU(),
        device="cpu",
        dropout=0.0,
        norm=True,
    ):
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
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = self.encoder(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)

        x = self.decoder(x)
        return x


class TimeEncoder(nn.Module):
    def __init__(
        self,
        n_layers,
        in_node_nf,
        out_node_nf,
        in_edge_nf,
        hidden_nf,
        activation=nn.SiLU(),
        device="cpu",
        dropout=0.0,
        norm=True,
    ):
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
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        T = t.size(0)
        N = x.size(0)
        x_repeated = x.unsqueeze(0).repeat(T, 1, 1)
        t_expanded = t.unsqueeze(1).unsqueeze(2).expand(T, N, 1)
        x = torch.cat([x_repeated, torch.log(t_expanded)], dim=-1)
        x = x.view(T * N, -1)
        cumsum = torch.arange(0, T).to(x.device) * N
        num_edges = edge_index[0].shape[0]
        cumsum_edges = cumsum.repeat_interleave(num_edges, dim=0)
        edges_0 = edge_index[0].repeat(T) + cumsum_edges
        edges_1 = edge_index[1].repeat(T) + cumsum_edges
        edge_index = torch.stack([edges_0, edges_1], dim=0)
        edge_attr = edge_attr.repeat(T, 1)

        if batch is not None:
            batch_repeated = batch.unsqueeze(0).repeat(
                T, 1
            )  # Shape: [T, N_total_nodes]
            shift_values = (
                torch.arange(T, device=x.device).unsqueeze(1) * data.num_graphs
            )  # Shape: [T, 1]
            node_batch_indices = (batch_repeated + shift_values).view(
                -1
            )  # Shape: [T * N_total_nodes]
        else:
            node_batch_indices = None
        x = self.encoder(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, node_batch_indices)

        x = self.decoder(x)
        x = x.view(T, N, -1)
        return x


class LatentGNN(nn.Module):
    def __init__(
        self,
        n_layers,
        in_node_nf,
        out_node_nf,
        latent_nf,
        in_edge_nf,
        hidden_nf,
        activation=nn.SiLU(),
        device="cpu",
        dropout=0.0,
        norm=True,
    ):
        super(LatentGNN, self).__init__()

        self.name = "LatentGNN"

        self.n_layers = n_layers
        self.in_node_nf = in_node_nf
        self.out_node_nf = out_node_nf
        self.latent_nf = latent_nf
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.activation = activation
        self.dropout = dropout
        self.norm = norm
        self.ic_encoder = GraphEncoder(
            n_layers,
            in_node_nf,
            latent_nf,
            in_edge_nf,
            hidden_nf,
            activation,
            device,
            dropout,
            norm,
        )

        self.vel_encoder = GraphEncoder(
            n_layers,
            in_node_nf,
            latent_nf,
            in_edge_nf,
            hidden_nf,
            activation,
            device,
            dropout,
            norm,
        )

        self.time_encoder = TimeEncoder(
            n_layers,
            1 + in_node_nf,
            latent_nf,
            in_edge_nf,
            hidden_nf,
            activation,
            device,
            dropout,
            norm,
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, out_node_nf),
        )

        self.to(device)

    def forward(self, data, t):
        e = self.ic_encoder(data)  # (N, x_dim) -> (N, latent_dim)
        c = self.vel_encoder(data)  # (N, x_dim) -> (N, latent_dim)
        tau = self.time_encoder(data, t)  # (N, T, x_dim) -> (N, T, latent_dim)

        y = e.unsqueeze(0) + tau * c.unsqueeze(
            0
        )  # (N, latent_dim) + (T, N, latent_dim)*(N, latent_dim) -> (T, N, latent_dim)
        N = data.x.size(0)
        T = t.size(0)
        y_flat = y.view(T * N, -1)  # (T, N, latent_nf) -> (T*N, latent_nf)
        x_flat = self.decoder(y_flat)  # (T*N, latent_nf) -> (T*N, out_node_nf)
        x = x_flat.view(T, N, -1)  # (T*N, out_node_nf) -> (T, N, out_node_nf)

        return x
