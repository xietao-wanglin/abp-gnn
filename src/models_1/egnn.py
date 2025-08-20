import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, LayerNorm
from torch_scatter import scatter


class EGNN_Layer(MessagePassing):
    def __init__(
        self,
        hidden_nf,
        edge_nf=0,
        activation=nn.SiLU(),
        dropout=0.0,
        norm=True,
        attention=False,
        coords_agg="sum",
    ):
        super(EGNN_Layer, self).__init__(aggr="add")
        self.hidden_nf = hidden_nf
        self.edge_nf = edge_nf
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = LayerNorm(hidden_nf) if norm else None
        self.attention = attention
        self.coords_agg = coords_agg
        self.epsilon = 1e-8

        edge_input_dim = (
            2 * hidden_nf + 1 + edge_nf
        )  # node_i + node_j + radial + edge_attr
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_nf),
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

        coord_mlp_layers = [
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, 1, bias=False),
        ]
        self.coord_mlp = nn.Sequential(*coord_mlp_layers)

        nn.init.xavier_uniform_(self.coord_mlp[-3].weight, gain=0.001)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def forward(self, x, coords, edge_index, edge_attr=None, batch=None):
        row, col = edge_index
        coord_diff = coords[row] - coords[col]
        radial = torch.sum(coord_diff**2, dim=1, keepdim=True)

        x_orig = x

        x_new = self.propagate(
            edge_index,
            x=x,
            coords=coords,
            coord_diff=coord_diff,
            radial=radial,
            edge_attr=edge_attr,
        )

        coords_new = self.update_coords(
            coords, edge_index, coord_diff, x, edge_attr, radial
        )

        return x_new + x_orig, coords_new

    def message(self, x_i, x_j, radial, edge_attr):
        if edge_attr is None:
            edge_features = torch.cat([x_i, x_j, radial], dim=-1)
        else:
            edge_features = torch.cat([x_i, x_j, radial, edge_attr], dim=-1)

        m_ij = self.edge_mlp(edge_features)
        if self.attention:
            att_weights = self.att_mlp(m_ij)
            m_ij = m_ij * att_weights

        return m_ij

    def update(self, aggr_out, x):
        if self.norm is not None:
            x_norm = self.norm(x)
        else:
            x_norm = x
        node_features = torch.cat([x_norm, aggr_out], dim=-1)
        x_new = self.node_mlp(node_features)

        return x_new

    def update_coords(self, coords, edge_index, coord_diff, x, edge_attr, radial):
        row, col = edge_index

        if edge_attr is None:
            edge_input = torch.cat([x[row], x[col], radial], dim=-1)
        else:
            edge_input = torch.cat([x[row], x[col], radial, edge_attr], dim=-1)

        edge_feat = self.edge_mlp(edge_input)

        if self.attention:
            att_weights = self.att_mlp(edge_feat)
            edge_feat = edge_feat * att_weights

        coord_updates = coord_diff * self.coord_mlp(edge_feat)

        if self.coords_agg == "sum":
            coords_agg = scatter(
                coord_updates, row, dim=0, dim_size=coords.size(0), reduce="sum"
            )
        elif self.coords_agg == "mean":
            coords_agg = scatter(
                coord_updates, row, dim=0, dim_size=coords.size(0), reduce="mean"
            )
        else:
            raise ValueError(f"Invalid coords_agg: {self.coords_agg}")

        return coords + coords_agg


class EGNN(nn.Module):
    def __init__(
        self,
        n_layers,
        in_node_nf,
        out_node_nf,
        in_edge_nf=0,
        hidden_nf=64,
        activation=nn.SiLU(),
        device="cpu",
        dropout=0.0,
        norm=True,
        attention=False,
        coords_agg="sum",
    ):
        super(EGNN, self).__init__()
        self.name = "EGNN"
        self.n_layers = n_layers
        self.in_node_nf = in_node_nf
        self.out_node_nf = out_node_nf
        self.in_edge_nf = in_edge_nf
        self.hidden_nf = hidden_nf
        self.activation = activation
        self.dropout = dropout
        self.norm = norm

        self.encoder = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
        )

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                EGNN_Layer(
                    hidden_nf=hidden_nf,
                    edge_nf=in_edge_nf,
                    activation=activation,
                    dropout=dropout,
                    norm=norm,
                    attention=attention,
                    coords_agg=coords_agg,
                )
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
        x = data.x
        coords = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x = self.encoder(x)

        for layer in self.layers:
            x, coords = layer(x, coords, edge_index, edge_attr, batch)

        x = self.decoder(x)

        return x, coords


if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    batch_size = 2
    n_nodes = 4
    n_feat = 1
    edge_feat = 1

    graphs = []
    for b in range(batch_size):
        x = torch.randn(n_nodes, n_feat)
        pos = torch.randn(n_nodes, 2)

        edge_index = torch.combinations(torch.arange(n_nodes), 2).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        edge_attr = torch.randn(edge_index.size(1), edge_feat)

        graph = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
        graphs.append(graph)

    batch_data = Batch.from_data_list(graphs)

    model = EGNN(
        n_layers=5,
        in_node_nf=n_feat,
        out_node_nf=1,
        in_edge_nf=edge_feat,
        hidden_nf=64,
        attention=False,
        norm=False,
    )

    with torch.no_grad():
        node_outputs, updated_coords = model(batch_data)

    print(f"Shapes -- Node: {node_outputs.shape}, coords: {updated_coords.shape}")
