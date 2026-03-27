import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from src.models.basic import MLP


def apply_pbc(diff, box_length):
    if box_length is None:
        return diff
    return diff - box_length * torch.round(diff / box_length)


class EGNNLayer(MessagePassing):
    def __init__(
        self,
        hidden_nf,
        edge_nf,
        activation=nn.SiLU(),
        dropout=0.0,
        norm=True,
        edge_mlp_depth=2,
        node_mlp_depth=2,
        aggr="sum",
    ):
        super(EGNNLayer, self).__init__(aggr=aggr)
        self.hidden_nf = hidden_nf

        self.edge_mlp = MLP(
            in_dim=2 * hidden_nf + edge_nf,
            out_dim=hidden_nf,
            hidden_dim=hidden_nf,
            n_layers=edge_mlp_depth,
            activation=activation,
            dropout=dropout,
            out_activation=True,
            norm=norm,
        )

        self.theta_mlp = MLP(
            in_dim=2 * hidden_nf,
            out_dim=1,
            hidden_dim=hidden_nf,
            n_layers=node_mlp_depth,
            activation=activation,
            dropout=dropout,
            out_activation=False,
            norm=False,
        )

        self.pos_mlp = MLP(
            in_dim=hidden_nf,
            out_dim=1,
            hidden_dim=hidden_nf,
            n_layers=2,
            activation=activation,
            out_activation=False,
            norm=False,
        )
        nn.init.xavier_uniform_(self.pos_mlp.net[-1].weight, gain=0.001)

        self.normal_mlp = MLP(
            in_dim=2 * hidden_nf,
            out_dim=2,
            hidden_dim=hidden_nf,
            n_layers=node_mlp_depth,
            activation=activation,
            out_activation=False,
            norm=False,
        )
        nn.init.xavier_uniform_(self.normal_mlp.net[-1].weight, gain=0.001)

        self.node_mlp = MLP(
            in_dim=2 * hidden_nf,
            out_dim=hidden_nf,
            hidden_dim=hidden_nf,
            n_layers=node_mlp_depth,
            activation=activation,
            dropout=dropout,
            out_activation=True,
            norm=norm,
        )

    def forward(self, h, x, theta, edge_index, edge_attr, box_length=None, batch=None):
        """
        h: [N, hidden_nf] - Node features
        x: [N, 2] - Node positions
        theta: [N, 1] - Node orientations (angles)
        edge_index: [2, E]
        edge_attr: [E, hidden_nf]
        """
        row, col = edge_index

        edge_features = torch.cat([h[row], h[col], edge_attr], dim=-1)
        m_ij = self.edge_mlp(edge_features)

        aggr_out, delta_x_sum = self.propagate(
            edge_index, h=h, x=x, m_ij=m_ij, box_length=box_length, batch=batch
        )

        delta_theta = self.theta_mlp(torch.cat([h, aggr_out], dim=-1))
        theta_new = theta + delta_theta

        n_l = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)  # n_i^l
        n_l_plus_1 = torch.cat(
            [torch.cos(theta_new), torch.sin(theta_new)], dim=-1
        )  # n_i^{l+1}

        p_weights = self.normal_mlp(torch.cat([h, aggr_out], dim=-1))

        pos_normal_update = (p_weights[:, 0:1] * n_l) + (p_weights[:, 1:2] * n_l_plus_1)

        x_new = x + pos_normal_update + delta_x_sum

        h_new = h + self.node_mlp(torch.cat([h, aggr_out], dim=-1))

        edge_attr_new = edge_attr + m_ij

        return h_new, x_new, theta_new, edge_attr_new

    def message(self, x_i, x_j, m_ij, box_length):
        w_ij = self.pos_mlp(m_ij)
        rel_pos = x_j - x_i
        rel_pos_pbc = apply_pbc(rel_pos, box_length)
        rel_pos_update = rel_pos_pbc * w_ij
        return m_ij, rel_pos_update

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        m_ij, rel_pos_update = inputs

        aggr_m = super().aggregate(m_ij, index, ptr, dim_size)
        aggr_pos = super().aggregate(rel_pos_update, index, ptr, dim_size)

        return aggr_m, aggr_pos


class EGNN(nn.Module):
    def __init__(
        self,
        n_layers,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        out_node_nf,
        activation=nn.SiLU(),
        device="cpu",
        norm=True,
        num_particle_types=1,
        particle_type_embedding_size=16,
    ):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf

        if num_particle_types > 1:
            self.particle_embedding = nn.Embedding(
                num_particle_types, particle_type_embedding_size
            )
            node_input_size = in_node_nf + particle_type_embedding_size
        else:
            self.particle_embedding = None
            node_input_size = in_node_nf

        if node_input_size == 0:
            node_input_size = 1
        self.node_encoder = nn.Linear(node_input_size, hidden_nf)
        self.edge_encoder = nn.Linear(in_edge_nf, hidden_nf)

        self.layers = nn.ModuleList(
            [
                EGNNLayer(hidden_nf, hidden_nf, activation=activation, norm=norm)
                for _ in range(n_layers)
            ]
        )

        self.decoder = nn.Linear(hidden_nf, out_node_nf)
        self.to(device)

    def forward(self, data, particle_types=None):
        edge_attr = self.edge_encoder(data.edge_attr)
        edge_index = data.edge_index
        x = data.x
        theta = data.theta
        batch = data.batch
        box_length = getattr(data, "box_length", None)
        if box_length is not None:
            box_length = box_length[0]
        h = getattr(data, "h", None)

        if self.particle_embedding is not None and particle_types is not None:
            type_embeddings = self.particle_embedding(particle_types)
            if h is None:
                h = type_embeddings
            else:
                h = torch.cat([h, type_embeddings], dim=-1)

        if h is None:
            h = torch.ones((x.size(0), 1), device=x.device)
        h = self.node_encoder(h)

        for layer in self.layers:
            h, x, theta, edge_attr = layer(
                h, x, theta, edge_index, edge_attr, box_length=box_length, batch=batch
            )

        return x
