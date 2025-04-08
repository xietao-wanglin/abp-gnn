"""
From: https://github.com/MinkaiXu/EGNO/
MIT License

Copyright (c) 2024 Minkai Xu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np
import math

def aggregate(message, row_index, n_node, aggr='sum', mask=None):
    """
    The aggregation function (aggregate edge messages towards nodes)
    :param message: The edge message with shape [M, K]
    :param row_index: The row index of edges with shape [M]
    :param n_node: The number of nodes, N
    :param aggr: aggregation type, sum or mean
    :param mask: the edge mask (used in mean aggregation for counting degree)
    :return: The aggreagated node-wise information with shape [N, K]
    """
    result_shape = (n_node, message.shape[1])
    result = message.new_full(result_shape, 0)  # [N, K]
    row_index = row_index.unsqueeze(-1).expand(-1, message.shape[1])  # [M, K]
    result.scatter_add_(0, row_index, message)  # [N, K]
    if aggr == 'sum':
        pass
    elif aggr == 'mean':
        count = message.new_full(result_shape, 0)
        ones = torch.ones_like(message)
        if mask is not None:
            ones = ones * mask.unsqueeze(-1)
        count.scatter_add_(0, row_index, ones)
        result = result / count.clamp(min=1)
    else:
        raise NotImplementedError('Unknown aggregation method:', aggr)
    return result  # [N, K]

class BaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, residual=False, last_act=False, flat=False):
        super(BaseMLP, self).__init__()
        self.residual = residual
        if flat:
            activation = nn.Tanh()
            hidden_dim = 4 * hidden_dim
        if residual:
            assert output_dim == input_dim
        if last_act:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim),
                activation
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        return self.mlp(x) if not self.residual else self.mlp(x) + x

class EquivariantScalarNet(nn.Module):
    def __init__(self, n_vector_input, hidden_dim, activation, n_scalar_input=0, norm=True, flat=True):
        """
        The universal O(n) equivariant network using scalars.
        :param n_input: The total number of input vectors.
        :param hidden_dim: The hidden dim of the network.
        :param activation: The activation function.
        """
        super(EquivariantScalarNet, self).__init__()
        self.input_dim = n_vector_input * n_vector_input + n_scalar_input
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        # self.output_dim = n_vector_input
        self.activation = activation
        self.norm = norm
        self.in_scalar_net = BaseMLP(self.input_dim, self.hidden_dim, self.hidden_dim, self.activation, last_act=True,
                                     flat=flat)
        self.out_vector_net = BaseMLP(self.hidden_dim, self.hidden_dim, n_vector_input, self.activation, flat=flat)
        self.out_scalar_net = BaseMLP(self.hidden_dim, self.hidden_dim, self.output_dim, self.activation, flat=flat)

    def forward(self, vectors, scalars=None):
        """
        :param vectors: torch.Tensor with shape [N, 3, K] or a list of torch.Tensor
        :param scalars: torch.Tensor with shape [N, L] (Optional)
        :return: A vector that is equivariant to the O(n) transformations of input vectors with shape [N, 3]
        """
        if type(vectors) == list:
            Z = torch.stack(vectors, dim=-1)  # [N, 3, K]
        else:
            Z = vectors
        K = Z.shape[-1]
        Z_T = Z.transpose(-1, -2)  # [N, K, 3]
        scalar = torch.einsum('bij,bjk->bik', Z_T, Z)  # [N, K, K]
        scalar = scalar.reshape(-1, K * K)  # [N, KK]
        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)  # [N, KK]
        if scalars is not None:
            scalar = torch.cat((scalar, scalars), dim=-1)  # [N, KK + L]
        scalar = self.in_scalar_net(scalar)  # [N, K]
        vec_scalar = self.out_vector_net(scalar)  # [N, K]
        vector = torch.einsum('bij,bj->bi', Z, vec_scalar)  # [N, 3]
        scalar = self.out_scalar_net(scalar)  # [N, H]

        return vector, scalar

class InvariantScalarNet(nn.Module):
    def __init__(self, n_vector_input, hidden_dim, output_dim, activation, n_scalar_input=0, norm=True, last_act=False,
                 flat=False):
        """
        The universal O(n) invariant network using scalars.
        :param n_vector_input: The total number of input vectors.
        :param hidden_dim: The hidden dim of the network.
        :param activation: The activation function.
        """
        super(InvariantScalarNet, self).__init__()
        self.input_dim = n_vector_input * n_vector_input + n_scalar_input
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.norm = norm
        self.scalar_net = BaseMLP(self.input_dim, self.hidden_dim, self.output_dim, self.activation, last_act=last_act,
                                  flat=flat)

    def forward(self, vectors, scalars=None):
        """
        :param vectors: torch.Tensor with shape [N, 3, K] or a list of torch.Tensor with shape [N, 3]
        :param scalars: torch.Tensor with shape [N, L] (Optional)
        :return: A scalar that is invariant to the O(n) transformations of input vectors  with shape [N, K]
        """
        if type(vectors) == list:
            Z = torch.stack(vectors, dim=-1)  # [N, 3, K]
        else:
            Z = vectors
        K = Z.shape[-1]
        Z_T = Z.transpose(-1, -2)  # [N, K, 3]
        scalar = torch.einsum('bij,bjk->bik', Z_T, Z)  # [N, K, K]
        scalar = scalar.reshape(-1, K * K)  # [N, KK]
        if self.norm:
            scalar = F.normalize(scalar, p=2, dim=-1)  # [N, KK]
        if scalars is not None:
            scalar = torch.cat((scalar, scalars), dim=-1)  # [N, KK + L]
        scalar = self.scalar_net(scalar)  # [N, K]
        return scalar

class EGNN_Layer(nn.Module):
    def __init__(self, in_edge_nf, hidden_nf, activation=nn.SiLU(), with_v=False, flat=False, norm=False,
                 h_update=True):
        super(EGNN_Layer, self).__init__()
        self.with_v = with_v
        self.edge_message_net = InvariantScalarNet(n_vector_input=1, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                                   activation=activation, n_scalar_input=2 * hidden_nf + in_edge_nf,
                                                   norm=norm, last_act=True, flat=flat)
        self.coord_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation,
                                 flat=flat)
        if self.with_v:
            self.node_v_net = BaseMLP(input_dim=hidden_nf, hidden_dim=hidden_nf, output_dim=1, activation=activation,
                                      flat=flat)
        else:
            self.node_v_net = None
        self.h_update = h_update
        if self.h_update:
            self.node_net = BaseMLP(input_dim=hidden_nf + hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                    activation=activation, flat=flat)

    def forward(self, x, h, edge_index, edge_fea, v=None):
        row, col = edge_index
        rij = x[row] - x[col]  # [BM, 3]
        hij = torch.cat((h[row], h[col], edge_fea), dim=-1)  # [BM, 2K+T]
        message = self.edge_message_net(vectors=[rij], scalars=hij)  # [BM, 3]
        coord_message = self.coord_net(message)  # [BM, 1]
        f = (x[row] - x[col]) * coord_message  # [BM, 3]
        tot_f = aggregate(message=f, row_index=row, n_node=x.shape[0], aggr='mean')  # [BN, 3]
        tot_f = torch.clamp(tot_f, min=-100, max=100)

        if v is not None:
            x = x + self.node_v_net(h) * v + tot_f
        else:
            x = x + tot_f  # [BN, 3]

        tot_message = aggregate(message=message, row_index=row, n_node=x.shape[0], aggr='sum')  # [BN, K]
        node_message = torch.cat((h, tot_message), dim=-1)  # [BN, K+K]
        if self.h_update:
            h = self.node_net(node_message)  # [BN, K]
        return x, v, h

class EGNN(nn.Module):
    def __init__(self, n_layers, in_node_nf, in_edge_nf, hidden_nf, activation=nn.SiLU(), device='cpu', with_v=False,
                 flat=False, norm=False):
        super(EGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.with_v = with_v
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(self.n_layers):
            layer = EGNN_Layer(in_edge_nf, hidden_nf, activation=activation, with_v=with_v, flat=flat, norm=norm)
            # if i == self.n_layers - 1:
            #     layer = EGNN_Layer(in_edge_nf, hidden_nf, activation=activation, with_v=with_v, flat=flat, norm=norm,
            #                        h_update=False)
            # else:
            #     layer = EGNN_Layer(in_edge_nf, hidden_nf, activation=activation, with_v=with_v, flat=flat, norm=norm)
            self.layers.append(layer)
        self.to(device)

    def forward(self, x, h, edge_index, edge_fea, v=None):
        h = self.embedding(h)
        for i in range(self.n_layers):
            x, v, h = self.layers[i](x, h, edge_index, edge_fea, v=v)
        return (x, v, h) if v is not None else (x, h)

class EGMN(nn.Module):
    def __init__(self, n_layers, n_vector_input, hidden_dim, n_scalar_input, activation=nn.SiLU(), norm=False, flat=False):
        super(EGMN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        for i in range(self.n_layers):
            cur_layer = EquivariantScalarNet(n_vector_input=n_vector_input + i, hidden_dim=hidden_dim,
                                             activation=activation, n_scalar_input=n_scalar_input if i == 0 else hidden_dim,
                                             norm=norm, flat=flat)
            self.layers.append(cur_layer)

    def forward(self, vectors, scalars):
        cur_vectors = vectors
        for i in range(self.n_layers):
            vector, scalars = self.layers[i](cur_vectors, scalars)
            cur_vectors.append(vector)
        return cur_vectors[-1], scalars

class GNN_Layer(nn.Module):
    def __init__(self, in_edge_nf, hidden_nf, activation=nn.SiLU(), with_v=False, flat=False, dropout=0.1, norm=True):
        super(GNN_Layer, self).__init__()
        self.with_v = with_v
        self.edge_message_net = BaseMLP(input_dim=in_edge_nf + 2 * hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                        activation=activation, flat=flat)
        self.node_net = BaseMLP(input_dim=hidden_nf + hidden_nf, hidden_dim=hidden_nf, output_dim=hidden_nf,
                                activation=activation, flat=flat)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.norm = nn.LayerNorm(hidden_nf) if norm else None

    def forward(self, h, edge_index, edge_fea):
        row, col = edge_index
        hij = torch.cat((h[row], h[col], edge_fea), dim=-1)  # [BM, 2K+T]
        message = self.edge_message_net(hij)  # [BM, K]
        agg = aggregate(message=message, row_index=row, n_node=h.shape[0], aggr='mean')  # [BN, K]
        h = h + self.node_net(torch.cat((agg, h), dim=-1))
        if self.norm:
            h = self.norm(h)

        if self.dropout:
            h = self.dropout(h)

        return h

class Linear_dynamics(nn.Module):
    def __init__(self, device='cpu'):
        super(Linear_dynamics, self).__init__()
        self.time = nn.Parameter(torch.ones(1))
        self.device = device
        self.to(self.device)

    def forward(self, x, v):
        return x + v * self.time

class RF_vel(nn.Module):
    def __init__(self, hidden_nf, edge_attr_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4):
        super(RF_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL_rf_vel(nf=hidden_nf, edge_attr_nf=edge_attr_nf, act_fn=act_fn))
        self.to(self.device)

    def forward(self, vel_norm, x, edges, vel, edge_attr):
        for i in range(0, self.n_layers):
            x, _ = self._modules["gcl_%d" % i](x, vel_norm, vel, edges, edge_attr)
        return x

class GCL_rf_vel(nn.Module):
    def __init__(self,  nf=64, edge_attr_nf=0, act_fn=nn.LeakyReLU(0.2), coords_weight=1.0):
        super(GCL_rf_vel, self).__init__()
        self.coords_weight = coords_weight
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(1, nf),
            act_fn,
            nn.Linear(nf, 1))

        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(1 + edge_attr_nf, nf),
                                 act_fn,
                                 layer,
                                 nn.Tanh())

    def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
        row, col = edge_index
        edge_m = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_m)
        x += vel * self.coord_mlp_vel(vel_norm)
        return x, edge_attr

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        return m_ij

    def node_model(self, x, edge_index, edge_m):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_m, row, num_segments=x.size(0))
        x_out = x + agg * self.coords_weight
        return x_out

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

class FullMLP(nn.ModuleList):
    def __init__(self, in_node_nf, hidden_nf, n_layers, activation=nn.SiLU(), flat=False, device='cpu'):
        super(FullMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(n_layers):
            self.layers.append(BaseMLP(hidden_nf, hidden_nf, hidden_nf, activation,
                                       residual=True, last_act=True, flat=flat))
        self.output = nn.Linear(hidden_nf, 3)
        self.to(device)

    def forward(self, x):
        x = self.embedding(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return self.output(x)

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, edge_dim=1, 
                 heads=1, 
                 concat=False, 
                 activation=nn.SiLU(), 
                 dropout=0.1, 
                 norm=True):
        super(GATLayer, self).__init__()
        self.gat = GATv2Conv(in_features, 
                             out_features // heads, 
                             heads=heads, 
                             concat=concat, 
                             dropout=dropout,
                             edge_dim=edge_dim)
        self.activation = activation
        self.norm = nn.LayerNorm(out_features) if norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
    
    def forward(self, x, edge_index, edge_attr):
        x = self.gat(x, edge_index, edge_attr)
        if self.activation:
            x = self.activation(x)
        if self.norm:
            x = self.norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class GNN(nn.Module):
    def __init__(self, n_layers, 
                 in_node_nf, 
                 in_edge_nf, 
                 hidden_nf, 
                 activation=nn.SiLU(), 
                 device='cpu', 
                 flat=False, 
                 dropout=0.1, 
                 norm=True,):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.norm = norm
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_nf) for _ in range(n_layers)]) if norm else None
        # input feature mapping
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(self.n_layers):
            layer = GNN_Layer(in_edge_nf, hidden_nf, activation=activation, flat=flat, dropout=dropout, norm=norm)
            self.layers.append(layer)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden_nf, 3)
        )
        self.to(device)

    def forward(self, h, edge_index, edge_fea):
        h = self.embedding(h)
        for i in range(self.n_layers):
            h = self.layers[i](h, edge_index, edge_fea)
            if self.norm:
                h = self.norm_layers[i](h)
            if self.dropout:
                h = self.dropout(h)
        h = self.decoder(h)
        return h

class GAT(nn.Module):
    def __init__(self, n_layers, in_node_nf, 
                 hidden_nf, heads=4, in_edge_nf=0,
                 activation=nn.SiLU(), device='cpu', dropout=0.0, norm=True):
        super(GAT, self).__init__()
        self.n_layers = n_layers
        self.norm = norm
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.edge_embedding = nn.Linear(in_edge_nf, in_edge_nf)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GATLayer(hidden_nf, 
                                        hidden_nf, 
                                        edge_dim=in_edge_nf,
                                        heads=heads, 
                                        activation=activation, 
                                        dropout=dropout, 
                                        norm=norm,
                                        concat=True))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_nf, 3)
        )
        self.to(device)
    
    def forward(self, x, edge_index, edge_attr):
        x = self.embedding(x)
        #edge_attr = self.edge_embedding(edge_attr)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        x = self.decoder(x)
        return x
    
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')


class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        # x: (B, C, H, W)
        y = torch.einsum('bchw,co->bohw', x, self.W) + self.b[None, :, None, None]
        if y.stride()[1] == 1:
            y = y.contiguous()
        return y
    

@torch.jit.script
def compl_mul1d(a, b):    
    # (M, N, in_ch), (in_ch, out_ch, M) -> (M, N, out_channel)
    return torch.einsum("mni,iom->mno", a, b)


class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1

        self.scale = (1 / (in_ch*out_ch))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, self.modes1, 2, dtype=torch.float))

    def forward(self, x):
        T, N, C = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        with torch.cuda.amp.autocast(enabled=False):
        # with torch.autocast(device_type='cuda', enabled=False):
            x_ft = torch.fft.rfftn(x.float(), dim=[0])
            # Multiply relevant Fourier modes
            out_ft = compl_mul1d(x_ft[:self.modes1], torch.view_as_complex(self.weights1))
            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=[T], dim=[0])
        return x


class TimeConv(nn.Module):
    def __init__(self, in_ch, out_ch, modes, act, with_nin=False):
        super(TimeConv, self).__init__()
        self.with_nin = with_nin
        self.t_conv = SpectralConv1d(in_ch, out_ch, modes)
        # if with_nin:
        #     self.nin = NIN(in_ch, out_ch)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        h = self.t_conv(x)
        # if self.with_nin:
        #     x = self.nin(x)
        out = self.act(h)
        return x + out


@torch.jit.script
def compl_mul1d_x(a, b):
    # (M, N, in_ch), (in_ch, out_ch, M) -> (M, N, out_channel)
    return torch.einsum("mndi,iom->mndo", a, b)


class SpectralConv1d_x(nn.Module):
    def __init__(self, in_ch, out_ch, modes1):
        super(SpectralConv1d_x, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1

        # self.scale = (1 / (in_ch*out_ch))
        self.scale = 0.1
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_ch, out_ch, self.modes1, 2, dtype=torch.float))

    def forward(self, x):
        T, N, D, C = x.shape  # D should be 3
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        with torch.cuda.amp.autocast(enabled=False):
        # with torch.autocast(device_type='cuda', enabled=False):
            x_ft = torch.fft.rfftn(x.float(), dim=[0])
            # Multiply relevant Fourier modes
            out_ft = compl_mul1d_x(x_ft[:self.modes1], torch.view_as_complex(self.weights1))
            # Return to physical space
            x = torch.fft.irfftn(out_ft, s=[T], dim=[0])
        return x


class TimeConv_x(nn.Module):
    def __init__(self, in_ch, out_ch, modes, act, with_nin=False):
        super(TimeConv_x, self).__init__()
        self.with_nin = with_nin
        self.t_conv = SpectralConv1d_x(in_ch, out_ch, modes)
        # if with_nin:
        #     self.nin = NIN(in_ch, out_ch)

    def forward(self, x):  # x: [T, N, D, C]
        # x = x.unsqueeze(-1)  # [T, N, D, C]
        h = self.t_conv(x)
        # if self.with_nin:
        #     x = self.nin(x)
        return x + h
    
class EGNO(EGNN):
    def __init__(self, n_layers, in_node_nf, in_edge_nf, hidden_nf, activation=nn.SiLU(), device='cpu', with_v=False,
                 flat=False, norm=False, use_time_conv=True, num_modes=2, num_timesteps=8, time_emb_dim=32):
        self.time_emb_dim = time_emb_dim
        in_node_nf = in_node_nf + self.time_emb_dim

        super(EGNO, self).__init__(n_layers, in_node_nf, in_edge_nf, hidden_nf, activation, device, with_v, flat, norm)
        self.use_time_conv = use_time_conv
        self.num_timesteps = num_timesteps
        self.device = device
        self.hidden_nf = hidden_nf

        if use_time_conv:
            self.time_conv_modules = nn.ModuleList()
            self.time_conv_x_modules = nn.ModuleList()
            for i in range(n_layers):
                self.time_conv_modules.append(TimeConv(hidden_nf, hidden_nf, num_modes, activation, with_nin=False))
                self.time_conv_x_modules.append(TimeConv_x(2, 2, num_modes, activation, with_nin=False))

        self.to(self.device)

    def forward(self, x, h, edge_index, edge_fea, v=None, loc_mean=None):  # [BN, H]

        T = self.num_timesteps

        num_nodes = h.shape[0]
        num_edges = edge_index[0].shape[0]

        cumsum = torch.arange(0, T).to(self.device) * num_nodes
        cumsum_nodes = cumsum.repeat_interleave(num_nodes, dim=0)
        cumsum_edges = cumsum.repeat_interleave(num_edges, dim=0)

        time_emb = get_timestep_embedding(torch.arange(T).to(x), embedding_dim=self.time_emb_dim, max_positions=10000)  # [T, H_t]
        h = h.unsqueeze(0).repeat(T, 1, 1)  # [T, BN, H]
        time_emb = time_emb.unsqueeze(1).repeat(1, num_nodes, 1)  # [T, BN, H_t]
        h = torch.cat((h, time_emb), dim=-1)  # [T, BN, H+H_t]
        h = h.view(-1, h.shape[-1])  # [T*BN, H+H_t]

        h = self.embedding(h)
        x = x.repeat(T, 1)
        loc_mean = loc_mean.repeat(T, 1)
        edges_0 = edge_index[0].repeat(T) + cumsum_edges
        edges_1 = edge_index[1].repeat(T) + cumsum_edges
        edge_index = [edges_0, edges_1]
        v = v.repeat(T, 1)

        edge_fea = edge_fea.repeat(T, 1)

        for i in range(self.n_layers):
            if self.use_time_conv:
                time_conv = self.time_conv_modules[i]
                h = time_conv(h.view(T, num_nodes, self.hidden_nf)).view(T * num_nodes, self.hidden_nf)
                x_translated = x - loc_mean
                time_conv_x = self.time_conv_x_modules[i]
                X = torch.stack((x_translated, v), dim=-1)
                temp = time_conv_x(X.view(T, num_nodes, 3, 2))
                x = temp[..., 0].view(T * num_nodes, 3) + loc_mean
                v = temp[..., 1].view(T * num_nodes, 3)

            x, v, h = self.layers[i](x, h, edge_index, edge_fea, v=v)
        return (x, v, h) if v is not None else (x, h)
