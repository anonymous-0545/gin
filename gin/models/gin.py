import inspect
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter

from torch.nn.init import xavier_uniform_, constant_

zeros_initializer = partial(constant_, val=0.0)

from torch_geometric.utils import softmax


class Dense(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            activation=None,
            weight_init=xavier_uniform_,
            bias_init=zeros_initializer,
            norm=None,
            gain=None,
    ):
        # initialize linear layer y = xW^T + b
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.gain = gain
        super(Dense, self).__init__(in_features, out_features, bias)
        if inspect.isclass(activation):
            self.activation = activation()
        self.activation = activation

        if norm == 'layer':
            self.norm = nn.LayerNorm(out_features)
        elif norm == 'batch':
            self.norm = nn.BatchNorm1d(out_features)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm1d(out_features)
        else:
            self.norm = None

    def reset_parameters(self):
        if self.gain:
            self.weight_init(self.weight, gain=self.gain)
        else:
            self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        y = super(Dense, self).forward(inputs)
        if self.norm is not None:
            y = self.norm(y)
        if self.activation:
            y = self.activation(y)
        return y


class CosineCutoff(nn.Module):
    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, n_rbf=50, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.n_rbf)
        betas = torch.tensor([(2 / self.n_rbf * (1 - start_value)) ** -2] * self.n_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)


class Distance(nn.Module):
    def __init__(self, cutoff, max_num_neighbors=32, loop=True):
        super(Distance, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.loop = loop

    def forward(self, pos, batch):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=self.loop,
                                  max_num_neighbors=self.max_num_neighbors)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        return edge_index, edge_weight, edge_vec


class NodeEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff, max_z=100):
        super(NodeEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)

        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W


class EdgeEmbedding(MessagePassing):

    def __init__(self, num_rbf, hidden_channels):
        super(EdgeEmbedding, self).__init__(aggr=None)
        self.edge_proj = nn.Linear(num_rbf, hidden_channels)

        self.edge_up = nn.Sequential(self.edge_proj)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.fill_(0)

    def forward(self, edge_index, edge_attr, x):
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_i, x_j, edge_attr):
        return (x_i + x_j) * self.edge_up(edge_attr)

    def aggregate(self, features, index):
        # no aggregate
        return features


class GIAI(MessagePassing):
    def __init__(self, C_h: int, activation: Callable, weight_init=nn.init.xavier_uniform_, bias_init=nn.init.zeros_,
                 aggr="add", node_dim=0, epsilon: float = 1e-7, cutoff=5.0, num_heads=8, dropout=0.0,
                 last_layer=False, scale_edge=True):

        super(GIAI, self).__init__(aggr=aggr, node_dim=node_dim)
        self.epsilon = epsilon
        self.last_layer = last_layer
        self.scale_edge = scale_edge

        self.dropout = dropout
        self.C_h = C_h

        PartialGamma = partial(Dense, weight_init=weight_init, bias_init=bias_init)
        self.gamma_s = nn.Sequential(
            PartialGamma(C_h, C_h, activation=activation),
            PartialGamma(C_h, 3 * C_h, activation=None),
        )

        self.num_heads = num_heads
        self.q_w = PartialGamma(C_h, C_h, activation=None)
        self.k_w = PartialGamma(C_h, C_h, activation=None)

        if not self.last_layer:
            self.w_ru = PartialGamma(C_h, C_h, activation=None)
            self.w_vq = PartialGamma(C_h, C_h, activation=None, bias=False)
            self.w_vk = PartialGamma(C_h, C_h, activation=None, bias=False)

        self.gamma_v = nn.Sequential(
            PartialGamma(C_h, C_h, activation=activation),
            PartialGamma(C_h, 3 * C_h, activation=None),
        )

        self.phi = CosineCutoff(cutoff)
        self.wre_sigmak = PartialGamma(
            C_h,
            C_h,
            activation=activation,
        )
        self.w_ra = PartialGamma(
            C_h,
            C_h * 3,
            activation=None,
        )

        self.layernorm = nn.LayerNorm(C_h)
        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        for l in self.gamma_s:
            l.reset_parameters()
        for l in self.gamma_v:
            l.reset_parameters()

        self.q_w.reset_parameters()
        self.k_w.reset_parameters()
        self.w_ra.reset_parameters()

        if not self.last_layer:
            self.w_ru.reset_parameters()
            self.w_vq.reset_parameters()
            self.w_vk.reset_parameters()

    def forward(
            self,
            edge_index,
            s: torch.Tensor,
            v: torch.Tensor,
            dir_ij: torch.Tensor,
            r_ij: torch.Tensor,
            d_ij: torch.Tensor,
    ):
        s = self.layernorm(s)

        q = self.q_w(s).reshape(-1, self.num_heads, self.C_h // self.num_heads)
        k = self.k_w(s).reshape(-1, self.num_heads, self.C_h // self.num_heads)

        x = self.gamma_s(s)
        val = self.gamma_v(s)
        f_ij = r_ij
        r_ij_attn = self.wre_sigmak(r_ij)
        r_ij = self.w_ra(r_ij)

        # propagate_type: (x: Tensor, vec: Tensor, q:Tensor, k:Tensor, val:Tensor, r_ij: Tensor, r_ij_attn: Tensor, d_ij:Tensor, dir_ij: Tensor)
        su, vu = self.propagate(edge_index=edge_index, x=x, q=q, k=k, val=val,
                                vec=v, r_ij=r_ij, r_ij_attn=r_ij_attn, d_ij=d_ij, dir_ij=dir_ij)

        s = s + su
        v = v + vu

        if not self.last_layer:
            vec = v
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=dir_ij, f_ij=f_ij)
            df_ij = f_ij + df_ij
            return s, v, df_ij
        else:
            return s, v, f_ij

        return s, v

    def message(
            self,
            edge_index,
            x_i: torch.Tensor,
            x_j: torch.Tensor,
            q_i: torch.Tensor,
            k_j: torch.Tensor,
            val_j: torch.Tensor,
            vec_j: torch.Tensor,
            r_ij: torch.Tensor,
            r_ij_attn: torch.Tensor,
            d_ij: torch.Tensor,
            dir_ij: torch.Tensor,
            index: torch.Tensor, ptr: OptTensor,
            dim_size: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        r_ij_attn = r_ij_attn.reshape(-1, self.num_heads, self.C_h // self.num_heads)
        alpha_ij = (q_i * k_j * r_ij_attn).sum(dim=-1, keepdim=True)
        attn = softmax(alpha_ij, index, ptr, dim_size)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        sea = attn * val_j.reshape(-1, self.num_heads, (self.C_h * 3) // self.num_heads)
        sea = sea.reshape(-1, 1, self.C_h * 3)

        saa = r_ij.unsqueeze(1) * x_j * self.phi(d_ij.unsqueeze(-1).unsqueeze(-1))

        x = saa + sea

        o_s, o_d, o_v = torch.split(x, self.C_h, dim=-1)
        v_out = o_d * dir_ij[..., None] + o_v * vec_j
        return o_s, v_out

    def edge_update(self, vec_i, vec_j, d_ij, f_ij):
        w1 = self.w_vq(vec_i)
        w2 = self.w_vk(vec_j)

        w_dot = (w1 * w2).sum(dim=1)
        w_dot = torch.tanh(w_dot)

        df_ij = self.w_ru(f_ij) * w_dot
        return df_ij

    # noinspection PyMethodOverriding
    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x, vec

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class EquivariantFeedForward(nn.Module):
    def __init__(self, C_h: int, activation: Callable, epsilon: float = 1e-8,
                 weight_init=nn.init.xavier_uniform_, bias_init=nn.init.zeros_):
        super(EquivariantFeedForward, self).__init__()
        self.C_h = C_h

        PartialGamma = partial(Dense, weight_init=weight_init, bias_init=bias_init)

        self.gamma_m = nn.Sequential(
            PartialGamma(2 * C_h, C_h, activation=activation),
            PartialGamma(C_h, 3 * C_h, activation=None),
        )
        self.w_vu = PartialGamma(
            C_h, C_h, activation=None, bias=False
        )
        self.epsilon = epsilon

    def reset_parameters(self):
        self.w_vu.reset_parameters()
        if self.layernorm_:
            self.layernorm.reset_parameters()
        if self.vector_norm_:
            self.vec_layernorm.reset_parameters()
        for l in self.gamma_m:
            l.reset_parameters()

    def forward(self, s, v):

        v_vu = self.w_vu(v)
        v_size = v_vu
        v_inv = torch.sqrt(torch.sum(v_size ** 2, dim=-2, keepdim=True) + self.epsilon)

        inv_cat = [s, v_inv]
        ctx = torch.cat(inv_cat, dim=-1)
        x = self.gamma_m(ctx)

        m1, m2 = torch.split(x, self.C_h, dim=-1)

        s = s + m1
        v = v + (m2 * v_vu)

        return s, v


class Ginformer(nn.Module):
    def __init__(
            self,
            C_h: int = 128,
            n_interactions: int = 8,
            n_rbf: int = 20,
            cutoff_fn: Optional[Union[Callable, str]] = None,
            max_z: int = 100,
            epsilon: float = 1e-8,
            weight_init=nn.init.xavier_uniform_,
            bias_init=nn.init.zeros_,
            max_num_neighbors: int = 32,
            activation=F.silu,
            num_heads=8,
            attn_dropout=0.0,
            scale_edge=True,
            aggr="add",
    ):
        super(Ginformer, self).__init__()

        self.scale_edge = scale_edge

        self.C_h = self.hidden_dim = C_h
        self.n_interactions = n_interactions
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff

        self.distance = Distance(self.cutoff, max_num_neighbors=max_num_neighbors, loop=True)

        self.neighbor_embedding = NodeEmbedding(self.hidden_dim, n_rbf, self.cutoff, max_z).jittable()
        self.edge_embedding = EdgeEmbedding(n_rbf, self.hidden_dim).jittable()

        self.radial_basis = ExpNormalSmearing(cutoff=self.cutoff, n_rbf=n_rbf)
        self.embedding = nn.Embedding(max_z, C_h, padding_idx=0)

        self.interactions = nn.ModuleList([
            GIAI(
                C_h=self.C_h, activation=activation, aggr=aggr,
                weight_init=weight_init, bias_init=bias_init, cutoff=self.cutoff, epsilon=epsilon,
                num_heads=num_heads, dropout=attn_dropout,
                last_layer=(i == self.n_interactions - 1),
                scale_edge=scale_edge,
            ).jittable() for i in range(self.n_interactions)
        ])

        self.ff = nn.ModuleList([
            EquivariantFeedForward(
                C_h=self.C_h, activation=activation, epsilon=epsilon,
                weight_init=weight_init, bias_init=bias_init,
            ) for i in range(self.n_interactions)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        self.edge_embedding.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        for l in self.interactions:
            l.reset_parameters()
        for l in self.mixing:
            l.reset_parameters()

    def forward(self, inputs):
        atomic_numbers, pos, batch, edge_index = inputs.z, inputs.pos, inputs.batch, inputs.edge_index

        s = self.embedding(atomic_numbers)[:]

        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
        edge_attr = self.radial_basis(edge_weight)

        s = self.neighbor_embedding(atomic_numbers, s, edge_index, edge_weight, edge_attr)
        edge_attr = self.edge_embedding(edge_index, edge_attr, s)
        mask = edge_index[0] != edge_index[1]
        dist = torch.norm(edge_vec[mask], dim=1).unsqueeze(1)
        edge_vec[mask] = edge_vec[mask] / dist
        equi_dim = 3

        ss = s.shape
        v = torch.zeros((ss[0], equi_dim, ss[1]), device=s.device)
        s.unsqueeze_(1)
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.ff)):
            s, v, edge_attr = interaction(edge_index, s, v, dir_ij=edge_vec, r_ij=edge_attr,
                                          d_ij=edge_weight)
            s, v = mixing(s, v)

        s = s.squeeze(1)
        return s, v
