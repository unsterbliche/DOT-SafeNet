import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRUCell, Linear, Parameter
from models.pna import PNAConv, GATEConv

from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax, degree, subgraph, to_scipy_sparse_matrix, segregate_self_loops, add_remaining_self_loops
import numpy as np 
import scipy.sparse as sp


class GCNCluster(torch.nn.Module):
    def __init__(self, dims, out_norm=False, in_norm=False): #L=nb_hidden_layers
        super().__init__()
        list_Conv_layers = [ GCNConv(dims[idx-1], dims[idx]) for idx in range(1,len(dims)) ]
        self.Conv_layers = nn.ModuleList(list_Conv_layers)
        self.hidden_layers = len(dims) - 2

        self.out_norm = out_norm
        self.in_norm = in_norm

        if self.out_norm:
            self.out_ln = nn.LayerNorm(dims[-1])
        if self.in_norm:
            self.in_ln = nn.LayerNorm(dims[0])

    def reset_parameters(self):
        for idx in range(self.hidden_layers+1):
            self.Conv_layers[idx].reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x, edge_index):
        y = x
        # Input Layer Norm
        if self.in_norm:
            y = self.in_ln(y)

        for idx in range(self.hidden_layers):
            y = self.Conv_layers[idx](y, edge_index)
            y = F.relu(y)
        y = self.Conv_layers[-1](y, edge_index)

        if self.out_norm:
            y = self.out_ln(y)

        return y


class AtomEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(AtomEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(9):
            self.embeddings.append(torch.nn.Embedding(100, hidden_channels))

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)

        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])
        return out


class BondEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BondEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(3):
            self.embeddings.append(torch.nn.Embedding(10, hidden_channels))

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        out = 0
        for i in range(edge_attr.size(1)):
            out += self.embeddings[i](edge_attr[:, i])
        return out


class PosLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, init_value=0.2,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PosLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # center_value = init_value
        # lower_bound = center_value - center_value/10
        # upper_bound = center_value + center_value/10

        lower_bound = init_value/2
        upper_bound = init_value
        weight = nn.init.uniform_(torch.empty((out_features, in_features),**factory_kwargs), a=lower_bound, b=upper_bound)
        # weight = nn.init.kaiming_uniform_(torch.empty((out_features, in_features),**factory_kwargs), a=math.sqrt(5))
        weight = torch.abs(weight)
        self.weight = nn.Parameter(weight.log())
        # self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()



    def reset_parameters(self) -> None:
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # self.weight = torch.abs(self.weight).log()
        if self.bias is not None:
            nn.init.uniform_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight.exp(), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class MLP(nn.Module):

    def __init__(self, dims, out_norm=False, in_norm=False, bias=True): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear(dims[idx-1], dims[idx], bias=bias) for idx in range(1,len(dims)) ]
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.hidden_layers = len(dims) - 2

        self.out_norm = out_norm
        self.in_norm = in_norm

        if self.out_norm:
            self.out_ln = nn.LayerNorm(dims[-1])
        if self.in_norm:
            self.in_ln = nn.LayerNorm(dims[0])

    def reset_parameters(self):
        for idx in range(self.hidden_layers+1):
            self.FC_layers[idx].reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x):
        y = x
        # Input Layer Norm
        if self.in_norm:
            y = self.in_ln(y)

        for idx in range(self.hidden_layers):
            y = self.FC_layers[idx](y)
            y = F.relu(y)
        y = self.FC_layers[-1](y)

        if self.out_norm:
            y = self.out_ln(y)

        return y

class Drug_PNAConv(nn.Module):
    def __init__(self, mol_deg, hidden_channels, edge_channels,
                 pre_layers=2, post_layers=2,
                 aggregators=['sum', 'mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 num_towers=4,
                 dropout=0.1):
        super(Drug_PNAConv, self).__init__()

        self.bond_encoder = torch.nn.Embedding(5, hidden_channels)

        self.atom_conv = PNAConv(
            in_channels=hidden_channels, out_channels=hidden_channels,
            edge_dim=edge_channels, aggregators=aggregators,
            scalers=scalers, deg=mol_deg, pre_layers=pre_layers,
            post_layers=post_layers,towers=num_towers,divide_input=True,
        )
        self.atom_norm = torch.nn.LayerNorm(hidden_channels)

        self.dropout = dropout

    def reset_parameters(self):
        self.atom_conv.reset_parameters()
        self.atom_norm.reset_parameters()


    def forward(self, atom_x, bond_x, atom_edge_index):
        atom_in = atom_x
        bond_x = self.bond_encoder(bond_x.squeeze())
        atom_x = atom_in + F.relu(self.atom_norm(self.atom_conv(atom_x, atom_edge_index, bond_x)))
        atom_x = F.dropout(atom_x, self.dropout, training=self.training)

        return atom_x


# 参考链接：https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/attentive_fp.html
class attentivefp_Conv(nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        # num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        # self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels) #200*200

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                  dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        # self.mol_conv = GATConv(hidden_channels, hidden_channels,
        #                         dropout=dropout, add_self_loops=False,
        #                         negative_slope=0.01)
        # self.mol_conv.explain = False  # Cannot explain global pooling.
        # self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        # self.mol_conv.reset_parameters()
        # self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, atom_x, bond_x, atom_edge_index):
        # x = atom_x, edge_index = atom_edge_index, edge_attr = bond_x
        """"""  # noqa: D419
        # Atom Embedding:
        atom_x = F.leaky_relu_(self.lin1(atom_x)) #400到200

        h = F.elu_(self.gate_conv(atom_x, atom_edge_index, bond_x))
        h = F.dropout(h, p=self.dropout, training=self.training) #atom_x是原子表征，h是对应的上下文表征，两个一起送入GRU层
        atom_x = self.gru(h, atom_x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(atom_x, atom_edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            atom_x = gru(h, atom_x).relu()

        # row = torch.arange(batch.size(0), device=batch.device)
        # edge_index = torch.stack([row, batch], dim=0)
        
        # out = global_add_pool(x, batch).relu_()
        # for t in range(self.num_timesteps):
        #     h = F.elu_(self.mol_conv((x, out), edge_index)) #残差连接
        #     h = F.dropout(h, p=self.dropout, training=self.training)
        #     out = self.mol_gru(h, out).relu_()

        # Predictor:
        x_out = F.dropout(atom_x, p=self.dropout, training=self.training)
        atom_x = self.lin2(x_out)
        return atom_x



def unbatch(src, batch, dim: int = 0):
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`

    Example:

        >>> src = torch.arange(7)
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)



def unbatch_edge_index(edge_index, batch):
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.

    :rtype: :class:`List[Tensor]`

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        >>> unbatch_edge_index(edge_index, batch)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]))
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def compute_connectivity(edge_index, batch):  ## for numerical stability (i.e. we cap inv_con at 100)

    edges_by_batch = unbatch_edge_index(edge_index, batch)

    nodes_counts = torch.unique(batch, return_counts=True)[1]

    connectivity = torch.tensor([nodes_in_largest_graph(e, n) for e, n in zip(edges_by_batch, nodes_counts)])
    isolation = torch.tensor([isolated_nodes(e, n) for e, n in zip(edges_by_batch, nodes_counts)])

    return connectivity, isolation


def nodes_in_largest_graph(edge_index, num_nodes):
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    num_components, component = sp.csgraph.connected_components(adj)

    _, count = np.unique(component, return_counts=True)
    subset = np.in1d(component, count.argsort()[-1:])

    return subset.sum() / num_nodes


def isolated_nodes(edge_index, num_nodes):
    r"""Find isolate nodes """
    edge_attr = None

    out = segregate_self_loops(edge_index, edge_attr)
    edge_index, edge_attr, loop_edge_index, loop_edge_attr = out

    mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    mask[edge_index.view(-1)] = 0

    return mask.sum() / num_nodes

# 图神经网络中的节点丢弃操作，以增强模型的鲁棒性。
# 在训练过程中，随机丢弃一些节点，可以防止模型过度依赖某些节点，从而提高模型的泛化能力。
def dropout_node(edge_index, p, num_nodes, batch, training):
    r"""Randomly drops nodes from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
        >>> edge_index
        tensor([[0, 1],
                [1, 0]])
        >>> edge_mask
        tensor([ True,  True, False, False, False, False])
        >>> node_mask
        tensor([ True,  True, False, False])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0: #直接返回所有节点和边，不进行任何丢弃
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask
    
    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p # 生成一个布尔掩码，表示哪些节点将被丢弃（node_mask为true的节点将被保留）
    
    ## ensure no graph is totally dropped out 确保图不全部丢弃
    batch_tf = global_add_pool(node_mask.view(-1,1),batch).flatten()
    unbatched_node_mask = unbatch(node_mask, batch)
    node_mask_list = []
    
    for true_false, sub_node_mask in zip(batch_tf, unbatched_node_mask):
        if true_false.item():
            node_mask_list.append(sub_node_mask)
        else:
            perm = torch.randperm(sub_node_mask.size(0))
            idx = perm[:1]
            sub_node_mask[idx] = True
            node_mask_list.append(sub_node_mask)
            
    node_mask = torch.cat(node_mask_list)
    
    edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True)
    return edge_index, edge_mask, node_mask

def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Randomly drops edges from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    or index indicating which edges were retained, depending on the argument
    :obj:`force_undirected`.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        force_undirected (bool, optional): If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
            (default: :obj:`False`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor` or :class:`LongTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask = dropout_edge(edge_index)
        >>> edge_index
        tensor([[0, 1, 2, 2],
                [1, 2, 1, 3]])
        >>> edge_mask # masks indicating which edges are retained
        tensor([ True, False,  True,  True,  True, False])

        >>> edge_index, edge_id = dropout_edge(edge_index,
        ...                                    force_undirected=True)
        >>> edge_index
        tensor([[0, 1, 2, 1, 2, 3],
                [1, 2, 3, 0, 1, 2]])
        >>> edge_id # indices indicating which edges are retained
        tensor([0, 2, 4, 0, 2, 4])
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask

