import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax

from torch_scatter import scatter
from torch_geometric.nn import global_add_pool
from models.layers import MLP, dropout_node
from models.set_rep import SetRep
from models.set_transformer import SetTransformer, DeepSet
from torch_geometric.utils import unbatch

from torch_geometric.nn.conv import MessagePassing, GATConv
from torch_geometric.nn.dense.linear import Linear
from models.pna import PNAConv, GATEConv
from torch.nn import GRUCell, Linear, Parameter

def graph_batch_to_set(X, batch, dim): #X是一个批次的图表征(由clique组成)，batch是一个批次的图，dim是图的clique特征维度
    # out_unbatched = unbatch(X, batch.batch) #解批次
    out_unbatched = unbatch(X, batch) #解批次

    batch_size = len(out_unbatched) #获取解批后的图的数量
    max_cardinality = max([b.shape[0] for b in out_unbatched]) #获取解批后的图的最大clique数目

    # X_set = torch.zeros((batch_size, max_cardinality, dim), device=batch.x.device) #创建全零张量
    X_set = torch.zeros((batch_size, max_cardinality, dim), device=X.device) #创建全零张量

    for i, b in enumerate(out_unbatched): #把解批后的图的clique特征填充到X_set中
        X_set[i, : b.shape[0], :] = b

    return X_set


# class MotifPool(torch.nn.Module):
#     def __init__(self, hidden_dim, heads, dropout_attn_score=0.1): 
#         super().__init__()
#         assert hidden_dim % heads == 0 

#         self.lin_proj = torch.nn.Linear(hidden_dim, hidden_dim) #线性层
#         hidden_dim = hidden_dim // heads
        
#         self.score_proj = torch.nn.ModuleList()
#         for _ in range(heads): 
#             self.score_proj.append( MLP([ hidden_dim, hidden_dim*2, 1]) ) #多头注意力评分层
        
#         self.heads = heads
#         self.hidden_dim = hidden_dim 
#         self.dropout_attn_score = dropout_attn_score

#     def reset_parameters(self):
#         self.lin_proj.reset_parameters()
#         for m in self.score_proj:
#             m.reset_parameters()

#     def forward(self, x, x_clique, atom2clique_index, mol_batch, clique_batch, clique_edge_index): # x是原子表征，x_clique是簇的类别的表征（簇的数目*簇表征），clique_edge_index是树中连接簇的边
#         row, col = atom2clique_index #原子编号及其对应的簇（簇编号）。x是原子表征，是按照序号从小到大放置的 可以根据索引取出原子表征
#         H = self.heads
#         C = self.hidden_dim
#         ## residual connection + atom2clique
#         score_atom = x.view(-1, H, C) #把原子特征reshape成一个batch_size内所有的原子数*heads*hidden_dim
#         score_atom = torch.cat([mlp(score_atom[:, i]) for i, mlp in enumerate(self.score_proj) ], dim=-1) #score_atom尺寸为 971*10
#         score_atom = F.dropout(score_atom, p=self.dropout_attn_score, training=self.training)
#         alpha_atom = softmax(score_atom, mol_batch) #计算注意力权重 mol_batch用于标记一个批次中每一原子所属的批次号。mol_batch大小就是一个批次中原子的数量，原子的编号就是0~batch_size-1
#         #最后是每个原子对应一个权重，所有原子的权重加起来是1

#         ## multihead aggregation of clique feature
#         # 原子表征*注意力权重获取簇表征
#         # 按照row取出对应的x和alpha_atom，然后按照col进行求和，得到簇的表征
#         x = x.view(-1, H, C) #把原子特征reshape成一个batch_size内所有的原子数*heads*hidden_dim
#         alpha_atom = alpha_atom.view(-1, H, 1) #把注意力权重reshape成一个batch_size内所有的原子数*heads*1
#         hx_clique = scatter(x[row, :, :] * alpha_atom[row, :, :], col, dim=0, dim_size=x_clique.size(0)) #x_clique尺寸为 971*300
#         hx_clique = hx_clique.view(-1, H * C)

#         x_clique = x_clique + F.relu(self.lin_proj(hx_clique)) #图中每一簇及其对应的表征（一个batch_size的所有簇）971*300

        
#         return x_clique

class MotifPool(torch.nn.Module):
    def __init__(self,
        hidden_channels: int,
        out_channels: int,
        # clique_dim: int,
        clique_num_timesteps: int,
        dropout: float = 0.0,): 
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_timesteps = clique_num_timesteps
        self.dropout = dropout
        # self.lin_proj = torch.nn.Linear(hidden_channels, hidden_channels) #线性层
        # self.gate_conv = GATEConv(hidden_channels, hidden_channels, clique_dim,
        #                           dropout)
        self.clique_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.clique_conv.explain = False  # Cannot explain global pooling.
        self.clique_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.clique_conv.reset_parameters()
        self.clique_gru.reset_parameters()
        # self.lin_proj.reset_parameters()
        self.lin.reset_parameters()
    
    def forward(self, x, x_clique, atom2clique_index): # x是原子表征，x_clique是簇的类别（和簇的数量一致），clique_edge_index是树中连接簇的边
        row, col = atom2clique_index #原子编号及其对应的簇（簇编号）。x是原子表征，是按照序号从小到大放置的 可以根据索引取出原子表征
        clique_atom = scatter(x[row], col, dim=0, dim_size=x_clique.size(0), reduce='sum')  # reduce='sum'，scatter 操作是对 x[row] 中的原子表征进行聚合，聚合到相应的簇（由 col 索引表示，在第一个维度上求和，得到簇的原子表征
        # #按照簇聚合原子特征 x[row]是按照原子序号取出要聚合的元素表征，col是要聚合到的簇，dim表示在第一个维度聚合，dim_size表示聚合张量的第一个维度(即簇的个数)，reduce表示聚合方式
        # 最后clique_atom是按照簇的顺序排列的（col从小到大）
        clique_atom = x_clique + F.relu(self.lin(clique_atom))
     
        for t in range(self.num_timesteps):
            h = F.elu_(self.clique_conv((x[row], clique_atom), atom2clique_index)) #x是原子表征，clique内原子算注意力，clique_atom是簇表征
            h = F.dropout(h, p=self.dropout, training=self.training)
            clique_atom = self.clique_gru(h, clique_atom).relu_() #h是原子表征，superatom是超原子表征
            # # 计算每个原子相对于supertom的权重
            # attention_weights = torch.matmul(x[row], superatom.unsqueeze(1)).squeeze(1) #x是原子表征，superatom是超原子表征

        

        clique_atom = F.dropout(clique_atom, p=self.dropout, training=self.training)
        clique_feat = self.lin(clique_atom) #超原子表征
       
        return clique_feat
    

class attentive_MotifPool(torch.nn.Module):
    def __init__(self,
        hidden_channels: int,
        out_channels: int,
        # edge_dim: int,
        num_timesteps: int,
        dropout: float = 0.0,): 
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        # self.edge_dim = edge_dim
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        # self.lin_proj = torch.nn.Linear(hidden_channels, hidden_channels) #线性层
        # self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
        #                           dropout)
        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_conv.explain = False  # Cannot explain global pooling.
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        # self.lin_proj.reset_parameters()
        self.lin.reset_parameters()
    
    def forward(self, x_clique, atom2clique_index, mol_batch, clique_batch, clique_edge_index): 
        # x_clique是簇的表征，clique_edge_index是树中连接簇的边（簇序号对簇序号）
        clique_center_atom = x_clique #1040*300
       
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv(clique_center_atom, clique_edge_index)) #x_clique是motif表征，clique_center_atom是超原子表征 1040*300
            h = F.dropout(h, p=self.dropout, training=self.training)
            clique_center_atom = self.mol_gru(h, clique_center_atom) #h是motif表征，clique_center_atom是超原子表征 1040*300
            # # 计算每个原子相对于supertom的权重
            # attention_weights = torch.matmul(x[row], superatom.unsqueeze(1)).squeeze(1) #x是原子表征，superatom是超原子表征

        

        clique_center_atom = F.dropout(clique_center_atom, p=self.dropout, training=self.training)
        clique_feat = self.lin(clique_center_atom) #超原子表征
       
        return clique_feat


# class attentive_MotifPool(torch.nn.Module):
#     def __init__(self,
#         hidden_channels: int,
#         out_channels: int,
#         # edge_dim: int,
#         num_timesteps: int,
#         dropout: float = 0.0,): 
#         super().__init__()

#         self.hidden_channels = hidden_channels
#         self.out_channels = out_channels
#         # self.edge_dim = edge_dim
#         self.num_timesteps = num_timesteps
#         self.dropout = dropout
#         # self.lin_proj = torch.nn.Linear(hidden_channels, hidden_channels) #线性层
#         # self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
#         #                           dropout)
#         self.mol_conv = GATConv(hidden_channels, hidden_channels,
#                                 dropout=dropout, add_self_loops=False,
#                                 negative_slope=0.01)
#         self.mol_conv.explain = False  # Cannot explain global pooling.
#         self.mol_gru = GRUCell(hidden_channels, hidden_channels)

#         self.lin = Linear(hidden_channels, out_channels)

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.mol_conv.reset_parameters()
#         self.mol_gru.reset_parameters()
#         # self.lin_proj.reset_parameters()
#         self.lin.reset_parameters()
    
#     def forward(self, x_clique, atom2clique_index, mol_batch, clique_batch, clique_edge_index): # x是原子表征，x_clique是簇的类别（和簇的数量一致），clique_edge_index是树中连接簇的边
#         row, col = atom2clique_index #原子编号及其对应的簇（簇编号）。x是原子表征，是按照序号从小到大放置的 可以根据索引取出原子表征
#         # # 设置一个超原子，基于注意力机制汇总节点特征得到簇表征
#         row_clique = torch.arange(x_clique.size(0), device=x_clique.device) #簇编号 1024个簇
#         edge_index = torch.stack([row_clique, clique_batch], dim=0) #簇编号和簇的类别 2*1024（一行是簇编号，一行是簇所属的batch批次号）
#         super_clique_atom = global_add_pool(x_clique, clique_batch).relu_()
#         # super_clique_atom = scatter(x_clique, col, dim=0, dim_size=x_clique.size(0), reduce='sum')  # reduce='sum'
#         # # #按照簇聚合原子特征 x[row]是按照原子序号取出要聚合的元素表征，col是要聚合到的簇，dim表示在第一个维度聚合，dim_size表示聚合张量的第一个维度(即簇的个数)，reduce表示聚合方式
#         # clique_center_atom = clique_center_atom + x_clique
#         # clique_center_atom = x_clique
       

#         for t in range(self.num_timesteps):
#             h = F.elu_(self.mol_conv((x_clique, super_clique_atom), edge_index)) #每个分子中的簇进行聚合 super_clique_atom（64*200）是一个分子中的超原子表征
#             h = F.dropout(h, p=self.dropout, training=self.training)
#             super_clique_atom = self.mol_gru(h, super_clique_atom).relu_() #h是原子表征，superatom是超原子表征
#             # # 计算每个原子相对于supertom的权重
#             # attention_weights = torch.matmul(x[row], superatom.unsqueeze(1)).squeeze(1) #x是原子表征，superatom是超原子表征

        

#         super_clique_atom = F.dropout(super_clique_atom, p=self.dropout, training=self.training)
#         drug_feat = self.lin(super_clique_atom) #超原子表征
       
#         return drug_feat