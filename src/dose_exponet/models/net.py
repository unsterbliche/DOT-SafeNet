import torch
from torch_geometric.nn import global_add_pool
from torch.nn import Embedding, Linear
from torch_geometric.utils import degree, to_scipy_sparse_matrix, segregate_self_loops
import torch.nn.functional as F
from torch_scatter import scatter
import numpy as np
import scipy.sparse as sp
from models.layers import MLP, AtomEncoder, Drug_PNAConv,PosLinear, dropout_edge, attentivefp_Conv
from models.set_rep import SetRep
from copy import deepcopy
## for drug pooling
from models.drug_pool import graph_batch_to_set, attentive_MotifPool, MotifPool
## for cluster
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch, dropout_adj, degree, subgraph, softmax, add_remaining_self_loops
## for cluster
from torch_geometric.nn.norm import GraphNorm
import torch_geometric
from models.set_transformer import DeepSet, SetTransformer


EPS = 1e-15
import math

class MaskedAttentionSetReadout(torch.nn.Module):
    def __init__(self, hidden_channels, dropout=0.0):
        super().__init__()
        inner_channels = max(hidden_channels // 2, 1)
        self.score = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Linear(hidden_channels, inner_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(inner_channels, 1),
        )
        self.value = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.out = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.GELU(),
        )

    def forward(self, X, mask):
        scores = self.score(X).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1)
        values = self.value(X)
        pooled = torch.sum(values * attention.unsqueeze(-1), dim=1)
        return self.out(pooled), attention


class net(torch.nn.Module):
    def __init__(self, mol_deg, 
                 # MOLECULE
                 mol_in_channels=43, 
                 edge_dim=10,
                 hidden_channels=200,
                #  heads = 10,
                 num_layers =3,
                 clique_num_timesteps = 2,
                 num_timesteps = 2,
                 n_hidden_sets = 128,
                 n_elements = 64,
                 # interaction
                 total_layer=3,              
                 # training
                 dropout=0,
                 # objective
                 regression_head = True,
                 classification_head = False,
                 multiclassification_head = 0, ## or any number
                 dose_mode = False,
                 device='cuda:0',
                 set_layer = "None"): #set_layer可设置为"SetRep"、"transformer"、"deepset"
        super(net, self).__init__()
        self.total_layer = total_layer
        self.dose_mode = dose_mode

        if (regression_head or classification_head) == False:
            raise Exception('must have one objective')
        
        self.regression_head = regression_head
        self.classification_head = classification_head
        self.multiclassification_head = multiclassification_head
        self.hidden_channels = hidden_channels
        self.clique_num_timesteps = clique_num_timesteps
        self.dropout = dropout
        # self.heads = heads
        # self.dropout_attn_score = dropout_attn_score
        
        # MOLECULE IN FEAT
        self.atom_type_encoder = Embedding(20,hidden_channels) #原子类型数量是20，模型可以处理20种不同的原子类型（20是怎么得来的？可以更加原子类型数目修改？）
        # 即使某些类别没有出现在数据中，嵌入层也会分配一个表征，虽然这些表征在训练过程中可能未被更新。结合前述数据处理，这里原子的类型应该只对应10类
        self.atom_feat_encoder = MLP([mol_in_channels, hidden_channels * 2, hidden_channels], out_norm=True) 


        # Clique IN feat
        self.clique_encoder = Embedding(4, hidden_channels) #簇的类型有四种(单原子、环、键和桥接化合物)

        # dose embedding
        self.dose_embedding = Embedding(93, hidden_channels) #45种剂量,93种剂量

    
        ### MOLECULE and PROTEIN
        self.mol_convs = torch.nn.ModuleList()

        self.mol_pools = torch.nn.ModuleList()

        self.mol_gn2 = torch.nn.ModuleList()
        self.clique_gn2 = torch.nn.ModuleList()
        self.clique_pools = torch.nn.ModuleList()

        

        self.total_layer = total_layer
        self.set_layer = set_layer

        for idx in range(total_layer):
            # 加入残差网络
            self.mol_convs.append(attentivefp_Conv(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                edge_dim=edge_dim,
                num_layers=num_layers,
                # num_timesteps: int,
                dropout=self.dropout,
            ))


            self.clique_pools.append(MotifPool(hidden_channels, hidden_channels, clique_num_timesteps, self.dropout))
            self.mol_pools.append(attentive_MotifPool(hidden_channels, hidden_channels, num_timesteps, self.dropout))
            self.mol_gn2.append(GraphNorm(hidden_channels)) #使用GraphNorm的GNN比使用其他规范化方法的GNN收敛更快。此外，GraphhNorm还改进了GNN的泛化能力
            self.clique_gn2.append(GraphNorm(hidden_channels))
        
        if self.set_layer == "SetRep" or self.set_layer == "atom_setrep":
            self.setrep = SetRep(n_hidden_sets, n_elements, hidden_channels, hidden_channels)
        elif self.set_layer == "transformer":
            self.setrep = SetTransformer(
                hidden_channels, hidden_channels, 1
            )
        elif self.set_layer == "deepset":
            self.setrep = DeepSet(
                hidden_channels, hidden_channels, 1
            )
        elif self.set_layer == "masked_attn":
            self.setrep = MaskedAttentionSetReadout(hidden_channels, self.dropout)
        else:
            print("Set layer not implemented")
            # raise ValueError(f"Set layer '{self.set_layer}' not implemented.")
        
        # self.mol_out= MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)
       
        # self.clique_gn.append(GraphNorm(hidden_channels))


        if self.regression_head:
            if self.dose_mode == False:
                self.reg_out = MLP([hidden_channels, hidden_channels//2, 1]) 
            else:
                self.reg_out = MLP([hidden_channels*2, hidden_channels, 1]) 
        if self.classification_head:
            self.cls_out = MLP([hidden_channels, hidden_channels//2, 1]) 
        if self.multiclassification_head:
            self.mcls_out = MLP([hidden_channels, hidden_channels//2, multiclassification_head]) 

        
        self.device = device

    def reset_parameters(self):
        self.atom_feat_encoder.reset_parameters()
        self.dose_embedding.reset_parameters()

        
        for idx in range(self.total_layer):
            self.mol_convs[idx].reset_parameters()

            self.clique_pools[idx].reset_parameters()

            self.mol_pools[idx].reset_parameters()

            self.mol_gn2[idx].reset_parameters()

            self.clique_gn2[idx].reset_parameters()


        if self.regression_head:
            self.reg_out.reset_parameters()
        if self.classification_head:
            self.cls_out.reset_parameters()
        if self.multiclassification_head:
            self.mcls_out.reset_parameters()


    def forward(self,
                # Molecule
                mol_x, mol_x_feat, bond_x, atom_edge_index, 
                clique_x, clique_edge_index, atom2clique_index, # drug cliques
                # Mol-Protein Interaction batch
                mol_batch=None, clique_batch=None,
                # dose
                mol_dose_label=None,
                ## only if you're interested in clustering algorithm 
                save_cluster = False):
        # Init variables        
        reg_pred = None
        cls_pred = None
        mcls_pred = None
        # mol_pool_feat = []

       
        # MOLECULE Featurize
        atom_x = self.atom_type_encoder(mol_x.squeeze()) + self.atom_feat_encoder(mol_x_feat) # mol_x是原子类型序号（atom_idx）, mol_x_feat是原子特征（atom_feature）
        # mol_x和mol_x_feat是按照原子序号的顺序存入的（从小到大）
        # Clique Featurize
        clique_x = self.clique_encoder(clique_x.squeeze())

        # atom_score = []
        # MOLECULE-PROTEIN Layers
        for idx in range(self.total_layer):
            atom_x = self.mol_convs[idx](atom_x, bond_x, atom_edge_index) # 图卷积神经网络卷积过后的原子表征
            # atom_x是原子表征, bond_x是边表征, atom_edge_index是边索引
            atom_x = self.mol_gn2[idx](atom_x, mol_batch)
            
            clique_x = self.clique_pools[idx](atom_x, clique_x, atom2clique_index)

            # clique_x = self.clique_gn2[idx](clique_x, clique_batch)
            
            ## 消融实验！！
            clique_x = self.mol_pools[idx](clique_x, atom2clique_index, mol_batch, clique_batch, clique_edge_index)
            # # clique_x是簇的类别表征 atom2clique_index是二维数组，表示分子中的原子分配到哪个簇
        
        # # clique_x = self.clique_gn(clique_x, clique_batch)
        if self.set_layer == "None":
            # print("Not use set layer!!!!!")
            mol_pool_feat = global_add_pool(clique_x, clique_batch) #这个不是tensor吗？clique_x尺寸是clique_batch*hidden_channels（1040*300），clique_batch是一个batch内clique的个数（1040）
            attention_dict = {
                    'drug_atom_index':mol_batch,
                    'drug_clique_index':clique_batch,
                    'mol_feature': mol_pool_feat,
                    'motif_attention': None,
                }
        elif self.set_layer == "SetRep" or self.set_layer == "atom_setrep" or self.set_layer == "deepset" or self.set_layer == "transformer" or self.set_layer == "masked_attn": #"transformer"、"deepset"
            # print("Use set layer!!!!!")
            # t = graph_batch_to_set(atom_x, mol_batch, self.hidden_channels)
            # 消融实验！！
            if self.set_layer == "atom_setrep":
                t = graph_batch_to_set(atom_x, mol_batch, self.hidden_channels)
                counts = torch.bincount(mol_batch)
            else:
                t = graph_batch_to_set(clique_x, clique_batch, self.hidden_channels) # 64*29*300
                counts = torch.bincount(clique_batch)
            if self.set_layer == "masked_attn":
                positions = torch.arange(t.size(1), device=t.device).unsqueeze(0)
                mask = positions < counts.to(t.device).unsqueeze(1)
                mol_pool_feat, attention_scores = self.setrep(t, mask)
            else:
                mol_pool_feat, attention_scores = self.setrep(t) # mol_pool_feat是64*300，attention_scores是64*29
            
            # 按照clique_batch中每个数字的个数挑选每行保留的前几个数值
            # counts = torch.bincount(mol_batch)
            # 按counts取值
            attention_scores_ = [attention_scores[i, :counts[i]] for i in range(attention_scores.size(0))]
            # # # atteion_scores_是list，每个元素是tensor，每个tensor做softmax
            # attentions = [F.softmax(att, dim=0) for att in attention_scores_]
            # # # # print("attentions", attentions)
            # attention_scores_o = torch.cat(attentions, dim=0)
            attention_scores_o = torch.cat(attention_scores_, dim=0)

            attention_dict = {
                    'drug_atom_index':mol_batch,
                    'drug_clique_index':clique_batch,
                    'mol_feature': mol_pool_feat,
                    'motif_attention': attention_scores_o,
                }
                    

        if self.dose_mode == True:
            # mol_pool_feat = mol_pool_feat + self.dose_embedding(mol_dose_label)
            dose_embedding = self.dose_embedding(mol_dose_label)
            mol_pool_feat_normalized = F.normalize(mol_pool_feat, p=2, dim=1)
            dose_embedding_normalized = F.normalize(dose_embedding, p=2, dim=1)
            # mol_pool_feat_normalized = mol_pool_feat
            # dose_embedding_normalized = dose_embedding
            # print('mol_pool_feat', mol_pool_feat[0], dose_embedding[0])
            # mol_pool_feat = mol_pool_feat_normalized + dose_embedding_normalized*10000
            mol_pool_feat = torch.cat((mol_pool_feat_normalized, dose_embedding_normalized), dim=1)
            
        
        
        if self.regression_head:
            reg_pred = self.reg_out(mol_pool_feat)
            # reg_pred_pK = self.reg_out_pK(mol_prot_feat)
            # reg_pred_pAC50 = self.reg_out_pAC50(mol_prot_feat)
        if self.classification_head:
            cls_pred = self.cls_out(mol_pool_feat)
        if self.multiclassification_head:
            mcls_pred = self.mcls_out(mol_pool_feat)
        
   
        # attention_dict = {
        #     'drug_atom_index':mol_batch,
        #     'drug_clique_index':clique_batch,
        #     'mol_feature': mol_pool_feat,
        # }
        
        return reg_pred, cls_pred, mcls_pred, attention_dict
    
    def temperature_clamp(self):
        pass
        # with torch.no_grad():
        #     for m in self.cluster:
        #         m.logit_scale.clamp_(0, math.log(100))
    
    def connect_mol_prot(self, mol_batch, prot_batch):
        mol_num_nodes = mol_batch.size(0)
        prot_num_nodes = prot_batch.size(0)
        mol_adj = mol_batch.reshape(-1, 1).repeat(1, prot_num_nodes)
        pro_adj = prot_batch.repeat(mol_num_nodes, 1)

        m2p_edge_index = (mol_adj == pro_adj).nonzero(as_tuple=False).t().contiguous()

        return m2p_edge_index
    
    def freeze_backbone_optimizers(self, finetune_module, weight_decay, learning_rate, betas, eps, amsgrad): ## only for fineTune Pretrain model
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()  
        whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, GraphNorm, PosLinear)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...

                ################## THIS BLOCK TO FREEZE NOT FINE TUNED LAYERS ##################
                if not any([mn.startswith(name) for name in finetune_module]):
                    p.requires_grad = False
                    continue
                else:
                    p.requires_grad = True
                    print(fpn,' will be finetuned')
                ################## THIS BLOCK TO FREEZE NOT FINE TUNED LAYERS ##################                


                if pn.endswith('bias') or pn.endswith('mean_scale'):# or pn.endswith('logit_scale'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                    # if mn.startswith('cluster'):
                    #     print(mn, 'not decayed!')
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)

        return optimizer
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, amsgrad):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()  
        # whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear)
        if self.set_layer == "SetRep":
            whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear, attentivefp_Conv, SetRep, attentive_MotifPool, MotifPool)
        elif self.set_layer == "atom_setrep":
            whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear, attentivefp_Conv, SetRep, attentive_MotifPool, MotifPool)
        elif self.set_layer == "transformer":
            whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear, attentivefp_Conv, SetTransformer, attentive_MotifPool, MotifPool)
        elif self.set_layer == "deepset":
            whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear, attentivefp_Conv, DeepSet, attentive_MotifPool, MotifPool)
        elif self.set_layer == "masked_attn":
            whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear, attentivefp_Conv, MaskedAttentionSetReadout, attentive_MotifPool, MotifPool)
        else:
            whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear, attentivefp_Conv, attentive_MotifPool, MotifPool)
        # whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.dense.linear.Linear, attentivefp_Conv, SetRep, attentive_MotifPool)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, GraphNorm, PosLinear)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                # if pn.endswith('bias') or pn.endswith('mean_scale'):# or pn.endswith('logit_scale'):
                if pn.endswith('bias') or pn.endswith('mean_scale'):# or pn.endswith('logit_scale'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                    # if mn.startswith('cluster'):
                    #     print(mn, 'not decayed!')
                elif pn.endswith('bias_ih') or pn.endswith('bias_hh') or pn.endswith('I') or pn.endswith('S'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                    # if mn.startswith('cluster'):

                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn) #本来是no_decay.add(fpn)
                elif pn.endswith('weight_ih') or pn.endswith('weight_hh') or pn.endswith('Wc'):
                    decay.add(fpn)
                elif pn.endswith('att_l') or pn.endswith('att_r') or pn.endswith('att_src') or pn.endswith('att_dst'):
                    decay.add(fpn)
                # elif pn.endswith('weight_ih') or pn.endswith('weight_hh') or pn.endswith('Wc'):
                #     decay.add(fpn)
                # elif pn.endswith('att_l') or pn.endswith('att_r') or pn.endswith('att_src') or pn.endswith('att_dst'):
                #     decay.add(fpn)


        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad)
        # optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate,
        #                         weight_decay=weight_decay)

        return optimizer
    

def _rbf(D, D_min=0., D_max=1., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D = torch.where(D < D_max, D, torch.tensor(D_max).float().to(device) )
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


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

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask
    
    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p
    
    ## ensure no graph is totally dropped out
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
