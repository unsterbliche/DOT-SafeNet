import torch.utils.data
from torch_geometric.data import Dataset
# from torch.utils.data import Dataset
import torch
import pandas as pd
from torch_geometric.data import Data
import pickle
import torch.utils.data
from copy import deepcopy
import numpy as np

# 0~5, 5~10, 10~15,15~20, 20~25, 25~30, 30~35, 35~40, 40~45, 45~50
# 50~60, 60~70, 70~80, 80~90, 90~100, 100~125, 125~150, 150~200, 200~250, 250~300, 300~350, 350~400, 400~450, 450~500
# 500~600,600~700, 700~800, 800~900, 900~1000, 1000~1250, 1250~1500, 1500~2000
# 2000~2500,2500~3000,3000~3500,3500~4000,4000~4500,4500~5000
# 5000~6000,6000~7000,7000~8000,8000~9000,9000~10000,10000~20000,20000~30000
# 构建上述dose字典
# dose_dict = {
#     '0~5': 0,'5~10': 1,'10~15': 2,'15~20': 3,'20~25': 4,'25~30': 5,'30~35': 6,'35~40': 7,'40~45': 8,'45~50': 9,
#     '50~60': 10,'60~70': 11,'70~80': 12,'80~90': 13,'90~100': 14,'100~125': 15,'125~150': 16,'150~200': 17,'200~250': 18,'250~300': 19,
#     '300~350': 20,'350~400': 21,'400~450': 22,'450~500': 23,'500~600': 24,'600~700': 25,'700~800': 26,'800~900': 27,'900~1000': 28,'1000~1250': 29,
#     '1250~1500': 30,'1500~2000': 31,'2000~2500': 32,'2500~3000': 33,'3000~3500': 34,'3500~4000': 35,'4000~4500': 36,'4500~5000': 37,
#     '5000~6000': 38,'6000~7000': 39,'7000~8000': 40,'8000~9000': 41,'9000~10000': 42,'10000~20000': 43,'20000~30000': 44
# }
# dose_dict = {
#     "0":[0,5],"1":[5,10],"2":[10,15],"3":[15,20],"4":[20,25],"5":[25,30],"6":[30,35],"7":[35,40],"8":[40,45],"9":[45,50],
#     "10":[50,60],"11":[60,70],"12":[70,80],"13":[80,90],"14":[90,100],"15":[100,125],"16":[125,150],"17":[150,200],"18":[200,250],"19":[250,300],
#     "20":[300,350],"21":[350,400],"22":[400,450],"23":[450,500],"24":[500,600],"25":[600,700],"26":[700,800],"27":[800,900],"28":[900,1000],"29":[1000,1250],
#     "30":[1250,1500],"31":[1500,2000],"32":[2000,2500],"33":[2500,3000],"34":[3000,3500],"35":[3500,4000],"36":[4000,4500],"37":[4500,5000],"38":[5000,6000],"39":[6000,7000],
#     "40":[7000,8000],"41":[8000,9000],"42":[9000,10000],"43":[10000,20000],"44":[20000,30000]
# }

dose_dict = {"0":[0,5],"1":[5,10],"2":[10,15],"3":[15,20],"4":[20,25],"5":[25,30],"6":[30,35],"7":[35,40],"8":[40,45],"9":[45,50],
             "10":[50,55],"11":[55,60],"12":[60,65],"13":[65,70],"14":[70,75],"15":[75,80],"16":[80,85],"17":[85,90],"18":[90,95],"19":[95,105],
             "20":[105,110],"21":[110,120],"22":[120,130],"23":[130,140],"24":[140,150],"25":[150,160],"26":[160,170],"27":[170,180],"28":[180,190],"29":[190,200],
             "30":[200,210],"31":[210,220],"32":[220,230],"33":[230,240],"34":[240,250],"35":[250,260],"36":[260,270],"37":[270,280],"38":[280,290],"39":[290,300],
             "40":[300,320],"41":[320,340],"42":[340,360],"43":[360,380],"44":[380,400],"45":[400,420],"46":[420,440],"47":[440,460],"48":[460,480],"49":[480,500],
             "50":[500,550],"51":[550,600],"52":[600,650],"53":[650,700],"54":[700,750],"55":[750,800],"56":[800,850],"57":[850,900],"58":[900,950],"59":[950,1000],
             "60":[1000,1100],"61":[1100,1200],"62":[1200,1300],"63":[1300,1400],"64":[1400,1500],"65":[1500,1600],"66":[1600,1700],"67":[1700,1800],"68":[1800,1900],"69":[1900,2000],
             "70":[2000,2200],"71":[2200,2400],"72":[2400,2600],"73":[2600,2800],"74":[2800,3000],"75":[3000,3200],"76":[3200,3400],"77":[3400,3600],"78":[3600,3800],"79":[3800,4000],
             "80":[4000,4500],"81":[4500,5000],"82":[5000,5500],"83":[5500,6000],"84":[6000,6500],"85":[6500,7000],"86":[7000,7500],"87":[7500,8000],"88":[8000,8500],"89":[8500,9000],
             "90":[9000,10000],"91":[10000,20000],"92":[20000,30000]
             }

def get_dose_label(dose):
    for key, value in dose_dict.items():
        if dose >= value[0] and dose < value[1]:
            int_key= int(key)
            return torch.tensor([int_key])

class MotifMoleculeDataset_dose(Dataset):
    def __init__(self, sequence_data, mol_obj, device='cpu', cache_transform=True):
        super(MotifMoleculeDataset_dose, self).__init__()

        if isinstance(sequence_data,pd.core.frame.DataFrame):
            self.pairs = sequence_data
        elif isinstance(sequence_data,str):
            self.pairs = pd.read_csv(sequence_data)
        else:
            raise Exception("provide dataframe object or csv path")
        
        # dose = self.pairs['dose'].apply(get_dose_label)
        self.pairs['dose_label'] = self.pairs['dose'].apply(get_dose_label)
        
        ## MOLECULES
        if isinstance(mol_obj, dict):
            self.mols = mol_obj
        elif isinstance(mol_obj, str):
            with open(mol_obj, 'rb') as f:
                self.mols = pickle.load(f)
        else:
            raise Exception("provide dict mol object or pickle path")


        self.device = device
        self.cache_transform = cache_transform

        if self.cache_transform:
            for _, v in self.mols.items(): #字典套字典，{seq1:{},seq2:{},...}
                v['atom_idx'] = v['atom_idx'].long().view(-1, 1)
                v['atom_feature'] = v['atom_feature'].float()
                # adj = v['bond_feature_adj'].long() #获取邻接矩阵
                # mol_edge_index =  adj.nonzero(as_tuple=False).t().contiguous() #获取非零元素索引
                # v['atom_edge_index'] = mol_edge_index
                v['atom_edge_index'] = v['edge_indices']
                v['atom_edge_attr'] = v['bond_feature'].float() #获取原子边特征（原子之间的键类型，单键、双键、三键）
                # v['atom_edge_attr'] = adj[mol_edge_index[0], mol_edge_index[1]].long() #获取原子边特征（原子之间的键类型，单键、双键、三键）
                v['atom_num_nodes'] = v['atom_idx'].shape[0]

                ## Clique
                v['x_clique'] = v['x_clique'].long().view(-1, 1) # clique类型编码
                v['clique_num_nodes'] = v['x_clique'].shape[0]
                v['tree_edge_index'] = v['tree_edge_index'].long()
                v['atom2clique_index'] = v['atom2clique_index'].long()


    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return self.__len__()
    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        # Extract data
        mol_key = self.pairs.loc[idx,'smiles']
        mol_dose_label = self.pairs.loc[idx,'dose_label']
        try: 
            reg_y = self.pairs.loc[idx,'activity']  # 列名
            reg_y = torch.tensor(reg_y).float()
            # reg_y_pK = self.pairs.loc[idx,'pK']
            # reg_y_pAC50 = self.pairs.loc[idx,'pAC50']
            # reg_y_pK = torch.tensor(reg_y_pK).float()
            # reg_y_pAC50 = torch.tensor(reg_y_pAC50).float()
        except KeyError:
            reg_y = None
            # reg_y = None
        

        try: 
            cls_y = self.pairs.loc[idx,'classification_label'] 
            cls_y = torch.tensor(cls_y).float()
        except KeyError:
            cls_y = None
        
        try: 
            mcls_y = self.pairs.loc[idx,'multiclass_label'] 
            mcls_y = torch.tensor(mcls_y + 1).float()
        except KeyError:
            mcls_y = None
            
        mol = self.mols[mol_key]
        
        ## PROT
        if self.cache_transform:
            ## atom
            mol_x = mol['atom_idx']
            mol_x_feat = mol['atom_feature']
            mol_edge_index  = mol['atom_edge_index']
            mol_edge_attr = mol['atom_edge_attr']
            mol_num_nodes = mol['atom_num_nodes']

            ## Clique
            mol_x_clique = mol['x_clique']
            clique_num_nodes = mol['clique_num_nodes']
            clique_edge_index = mol['tree_edge_index']
            atom2clique_index = mol['atom2clique_index']

        else:
            # MOL
            mol_x = mol['atom_idx'].long().view(-1, 1)
            mol_x_feat = mol['atom_feature'].float()
            # adj = mol['bond_feature_adj'].long()
            # mol_edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            mol_edge_index = mol['edge_indices']
            mol_edge_attr = mol['bond_feature'].float()
            # mol_edge_attr = adj[mol_edge_index[0], mol_edge_index[1]].long()
            mol_num_nodes = mol_x.shape[0]

            ## Clique
            mol_x_clique = mol['x_clique'].long().view(-1, 1)
            clique_num_nodes = mol_x_clique.shape[0]
            clique_edge_index = mol['tree_edge_index'].long()
            atom2clique_index = mol['atom2clique_index'].long()


        out = MultiGraphData(
                ## MOLECULE
                mol_x=mol_x, mol_x_feat=mol_x_feat, mol_edge_index=mol_edge_index,
                mol_edge_attr=mol_edge_attr, mol_num_nodes= mol_num_nodes,
                clique_x=mol_x_clique, clique_edge_index=clique_edge_index, atom2clique_index=atom2clique_index,
                clique_num_nodes=clique_num_nodes,
                ## Y output
                reg_y=reg_y,
                # reg_y_pK=reg_y_pK, reg_y_pAC50=reg_y_pAC50,
                cls_y=cls_y, mcls_y=mcls_y,
                ## keys
                mol_key = mol_key,
                ## dose
                mol_dose_label = mol_dose_label

        )

        return out

def maybe_num_nodes(index, num_nodes=None):
    # NOTE(WMF): I find out a problem here, 
    # index.max().item() -> int
    # num_nodes -> tensor
    # need type conversion.
    # return index.max().item() + 1 if num_nodes is None else num_nodes
    return index.max().item() + 1 if num_nodes is None else int(num_nodes)

def get_self_loop_attr(edge_index, edge_attr, num_nodes):
    r"""Returns the edge features or weights of self-loops
    :math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
    graph given by :attr:`edge_index`. Edge features of missing self-loops not
    present in :attr:`edge_index` will be filled with zeros. If
    :attr:`edge_attr` is not given, it will be the vector of ones.

    .. note::
        This operation is analogous to getting the diagonal elements of the
        dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_weight = torch.tensor([0.2, 0.3, 0.5])
        >>> get_self_loop_attr(edge_index, edge_weight)
        tensor([0.5000, 0.0000])

        >>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
        tensor([0.5000, 0.0000, 0.0000, 0.0000])
    """
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = torch.ones_like(loop_index, dtype=torch.float)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    full_loop_attr = loop_attr.new_zeros((num_nodes, ) + loop_attr.size()[1:])
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr



class MultiGraphData(Data):
    def __inc__(self, key, item, *args):
        if key == 'mol_edge_index':
            return self.mol_x.size(0)
        elif key == 'clique_edge_index':
            return self.clique_x.size(0)
        elif key == 'atom2clique_index':
            return torch.tensor([[self.mol_x.size(0)], [self.clique_x.size(0)]])
        # elif key == 'm2p_edge_index':
        #      return torch.tensor([[self.mol_x.size(0)], [self.prot_node_aa.size(0)]])
        # elif key == 'edge_index_p2m':
        #     return torch.tensor([[self.prot_node_s.size(0)],[self.mol_x.size(0)]])
        else:
            return super(MultiGraphData, self).__inc__(key, item, *args)

