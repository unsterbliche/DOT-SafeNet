import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_max_pool as gmp, global_mean_pool as gep
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
import warnings
import time
warnings.filterwarnings("ignore")


class CPI_classification(torch.nn.Module):
    def __init__(self, device, emb_size = 1280, max_length=1024, n_output=2, modulator_emb_dim=256, ppi_emb_dim=512, 
                 h_dim=512, n_heads=2, num_features_mol=78,dropout=0.1):
        super(CPI_classification, self).__init__()
        print('CPI_classification model Loaded..')
        self.device = device
        self.skip = 1
        self.n_output = n_output
        self.max_length = max_length
        self.emb_size = emb_size
        self.modulator_emb_dim = modulator_emb_dim
        # compounds network
        self.mol_conv1 = SAGEConv(num_features_mol, num_features_mol*2,'mean') #num_features_mol: The feature dimension of molecules (78 in this case)
        self.mol_conv2_f = SAGEConv(num_features_mol*2,num_features_mol*2,'mean')
        self.mol_conv3_f = SAGEConv(num_features_mol*2,num_features_mol*4,'mean')
        self.mol_fc_g1 = nn.Linear(num_features_mol*4, modulator_emb_dim)
        self.dropout = nn.Dropout(dropout)
        # proteins network
        self.prot_rnn = nn.LSTM(self.emb_size, ppi_emb_dim, 2)
        self.relu = nn.LeakyReLU()
        
        ##### bilinear attention #####
        self.bcn = weight_norm(
            BANLayer(v_dim=modulator_emb_dim, q_dim=ppi_emb_dim, h_dim=h_dim, h_out=n_heads, k=3),
            name='h_mat', dim=None) 
        
        self.fc = nn.Sequential(nn.Linear(h_dim, 256),nn.LeakyReLU(),nn.Dropout(dropout),
                                nn.Linear(256, 128),nn.LeakyReLU(),nn.Dropout(dropout),
                                nn.Linear(128, self.n_output))


    def forward(self, data_mol, data_pro, data_pro_len):
        # protein network
        time_0 = time.time()
        pro_seq_lengths, pro_idx_sort = torch.sort(data_pro_len,descending=True)[::-1][1], torch.argsort(
            -data_pro_len) 
        pro_idx_unsort = torch.argsort(pro_idx_sort)
        data_pro = data_pro.index_select(0, pro_idx_sort)
        xt = nn.utils.rnn.pack_padded_sequence(data_pro, pro_seq_lengths.cpu(), batch_first=True) 

        xt,_ = self.prot_rnn(xt)
        xt = nn.utils.rnn.pad_packed_sequence(xt, batch_first=True, total_length=self.max_length)[0]
        xt = xt.index_select(0, pro_idx_unsort)
        
        # compound network
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_conv2_f(x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_conv3_f(x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_fc_g1(x)
        x = self.dropout(x)
  
        mol_num_nodes = torch.bincount(mol_batch)
        max_num_nodes = torch.max(mol_num_nodes)
        num_molecules = mol_batch[-1] + 1
        padded_atom_features = torch.zeros((num_molecules, max_num_nodes, 256), dtype=x.dtype, device=x.device)
        node_indices = torch.cat([torch.arange(n, device=x.device) for n in mol_num_nodes])
        len_node_indices = len(node_indices)
        padded_atom_features[mol_batch, node_indices, :] = x
        x_pad = padded_atom_features.to(self.device)
        prot_mask = (xt.sum(dim=-1) != 0).unsqueeze(1)
        mol_mask = (x_pad.sum(dim=-1) != 0).unsqueeze(-1)

        prot_mask = prot_mask.float() # (batch_size,1,max_length)
        mol_mask = mol_mask.float() # (batch_size,max_num_nodes,1)
        mask_q = torch.matmul(mol_mask,prot_mask) #mask shape:(batch_size,max_num_nodes,max_length)
        ##### bilinear attention #####
        comp_out = x_pad
        prot_out = xt
        output, att = self.bcn(comp_out, prot_out,mask=mask_q)
        output = self.fc(output)
        return output

