
import sys, os
import time
import torch
import json, pickle
import numpy as np
import pandas as pd
from math import sqrt
import networkx as nx
from collections import OrderedDict
from rdkit import Chem
from scipy import stats
from rdkit.Chem import MolFromSmiles
from torch_geometric import data as DATA
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Dataset,InMemoryDataset, DataLoader, Batch
import math
import pandas as pd
import os.path as osp


# # initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset=None,
                 xd=None, y_pK=None, y_pAC50=None, y_pK_mask=None, y_pAC50_mask=None, transform=None,
                 pre_transform=None, smile_graph=None, target_key=None, target_graph=None):

        self.root = root
        self.dataset = dataset
        # print('self.dataset', self.dataset)
        self.xd = xd
        self.y_pK = y_pK
        self.y_pAC50 = y_pAC50
        self.y_pK_mask = y_pK_mask
        self.y_pAC50_mask = y_pAC50_mask
        self.smile_graph = smile_graph
        self.target_key = target_key
        self.target_graph = target_graph
        self._process()
        super(DTADataset, self).__init__(root, transform, pre_transform)
        # self.process(xd, target_key, y_pK, y_pAC50, y_pK_mask, y_pAC50_mask, smile_graph, target_graph)
    
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def _download(self):
        pass
    
    # @property
    # def processed_dir(self):
    #     return osp.join(self.root, f'/smiles_mol_pro{self.dataset}')
    def _process(self):
        self.processed_dir_pickle = self.root + f'/smiles_mol_dict_{self.dataset}.pickle'
        # print("self.root is:", self.root)
        # print("root is:",self.processed_dir_pickle)
        if not os.path.exists(self.processed_dir_pickle):
            # os.makedirs(self.processed_dir)
            self.smiles_mol_dict = self.process(self.xd, self.target_key, self.y_pK, self.y_pAC50, self.y_pK_mask, self.y_pAC50_mask, self.smile_graph)
        else:
            self.smiles_mol_dict = pickle.load(open(self.processed_dir_pickle, 'rb'))

    def process(self, xd, target_key, y_pK, y_pAC50, y_pK_mask, y_pAC50_mask, smile_graph):
        assert (len(xd) == len(target_key) and len(xd) == len(y_pK)), 'These lists must be the same length!'
        data_dict_mol = {}
        data_len = len(xd)
        for i in range(data_len):
            smiles = xd[i]
            tar_key = target_key[i]
            labels_pK = y_pK[i]
            labels_pAC50 = y_pAC50[i]
            labels_pK_mask = y_pK_mask[i]
            labels_pAC50_mask = y_pAC50_mask[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            GCNData_mol = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y_pK=torch.FloatTensor([labels_pK]),
                                    y_pAC50=torch.FloatTensor([labels_pAC50]),
                                    y_pK_mask=torch.FloatTensor([labels_pK_mask]),
                                    y_pAC50_mask=torch.FloatTensor([labels_pAC50_mask]))
            GCNData_mol.target_key = tar_key
            GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))
            data_dict_mol[smiles + tar_key] = GCNData_mol
        # Save data_dict_mol to the pickle file
        with open(self.processed_dir_pickle, 'wb+') as f:
            pickle.dump(data_dict_mol, f)
        return data_dict_mol

    def __len__(self):
        return len(self.xd)

    def __getitem__(self, idx):
        smiles = self.xd[idx]
        tar_key = self.target_key[idx]
        pair_key = smiles + tar_key
        data_mol = self.smiles_mol_dict[pair_key]
        target_features, target_size= self.target_graph[tar_key]
        data_pro = target_features
        data_pro_len = target_size
        return data_mol, data_pro, data_pro_len
        # return self.data_mol[idx], self.data_pro[idx], self.data_pro_len[idx]

# #prepare the protein and drug pairs
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = default_collate([data[1] for data in data_list])
    batchC = default_collate([data[2] for data in data_list])
    return batchA, batchB, batchC



# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = []
    loss_fn = torch.nn.MSELoss(reduction='none')

    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        data_pro_len = data[2].to(device)
        max_data_pro_len = data_pro_len.max().item()
        new_data_pro = data_pro[:, :max_data_pro_len, :]
        optimizer.zero_grad()
        # output, out_gnn, att = model(data_mol, data_pro, data_pro_len)
        output, out_gnn, att = model(data_mol, new_data_pro, data_pro_len)
        #Output two values: the first one is the pK value, and the second one is the pAC50 value
        output_pK = output[:, 0]
        output_pAC50 = output[:, 1]
        loss_pK = loss_fn(output_pK.view(-1,1), data_mol.y_pK.view(-1, 1).float().to(device))
        y_pK_mask = data_mol.y_pK_mask.view(-1, 1).float().to(device)
        loss_pK = torch.sum(loss_pK[y_pK_mask.bool()]) / (torch.sum(y_pK_mask) + 1e-9)
        loss_pAC50 = loss_fn(output_pAC50.view(-1,1), data_mol.y_pAC50.view(-1, 1).float().to(device))
        y_pAC50_mask = data_mol.y_pAC50_mask.view(-1, 1).float().to(device)
        loss_pAC50 = torch.sum(loss_pAC50[y_pAC50_mask.bool()]) / (torch.sum(y_pAC50_mask) + 1e-9)
        loss = (loss_pK + loss_pAC50)/2
        loss.backward() #44
        optimizer.step()
        train_loss.append(loss.item())

    train_loss = np.average(train_loss)
    return train_loss

def evaluate(model, device, loader):
    model.eval()
    total_preds_pK = torch.Tensor()
    total_labels_pK = torch.Tensor()
    total_labels_pK_mask = torch.Tensor()
    total_preds_pAC50 = torch.Tensor()
    total_labels_pAC50 = torch.Tensor()
    total_labels_pAC50_mask = torch.Tensor()
    # total_labels = torch.Tensor()
    loss_fn = torch.nn.MSELoss(reduction='none')
    # loss_fn = masked_MSE_loss()

    eval_loss = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            data_pro_len = data[2].to(device)
            max_data_pro_len = data_pro_len.max().item()
            new_data_pro = data_pro[:, :max_data_pro_len, :]
            output, out_gnn, att = model(data_mol, new_data_pro, data_pro_len)
            output_pK = output[:, 0]
            output_pAC50 = output[:, 1]

            total_preds_pK = torch.cat((total_preds_pK, output_pK.cpu()), 0)
            total_labels_pK = torch.cat((total_labels_pK.view(-1,1), data_mol.y_pK.view(-1, 1).cpu()), 0)
            total_labels_pK_mask = torch.cat((total_labels_pK_mask.view(-1,1), data_mol.y_pK_mask.view(-1, 1).cpu()), 0)

            total_preds_pAC50 = torch.cat((total_preds_pAC50, output_pAC50.cpu()), 0)
            total_labels_pAC50 = torch.cat((total_labels_pAC50.view(-1,1), data_mol.y_pAC50.view(-1, 1).cpu()), 0)
            total_labels_pAC50_mask = torch.cat((total_labels_pAC50_mask.view(-1,1), data_mol.y_pAC50_mask.view(-1, 1).cpu()), 0)

            loss_pK = loss_fn(output_pK.view(-1,1), data_mol.y_pK.view(-1, 1).float().to(device))
            y_pK_mask = data_mol.y_pK_mask.view(-1, 1).float().to(device)
            loss_pK = torch.sum(loss_pK[y_pK_mask.bool()]) / (torch.sum(y_pK_mask) + 1e-9)
            loss_pAC50 = loss_fn(output_pAC50.view(-1,1), data_mol.y_pAC50.view(-1, 1).float().to(device))
            y_pAC50_mask = data_mol.y_pAC50_mask.view(-1, 1).float().to(device)
            loss_pAC50 = torch.sum(loss_pAC50[y_pAC50_mask.bool()]) / (torch.sum(y_pAC50_mask) + 1e-9)
            loss = (loss_pK + loss_pAC50)/2
            eval_loss.append(loss.item())
    eval_loss = np.average(eval_loss)
    total_labels_pK2 = total_labels_pK[torch.flatten(total_labels_pK_mask).bool()] #Take out the label with the corresponding value (the mask value is 1)
    total_preds_pK2 = total_preds_pK[torch.flatten(total_labels_pK_mask).bool()]
    total_labels_pAC502 = total_labels_pAC50[torch.flatten(total_labels_pAC50_mask).bool()]
    total_preds_pAC502 = total_preds_pAC50[torch.flatten(total_labels_pAC50_mask).bool()]
    

    total_labels = torch.cat((total_labels_pK2, total_labels_pAC502), 0)
    total_preds = torch.cat((total_preds_pK2, total_preds_pAC502), 0)
    
    
    # return eval_loss, total_labels.numpy().flatten(), total_preds.numpy().flatten()

    return eval_loss, total_labels_pK2.numpy().flatten(), total_preds_pK2.numpy().flatten(), total_labels_pAC502.numpy().flatten(), total_preds_pAC502.numpy().flatten()



# predict
def predicting(model, device, loader):
    model.eval()
    total_labels_pK = torch.Tensor()
    total_preds_pK = torch.Tensor()
    total_labels_pAC50 = torch.Tensor()
    total_preds_pAC50 = torch.Tensor()
    total_labels_pK_mask = torch.Tensor()
    total_labels_pAC50_mask = torch.Tensor()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            data_pro_len = data[2].to(device)
            output, out_gnn, att = model(data_mol, data_pro, data_pro_len)
            output_pK = output[:, 0]
            output_pAC50 = output[:, 1]

            total_preds_pK = torch.cat((total_preds_pK, output_pK.cpu()), 0)
            total_labels_pK = torch.cat((total_labels_pK.view(-1,1), data_mol.y_pK.view(-1, 1).cpu()), 0)
            total_labels_pK_mask = torch.cat((total_labels_pK_mask.view(-1,1), data_mol.y_pK_mask.view(-1, 1).cpu()), 0)

            total_preds_pAC50 = torch.cat((total_preds_pAC50, output_pAC50.cpu()), 0)
            total_labels_pAC50 = torch.cat((total_labels_pAC50.view(-1,1), data_mol.y_pAC50.view(-1, 1).cpu()), 0)
            total_labels_pAC50_mask = torch.cat((total_labels_pAC50_mask.view(-1,1), data_mol.y_pAC50_mask.view(-1, 1).cpu()), 0)

        total_labels_pK2 = total_labels_pK[torch.flatten(total_labels_pK_mask).bool()]
        total_preds_pK2 = total_preds_pK[torch.flatten(total_labels_pK_mask).bool()]
        total_labels_pAC502 = total_labels_pAC50[torch.flatten(total_labels_pAC50_mask).bool()]
        total_preds_pAC502 = total_preds_pAC50[torch.flatten(total_labels_pAC50_mask).bool()]
    return total_labels_pK2.numpy().flatten(), total_preds_pK2.numpy().flatten(), total_labels_pAC502.numpy().flatten(), total_preds_pAC502.numpy().flatten()




# nomarlize

# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    return c_size, features, edge_index


def target_matrics(key,embedding_path):
    eembedding_file = os.path.join(embedding_path, key)
    input_feature = torch.load(eembedding_file)
    return input_feature['feature'], input_feature['size']


def create_dataset_for_test(dataset_path,embedding_path):
    # load dataset
    ligands = json.load(open(dataset_path + 'compounds.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    affinity_df = pd.read_csv(dataset_path + 'affinity.csv')
    smile_file_name = dataset_path + '/smile_graph'
    
    # load compounds graph 
    compound_iso_smiles = {}
    compound_iso_smiles_list = []
    smile_graph = {}
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        compound_iso_smiles[d] = lg
        compound_iso_smiles_list.append(lg)
    #将train.csv中的compound_id转换为smiles，再转换为compound_iso_smiles
    affinity_df['compound_iso_smiles'] = affinity_df['compound_id'].apply(lambda x: compound_iso_smiles[x])

    if os.path.exists(smile_file_name):
        print('load smile graph ...')
        smile_graph = pickle.load(open(smile_file_name, 'rb'))
    else:
        # create smile graph
        print('create smile_graph ...')
        smile_graph = {}
        for smile in compound_iso_smiles_list:
            g = smile_to_graph(smile)
            smile_graph[smile] = g
        with open(smile_file_name,'wb+') as f:
            pickle.dump(smile_graph,f)
    # load seqs
    target_key = []
    target_graph = {}
    for key in proteins.keys():
        target_key.append(key)
        target_graph[key] = target_matrics(key,embedding_path)
    # load affinity matrix...
    print('load affinity dataframe...')
    stime=time.time()

    print('load train data...')
    # pick out test entries
    test_drugs, test_prot_keys, test_YpK, test_YAC50, test_YpK_mask, test_YAC50_mask = list(affinity_df['compound_iso_smiles']), list(
        affinity_df['uniprot_id']), list(affinity_df ['pK']), list(affinity_df['pAC50']), list(affinity_df['pK_mask']), list(affinity_df['pAC50_mask'])
    test_drugs, test_prot_keys, test_YpK, test_YAC50, test_YpK_mask,test_YAC50_mask = np.asarray(test_drugs), np.asarray(test_prot_keys), np.asarray(test_YpK), \
                                                                                            np.asarray(test_YAC50), np.asarray(test_YpK_mask), np.asarray(test_YAC50_mask)
    test_dataset = DTADataset(root=dataset_path, dataset='test', xd=test_drugs, target_key=test_prot_keys,
                               y_pK=test_YpK, y_pAC50=test_YAC50, y_pK_mask=test_YpK_mask, y_pAC50_mask=test_YAC50_mask,
                               smile_graph=smile_graph, target_graph=target_graph)
    return test_dataset


def create_dataset_for_train(dataset_path,embedding_path):
    # load dataset
    ligands = json.load(open(dataset_path + 'compounds.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    affinity_train_df = pd.read_csv(dataset_path + 'train.csv')
    affinity_valid_df = pd.read_csv(dataset_path + 'valid.csv')
    smile_file_name = dataset_path + '/smile_graph'
    # load compounds graph 
    compound_iso_smiles = {}
    compound_iso_smiles_list = []
    smile_graph = {}
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        compound_iso_smiles[d] = lg
        compound_iso_smiles_list.append(lg)
    
    affinity_train_df['compound_iso_smiles'] = affinity_train_df['compound_id'].apply(lambda x: compound_iso_smiles[x])
    affinity_valid_df['compound_iso_smiles'] = affinity_valid_df['compound_id'].apply(lambda x: compound_iso_smiles[x])

    if os.path.exists(smile_file_name):
        print('load smile graph ...')
        smile_graph = pickle.load(open(smile_file_name, 'rb'))
    else:
        # create smile graph
        print('create smile_graph ...')
        smile_graph = {}
        for smile in compound_iso_smiles_list:
            g = smile_to_graph(smile)
            smile_graph[smile] = g
        with open(smile_file_name,'wb+') as f:
            pickle.dump(smile_graph,f)
    # load seqs
    target_key = []
    target_graph = {}
    for key in proteins.keys():
        target_key.append(key)
        target_graph[key] = target_matrics(key,embedding_path)
    # load affinity matrix...
    print('load affinity dataframe...')
    
    stime=time.time()

    print('load train data...')
    
    #uniprot_id,compound_id,pK,pAC50,pK_mask,pAC50_mask
    train_drugs, train_prot_keys, train_YpK, train_YAC50, train_YpK_mask,train_YAC50_mask = list(affinity_train_df['compound_iso_smiles']), list(
        affinity_train_df['uniprot_id']), list(affinity_train_df ['pK']), list(affinity_train_df['pAC50']), list(affinity_train_df['pK_mask']), list(affinity_train_df['pAC50_mask'])
    train_drugs, train_prot_keys, train_YpK, train_YAC50, train_YpK_mask,train_YAC50_mask = np.asarray(train_drugs), np.asarray(train_prot_keys), np.asarray(train_YpK), \
                                                                                            np.asarray(train_YAC50), np.asarray(train_YpK_mask), np.asarray(train_YAC50_mask)
    train_dataset = DTADataset(root=dataset_path, dataset='train', xd=train_drugs, target_key=train_prot_keys,
                               y_pK=train_YpK, y_pAC50=train_YAC50, y_pK_mask=train_YpK_mask, y_pAC50_mask=train_YAC50_mask,
                               smile_graph=smile_graph, target_graph=target_graph)


    valid_drugs, valid_prot_keys, valid_YpK, valid_YAC50, valid_YpK_mask, valid_YAC50_mask = list(affinity_valid_df['compound_iso_smiles']), list(
        affinity_valid_df['uniprot_id']), list(affinity_valid_df['pK']), list(affinity_valid_df['pAC50']), list(affinity_valid_df['pK_mask']), list(affinity_valid_df['pAC50_mask'])
    valid_drugs, valid_prot_keys, valid_YpK, valid_YAC50, valid_YpK_mask, valid_YAC50_mask = np.asarray(valid_drugs), np.asarray(valid_prot_keys), np.asarray(valid_YpK), \
                                                                                            np.asarray(valid_YAC50), np.asarray(valid_YpK_mask), np.asarray(valid_YAC50_mask)
    valid_dataset = DTADataset(root=dataset_path, dataset='valid', xd=valid_drugs,target_key=valid_prot_keys, 
                               y_pK=valid_YpK, y_pAC50=valid_YAC50, y_pK_mask=valid_YpK_mask, y_pAC50_mask=valid_YAC50_mask,
                               smile_graph=smile_graph, target_graph=target_graph)
    return train_dataset, valid_dataset


def create_dataset_for_prediction(compound_smiles, protein_id_list, embedding_path):
    #处理化合物信息
    data_list_mol = []
    data_list_pro = []
    data_list_pro_len = []
    
    lg = Chem.MolToSmiles(Chem.MolFromSmiles(compound_smiles), isomericSmiles=True)
    c_size, features, edge_index = smile_to_graph(lg)
    GCNData_mol = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0))
    GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))

    for protein_id in protein_id_list:
        target_graph_feature, target_len_ = target_matrics(protein_id, embedding_path)
        data_list_mol.append(GCNData_mol)
        data_list_pro.append(target_graph_feature)
        data_list_pro_len.append(target_len_)
    
    GCNData_mol = Batch.from_data_list(data_list_mol)
    target_graph_feature = default_collate(data_list_pro)
    target_len = default_collate(data_list_pro_len)

    return GCNData_mol, target_graph_feature, target_len

def data_predict(model, device, compound_smiles, protein_id_list, embedding_path):
    model.eval()
    data_mol, data_pro, data_pro_len = create_dataset_for_prediction(compound_smiles, protein_id_list, embedding_path)
    data_mol = data_mol.to(device)
    data_pro = data_pro.to(device)
    data_pro_len = data_pro_len.to(device)

    with torch.no_grad():
        output = model(data_mol, data_pro, data_pro_len)
        output, out_gnn, att = model(data_mol, data_pro, data_pro_len)
        output = output.view(-1, 2)
    
    return output.cpu().numpy() #返回的是tensor,之前是output.cpu()

def create_dataset_for_prediction_smilesbatch(compound_smiles, protein_id, embedding_path):
    
    data_list_mol = []
    data_list_pro = []
    data_list_pro_len = []
    
    for compound_smile in compound_smiles:
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(compound_smile), isomericSmiles=True)
        c_size, features, edge_index = smile_to_graph(lg)
        GCNData_mol = DATA.Data(x=torch.Tensor(features),
                                        edge_index=torch.LongTensor(edge_index).transpose(1, 0))
        GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))
        data_list_mol.append(GCNData_mol)

        target_graph_feature, target_len_ = target_matrics(protein_id, embedding_path)
        data_list_pro.append(target_graph_feature)
        data_list_pro_len.append(target_len_)
    
    GCNData_mol = Batch.from_data_list(data_list_mol)
    target_graph_feature = default_collate(data_list_pro)
    target_len = default_collate(data_list_pro_len)
   

    return GCNData_mol, target_graph_feature, target_len

def data_predict_smilesbatch(model, device, compound_smiles, protein_id, embedding_path):
    model.eval()
    data_mol, data_pro, data_pro_len = create_dataset_for_prediction_smilesbatch(compound_smiles, protein_id, embedding_path)
    data_mol = data_mol.to(device)
    data_pro = data_pro.to(device)
    data_pro_len = data_pro_len.to(device)

    with torch.no_grad():
        output = model(data_mol, data_pro, data_pro_len)
        output, out_gnn, att = model(data_mol, data_pro, data_pro_len)
        output = output.view(-1, 2)
    
    return output.cpu().numpy()


    
#############  Evaluate matrix   #############
def get_cindex(Y, P):
    summ = 0
    pair = 0
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair != 0:
        return summ / pair
    else:
        return 0

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


from sklearn.metrics import r2_score
def get_r2(ys_orig, ys_line):
    return r2_score(ys_orig, ys_line)

def get_rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse

def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse

def get_pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def get_spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def get_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def performance_evaluation(labels, output):
    rmse = get_rmse(labels, output)
    mse = get_mse(labels, output)
    pearson = get_pearson(labels, output)
    spearman = get_spearman(labels, output)
    r2 = get_r2(labels, output)
    rm2 = get_rm2(labels, output)
    ci = get_ci(labels, output)
    cindex = get_cindex(labels, output)
    return rmse, mse, pearson, spearman, r2, rm2, ci, cindex