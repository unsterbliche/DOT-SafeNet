
import os
import os.path as osp
import time
import torch
import json, pickle
import numpy as np
import pandas as pd
import networkx as nx
import math
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from torch_geometric import data as DATA
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Dataset,InMemoryDataset, DataLoader, Batch
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc,accuracy_score,f1_score,balanced_accuracy_score
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
from torch.cuda.amp import autocast, GradScaler


# initialize the dataset
class DTADataset(Dataset):
    def __init__(self, root='/tmp', dataset=None,
                 xd=None, y=None, transform=None,
                 pre_transform=None, target_key=None, target_graph=None):
        
        self.root = root
        self.dataset = dataset
        self.xd = xd
        self.y = y
        self.target_key = target_key
        self.target_graph = target_graph
        self._indices = None
        self.transform = transform
        self.pre_transform = pre_transform
        self._process()
        super(DTADataset, self).__init__(root, transform, pre_transform)
        # self.process(xd, target_key, y)
    
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        num_npy = math.ceil(len(self.xd)/512)
        return [f'data_mol_{i}.pickle' for i in range(num_npy)]

    @property
    def processed_dir(self):
        return osp.join(self.root, self.dataset + 'processed_npy_batch256')
    
    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir) 
        # Check if all processed files exist
        all_files_exist = all(os.path.exists(osp.join(self.processed_dir, filename)) for filename in self.processed_file_names)
        
        # Only process the data if not all processed files exist
        if not all_files_exist:
            self.process(self.xd, self.target_key, self.y)

    # mile_graph/target_graph: dicts mapping compound SMILES/protein IDs to their graph representations
    def process(self, xd, target_key, y):
        assert (len(xd) == len(target_key) and len(xd) == len(y)), 'These lists must be the same length!'
        #Split xd, target, and y into batches of 256 for downstream processing
        xd_list = []
        target_key_list = []
        y_list = []
        for i in range(math.ceil(len(xd)/256)):
            xd_list.append(xd[i*256:(i+1)*256])
            target_key_list.append(target_key[i*256:(i+1)*256])
            y_list.append(y[i*256:(i+1)*256])
        for batch_idx in range(math.ceil(len(xd)/256)):
            data_list = []
            xd_ch = xd_list[batch_idx]
            target_key_ch = target_key_list[batch_idx]
            y_ch = y_list[batch_idx]
            for i in range(len(xd_ch)):
                smiles = xd_ch[i]
                tar_key = target_key_ch[i]
                labels = y_ch[i]

                #Get molecular graph data
                #smiles_graph: dict where keys are compound SMILES and values are molecular graph data
                c_size, features, edge_index = smile_to_graph(smiles)
                GCNData_mol = DATA.Data(x=torch.Tensor(features),
                                        edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                        y=torch.FloatTensor([labels])
                                        )
                GCNData_mol.target_key = tar_key
                GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))

                # data = {'GCNData_mol': GCNData_mol, 'target_key': tar_key}
                data_list.append(GCNData_mol)
           
            with open(osp.join(self.processed_dir,f'data_mol_{batch_idx}.pickle'), 'wb+') as f:
                pickle.dump(data_list, f)
        
            
    def len(self):
        return len(self.processed_file_names)
    
 
    def get(self, idx):
        # data_dict_list = np.load(osp.join(self.processed_dir,f'data_mol_{idx}.npy'),allow_pickle=True).item()
        data_dict_list = pickle.load(open(osp.join(self.processed_dir,f'data_mol_{idx}.pickle'), 'rb'))
        #data_dict_list: a list of dicts, each containing one batch's data
        batch_mol, batch_pro, batch_pro_len = [], [], []
        for i in range(len(data_dict_list)):
            data_dict = data_dict_list[i]
            data_mol = data_dict
            target_key = data_mol.target_key
            #Get protein data
            data_pro, data_pro_len = self.target_graph[target_key]
            batch_mol.append(data_mol)
            batch_pro.append(data_pro)
            batch_pro_len.append(data_pro_len)
        return batch_mol, batch_pro, batch_pro_len
 

#prepare the protein and drug pairs
def collate(data_list):
    batch_mol_list = []
    batch_pro_list = []
    batch_pro_len_list = []
    for i in range(len(data_list)):
        batch_mol_list.extend(data_list[i][0]) # Get molecular graph features of the i-th batch
        batch_pro_list.extend(data_list[i][1]) # Get protein data from the i-th batch
        batch_pro_len_list.extend(data_list[i][2]) # Get protein length information in the i-th batch
    batchA = Batch.from_data_list(batch_mol_list)
    batchB = default_collate(batch_pro_list) #The batch_pro_list tensors have consistent dimensions and can be directly concatenated
    batchC = default_collate(batch_pro_len_list)
    return batchA, batchB, batchC 

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = []
    scaler = GradScaler() #Enable mixed-precision training
    loss_fn = torch.nn.CrossEntropyLoss()
  
    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        data_pro_len = data[2].to(device)
        optimizer.zero_grad()
        with autocast():
            output = model(data_mol, data_pro, data_pro_len)
            loss = loss_fn(output, data_mol.y.view(-1, 1).long().squeeze().to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss.append(loss.item())
    train_loss = np.average(train_loss)
    return train_loss

def evaluate(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    loss_fn = torch.nn.CrossEntropyLoss()
    eval_loss = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            data_pro_len = data[2].to(device)
            output = model(data_mol, data_pro, data_pro_len)
            loss = loss_fn(output, data_mol.y.view(-1, 1).long().squeeze().to(device))
            total_preds = torch.cat((total_preds, output.cpu()), 0) #Concatenate the predictions from each batch
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
            eval_loss.append(loss.item())
    eval_loss = np.average(eval_loss)
    
    return total_labels.numpy().flatten(), total_preds.numpy(),eval_loss



# predict
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            data_pro_len = data[2].to(device)
            output = model(data_mol, data_pro, data_pro_len)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy()



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


def create_dataset_for_test(dataset_path,embedding_path,test_fold):
    # load dataset

    ligands = json.load(open(dataset_path + 'compounds.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(dataset_path + 'Y_dict', 'rb'), encoding='latin1')
    compound_iso_smiles_dict_file = dataset_path + '/compound_iso_smiles_dict.json'
    # load compounds graph 
    compound_iso_smiles = []
   
    if os.path.exists(compound_iso_smiles_dict_file):
        print('load compound_iso_smiles_dict ...')
        compound_iso_smiles_dict = json.load(open(compound_iso_smiles_dict_file, 'r'))
        # for d in ligands.keys():
        #     compound_iso_smiles.append(compound_iso_smiles_dict[d])
    else:
        compound_iso_smiles_dict = {}
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
            compound_iso_smiles.append(lg)
            compound_iso_smiles_dict[d] = lg #compound_iso_smiles_dict是一个字典，键是药物的id，值是药物的smiles
        with open(compound_iso_smiles_dict_file,'w') as f:
            json.dump(compound_iso_smiles_dict,f)
    
    # load seqs
    print('load target seqs ...')
    target_graph = {}
    for key in proteins.keys():
        target_graph[key] = target_matrics(key,embedding_path)
    # load affinity matrix...
    print('load affinity matrix...')
    print('test entries:', len(test_fold))
    stime=time.time()

    #Get data using train_fold indices keyed by the affinity dictionary
    key_list = list(affinity.keys())
    print('load train data...')
    key_test_list = test_fold
    test_fold_entries = {'compound_iso_smiles':[],'target_key':[],'affinity':[]}
    for test_ind in key_test_list:
        test_fold_entries['compound_iso_smiles'].append(compound_iso_smiles_dict[list(affinity[test_ind].keys())[0][1]]) #这里是根据化合物的id在compound_iso_smiles_dict中找到对应的smiles
        test_fold_entries['target_key'].append(list(affinity[test_ind].keys())[0][0])
        test_fold_entries['affinity'].append(list(affinity[test_ind].values())[0])

    df_test_fold = pd.DataFrame(test_fold_entries)
    test_drugs, test_prots_keys, test_Y = list(df_test_fold['compound_iso_smiles']), list(
        df_test_fold['target_key']), list(df_test_fold['affinity'])
    test_dataset = DTADataset(root=dataset_path, dataset=dataset_path + 'test_', xd=test_drugs,
                               target_key=test_prots_keys, y=test_Y, target_graph=target_graph)
    
    return test_dataset


def create_dataset_for_train(dataset_path,embedding_path,train_valid):
    # load dataset
    ligands = json.load(open(dataset_path + 'compounds.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(dataset_path + 'Y_dict', 'rb'), encoding='latin1')
    compound_iso_smiles_dict_file = dataset_path + '/compound_iso_smiles_dict.json'
    # load compounds graph 
    compound_iso_smiles = []
   
    if os.path.exists(compound_iso_smiles_dict_file):
        print('load compound_iso_smiles_dict ...')
        compound_iso_smiles_dict = json.load(open(compound_iso_smiles_dict_file, 'r'))
    else:
        compound_iso_smiles_dict = {}
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
            compound_iso_smiles.append(lg)
            compound_iso_smiles_dict[d] = lg
        with open(compound_iso_smiles_dict_file,'w') as f:
            json.dump(compound_iso_smiles_dict,f)
    
    
    # load seqs
    print('load target seqs ...')
    target_graph = {}
    for key in proteins.keys():
        target_graph[key] = target_matrics(key,embedding_path)
    # load affinity matrix...
    print('load affinity matrix...')
    train_fold = train_valid[0]
    valid_fold = train_valid[1]
    print('train entries:', len(train_fold))
    print('valid entries:', len(valid_fold))
    stime=time.time()

    key_list = list(affinity.keys())
    print('load train data...')
    key_train_list = train_fold
    train_fold_entries = {'compound_iso_smiles':[],'target_key':[],'affinity':[]}
    for train_ind in key_train_list:
        train_fold_entries['compound_iso_smiles'].append(compound_iso_smiles_dict[list(affinity[train_ind].keys())[0][1]]) #这里是根据化合物的id在compound_iso_smiles_dict中找到对应的smiles
        train_fold_entries['target_key'].append(list(affinity[train_ind].keys())[0][0])
        train_fold_entries['affinity'].append(list(affinity[train_ind].values())[0])
    
    print('load valid data...')
    key_valid_list = valid_fold
    valid_fold_entries = {'compound_iso_smiles':[],'target_key':[],'affinity':[]}
    for valid_ind in key_valid_list:
        valid_fold_entries['compound_iso_smiles'].append(compound_iso_smiles_dict[list(affinity[valid_ind].keys())[0][1]])
        valid_fold_entries['target_key'].append(list(affinity[valid_ind].keys())[0][0])
        valid_fold_entries['affinity'].append(list(affinity[valid_ind].values())[0])
    print('done time consuming:',time.time()-stime)
    
    df_train_fold = pd.DataFrame(train_fold_entries)
    train_drugs, train_prot_keys, train_Y = list(df_train_fold['compound_iso_smiles']), list(
        df_train_fold['target_key']), list(df_train_fold['affinity'])
    train_dataset = DTADataset(root=dataset_path, dataset=dataset_path + 'train_', xd=train_drugs, target_key=train_prot_keys,
                               y=train_Y, target_graph=target_graph) #embedding_path = embedding_path


    df_valid_fold = pd.DataFrame(valid_fold_entries)
    valid_drugs, valid_prots_keys, valid_Y = list(df_valid_fold['compound_iso_smiles']), list(
        df_valid_fold['target_key']), list(df_valid_fold['affinity'])
    valid_dataset = DTADataset(root=dataset_path, dataset=dataset_path + 'valid_', xd=valid_drugs,
                               target_key=valid_prots_keys, y=valid_Y, target_graph=target_graph)
    return train_dataset, valid_dataset


def remove_module_from_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, \
        precision_score, recall_score, \
        f1_score, confusion_matrix, accuracy_score, matthews_corrcoef

 


def performance_evaluation(output, labels):
    output = torch.softmax(torch.from_numpy(output), dim=1)
    pred_scores = output[:, 1]
    roc_auc = roc_auc_score(labels, pred_scores)
    prec, reca, _ = precision_recall_curve(labels, pred_scores)
    aupr = auc(reca, prec)

    best_threshold = 0.5
    pred_labels = output[:, 1] >= best_threshold
    precision = precision_score(labels, pred_labels)
    accuracy = accuracy_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    (tn, fp, fn, tp) = confusion_matrix(labels, pred_labels).ravel()
    specificity = tn / (tn + fp) 
    mcc = matthews_corrcoef(labels, pred_labels)
    bacc = balanced_accuracy_score(labels, pred_labels) 

    return roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, bacc, pred_labels,best_threshold


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
        output = output.cpu().detach().numpy()
        output = torch.softmax(torch.from_numpy(output), dim=1)
        pred_scores = output[:, 1]
        pred_scores = pred_scores.view(-1, 1)
    
    return pred_scores.cpu().numpy()


