#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import pickle
import json
import numpy as np
import os,sys
from collections import OrderedDict
from collections import defaultdict
import torch
from torch.utils.data import Dataset
# import biovec
from Bio.Seq import Seq
# from biovec import models
from tape import ProteinBertModel, TAPETokenizer
import esm

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import pandas as pd


def generate_fold(dataset_path):
    affinity_df = pd.read_csv(dataset_path + '/affinity.csv')
    affinity_df['combined_mask'] = affinity_df['pK_mask'].astype(str) + affinity_df['pAC50_mask'].astype(str) #11,01,10
    from sklearn.model_selection import train_test_split
    train_df, valid_df = train_test_split(affinity_df, test_size=0.1, stratify=affinity_df['combined_mask'], random_state=2024)
    
    train_df = train_df.drop(columns=['combined_mask'])
    valid_df = valid_df.drop(columns=['combined_mask'])

    train_df.to_csv(dataset_path + '/train.csv', index=False)
    valid_df.to_csv(dataset_path + '/valid.csv', index=False)



def seq_to_kmers(seq, k=3):
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]



def onehot(sequence_dict,out_file_path,max_length):
    Alfabeto = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    count = 0
    for key in sequence_dict.keys():
        sequence = sequence_dict[key]
        count += 1
        feature = np.zeros(shape=[max_length, len(Alfabeto)],dtype='float32')
        sequence = sequence.upper()
        size = len(sequence)
        indices = [Alfabeto.index(c) for c in sequence if c in Alfabeto]
        for j, index in enumerate(indices):
            feature[j, index] = float(1.0)
        percent = int((count / len(sequence_dict)) * 100)
        bar = '#' * int(count / len(sequence_dict) * 20)
        print(f'\r[{bar:<20}] {percent:>3}% ({count}/{len(sequence_dict)})', end='')
        feature = torch.from_numpy(feature)
        embeddings = {"feature":feature, "size":size}
        torch.save(embeddings, out_file_path + '/'+key)


def Bio2vec(sequence_dict,out_file_path,max_length):
    fasta_txt = ''
    for key in sequence_dict.keys():
        fasta_txt = fasta_txt + '>' + key + '\n'+ sequence_dict[key] + '\n'
    with open(out_file_path+"/bio2vec.fasta",'w') as f:
        print(fasta_txt.strip(),file=f)
    pv = biovec.models.ProtVec(out_file_path+ "/bio2vec.fasta", corpus_fname=out_file_path+ "/bio2vec_corpus.txt", n=3)
    count = 0
    for seq in sequence_dict.keys():
        sequence = seq_to_kmers(sequence_dict[seq], k=3)
        size = len(sequence)
        vec = np.zeros((max_length, 100),dtype='float32')
        i = 0
        for word in sequence: 
            vec[i] = pv.to_vecs(word)[0]
            i += 1
        feature = torch.from_numpy(vec)
        embeddings = {"feature":feature, "size":size}
        torch.save(embeddings, out_file_path + '/'+ seq)
        count+=1
        percent = int((count / len(sequence_dict)) * 100) 
        bar = '#' * int(count / len(sequence_dict) * 20)
        print(f'\r[{bar:<20}] {percent:>3}% ({count+1}/{len(sequence_dict)})', end='')

def tape_embedding(sequence_dict,out_file_path,max_length):
    model = ProteinBertModel.from_pretrained('bert-base')
    tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model
    model.eval()
    count = 0
    for key in sequence_dict.keys():
        tmp_list = [tokenizer.encode(sequence_dict[key].upper())]
        tmp_array = np.array(tmp_list)
        token_ids = torch.from_numpy(tmp_array)  # encoder
        sequence_output, _ = model(token_ids) # sequence_output is a tensor of shape (batch_size, sequence_length, hidden_size)
        sequence_output = sequence_output.detach().numpy()
        # padding to same size [???,768]
        feature = sequence_output.squeeze()
        feature = np.delete(feature,-1,axis=0) # -1 represents the last element
        feature = np.delete(feature,0,axis=0) # 0 represents the first element, and axis=0 indicates deletion by row
        size = feature.shape[0]
        pad_length = max_length - size
        if pad_length:
            padding = np.zeros((pad_length,768),dtype='float32')
            feature = np.r_[feature,padding]
        feature = torch.from_numpy(feature)
        embeddings = {"feature":feature, "size":size}
        torch.save(embeddings, out_file_path + '/'+key)
        count+=1
        percent = int((count / len(sequence_dict)) * 100)
        bar = '#' * int(count / len(sequence_dict) * 20)
        print(f'\r[{bar:<20}] {percent:>3}% ({count+1}/{len(sequence_dict)})', end='')


def esm1b_embedding(sequence_dict,out_file_path,max_length):
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    count = 0
    for key in sequence_dict.keys():
        target_seq = [(key,sequence_dict[key][:1022])] 
        # The maximum length of the input sequence for ESM1b is 1024. Considering that the characters <cls> and <eos> need to be added during the subsequent sequence processing, the first 1022 amino acids are taken here
        # target_seq = [(key,sequence_dict[key])]
        batch_labels, batch_strs, batch_tokens = batch_converter(target_seq)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        token_representations = token_representations.detach().numpy()
        # sequence_representations = token_representations[0,1:len(sequence_dict[key])+1].mean(axis=0) 
       
        feature = token_representations.squeeze() # token_representations (1,pro_len+2,1280), feature (pro_len+2,1280)
        feature = np.delete(feature,-1,axis=0) # Delete the representation of <eos>
        feature = np.delete(feature,0,axis=0) # Delete the representation of <cls>
        size = feature.shape[0]
        pad_length = 1024 - size
        if pad_length:
            padding = np.zeros((pad_length,1280),dtype='float32')
            feature = np.r_[feature,padding]
        # else:
        #     feature = feature[:max_length,:] # Extract the first max_length amino acids
        feature = torch.from_numpy(feature)
        embeddings = {"feature":feature, "size":size}
        torch.save(embeddings, out_file_path + '/'+key)
        count+=1
        percent = int((count / len(sequence_dict)) * 100)
        bar = '#' * int(count / len(sequence_dict) * 20)
        print(f'\r[{bar:<20}] {percent:>3}% ({count+1}/{len(sequence_dict)})', end='')


import ssl
import urllib.request

context = ssl._create_unverified_context()
urllib.request.urlopen("https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt", context=context)


def esm2_embedding(sequence_dict,out_file_path,max_length):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    count = 0
    for key in sequence_dict.keys():
        target_seq = [(key,sequence_dict[key][:2000])]
        batch_labels, batch_strs, batch_tokens = batch_converter(target_seq)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        token_representations = token_representations.detach().numpy()
        feature = token_representations.squeeze()
        feature = feature[1:-1] # Use the slicing operation to delete the representations of <cls> and <eos>
        size = feature.shape[0]
        pad_length = max_length - size
        if pad_length > 0:
            padding = np.zeros((pad_length,1280),dtype='float32')
            feature = np.concatenate((feature, padding), axis=0)
        else:
            feature = feature[:max_length,:]
        feature = torch.from_numpy(feature)
        embeddings = {"feature":feature, "size":size}
        torch.save(embeddings, out_file_path + '/'+key)
        count+=1
        percent = int((count / len(sequence_dict)) * 100)
        bar = '#' * int(count / len(sequence_dict) * 20)
        print(f'\r[{bar:<20}] {percent:>3}% ({count+1}/{len(sequence_dict)})', end='')
        del token_representations, feature
    
    


def generate_embeddings(dataset,embedding_type):
    sequence_dict = eval(open(dataset+'/proteins.txt','r').read())
    out_file_path = dataset + '/' + embedding_type
    # max_length = 0
    # for key in sequence_dict.keys():
    #     max_length = max(max_length,len(sequence_dict[key])+1)
    # #max_length = min(max_length,1024)
    # max_length = min(max_length,2000)
    max_length = 2000
    if not os.path.exists(out_file_path):
        os.mkdir(out_file_path)
    if embedding_type=='tape':
        tape_embedding(sequence_dict,out_file_path,max_length)
    elif embedding_type=='onehot':
        onehot(sequence_dict,out_file_path,max_length)
    elif embedding_type=='bio2vec':
        Bio2vec(sequence_dict,out_file_path,max_length)
    elif embedding_type=='esm1b':
        # esm1b_embedding(sequence_dict,out_file_path)
        esm1b_embedding(sequence_dict,out_file_path,max_length)
    elif embedding_type=='esm2':
        esm2_embedding(sequence_dict,out_file_path,max_length)
    with open(dataset + '/max_length.txt','w') as f:
        print(max_length,file=f)
    print('embedding files at:%s; max_length=%s'%(out_file_path,max_length))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--embedding-type", default="esm2", choices=["onehot", "bio2vec", "tape", "esm1b", "esm2"])
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    generate_embeddings(str(dataset_dir), args.embedding_type)
    generate_fold(str(dataset_dir / "train"))
