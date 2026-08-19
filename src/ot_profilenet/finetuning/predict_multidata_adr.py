from model import *
from utils import *
import numpy as np
import torch
import os,sys
import argparse
import json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def main(args, drug_smiles_list):

    drug_predict_result = defaultdict(dict)
    for dataset in ['GPCR','IonChannel','Enzyme','Kinase','NHR','Transporter','Other']: #'GPCR','IonChannel','Enzyme','Kinase','NHR','Transporter','Other'
        family_root = Path(args.data_root).expanduser().resolve() / dataset
        trained_cpi_model_path = Path(args.checkpoint_root).expanduser().resolve() / dataset / args.checkpoint_name

        USE_CUDA = torch.cuda.is_available()
        device = torch.device(args.device if USE_CUDA else 'cpu')
        print(device)

        embedding_path = str(family_root / args.protein_embedding) + os.sep
        protein_dict = eval((family_root / "proteins.txt").read_text())
        uniprot_id_list = list(protein_dict.keys())
        print("{} dataset has {} proteins".format(dataset,len(uniprot_id_list)))

        model = CPI_classification(device,
                                emb_size=args.emb_size,
                                max_length=2000,
                                dropout=args.dropout_global,
                                modulator_emb_dim=args.modulator_emb_dim,
                                ppi_emb_dim=args.ppi_emb_dim,
                                h_dim=args.h_dim,
                                n_heads=args.n_heads,
                                n_output=1)
        model.to(device)
        
        checkpoint = torch.load(trained_cpi_model_path, map_location=device)
        model.load_state_dict(checkpoint['net'])
        print(f'predicting on model:{trained_cpi_model_path};device:{device};')
        
        for uniprot_id in tqdm(uniprot_id_list):
            batches_list = [drug_smiles_list[i:i+516] for i in range(0, len(drug_smiles_list), 516)]
            for batch_smiles in batches_list:
                affinity_value = data_predict_smilesbatch(model, device, batch_smiles, uniprot_id, embedding_path)
                for drug_smiles,affinity_value_item in zip(batch_smiles,affinity_value):
                    affinity_value_item = affinity_value_item.tolist() # The former one is the pK value, and the latter one is the pAC50 value
                    drug_predict_result[drug_smiles][uniprot_id] = affinity_value_item # {"smiles1":{"uniprot_id1":affinity_value1, "uniprot_id2":affinity_value2}}    

    return drug_predict_result


            

#['GPCR','IonChannel','Enzyme','Kinase','NHR','Transporter','Other']
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:4',type=str,help='device id (0,1,2,3)')
    # parser.add_argument('--dataset', type=str, default='Kinase',choices=['GPCR','IonChannel','Enzyme','Kinase','NHR','Transporter','Other'])
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dropout_global', type=float, default=0)
    parser.add_argument('--train_val_ratio', type=float, default=0.9)
    parser.add_argument('--early_stop', type=int, default=25) #10
    parser.add_argument('--stop_epoch', type=int, default=0)
    parser.add_argument('--best_epoch', type=int, default=-1)
    parser.add_argument('--best_pearson', type=float, default=0)
    parser.add_argument('--last_epoch', type=int, default=1)
    parser.add_argument('--best_model', type=str, default=None)
    parser.add_argument('--modulator_emb_dim', type=int, default=256)
    parser.add_argument('--ppi_emb_dim', type=int, default=512)
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--protein_embedding', type=str, default='esm2',choices=['onehot','tape', 'esm1b', 'esm2','esm2_1024'])
    parser.add_argument('--emb_size', type=int, default=1280,choices=[20, 768, 1280, 1280])
    parser.add_argument('--data-root', required=True, help='Directory containing target-family protein files and embeddings')
    parser.add_argument('--checkpoint-root', required=True, help='Directory containing one checkpoint subdirectory per target family')
    parser.add_argument('--checkpoint-name', default='ratio_0.9batch128LR_1e-4random_0_esm2.pt')
    parser.add_argument('--adr-data-root', required=True, help='Directory containing the 18 SOC subdirectories')
    parser.add_argument('--output-json', required=True)
    args = parser.parse_args()
    

    data_path = str(Path(args.adr_data_root).expanduser().resolve())
    adr_name_list = ['Blood and lymphatic system disorders','Cardiac disorders','Ear and labyrinth disorders','Endocrine disorder',
                    'Eye disorders','Gastrointestinal disorders','Hepatobiliary disorders','Immune system disorders','Infections and infestations',
                    'Metabolism and nutrition disorders','Musculoskeletal and connective tissue disorders','Nervous system disorders',
                    'Psychiatric disorders','Renal and urinary disorders','Reproductive system and breast disorders',
                    'Respiratory, thoracic and mediastinal disorders','Skin and subcutaneous tissue disorders','Vascular disorders']
    all_smiles_list = []
    for adr_name in adr_name_list:
        data_df_path = os.path.join(data_path,adr_name,"adrdrug_dup_label.csv")
        data_df = pd.read_csv(data_df_path)
        drug_smiles_list = data_df['canonical_smiles'].to_list()
        all_smiles_list.extend(drug_smiles_list)
    

    all_smiles_list = list(set(all_smiles_list))
    print("all smiles length is:",len(all_smiles_list))

    drug_predict_result = main(args, all_smiles_list)


    # Save the results
    result_save_path = Path(args.output_json).expanduser().resolve()
    result_save_path.parent.mkdir(parents=True, exist_ok=True)
    with result_save_path.open('w', encoding='utf-8') as f:
        json.dump(drug_predict_result,f)


# nohup python -u predict_multidata_adr.py >adr_tree_predict.log 2>&1 &
