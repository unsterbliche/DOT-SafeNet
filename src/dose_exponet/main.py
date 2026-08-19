import json
import pandas as pd
import torch
import numpy as np
import os
import random
# Utils
from utils.utils import DataLoader, compute_pna_degrees, virtual_screening, CustomWeightedRandomSampler, eval_test, eval_test_dose
from utils.dataset import *  # data
from utils.dataset_dose import *
from utils.trainer import Trainer
from utils.metrics import *
# Preprocessing
from utils import ligand_init
# Model
from models.net import net
import argparse
import ast
import warnings
warnings.filterwarnings("ignore")

def tuple_type(s):
    try:
        # Safely evaluate the string as a tuple
        value = ast.literal_eval(s)
        if not isinstance(value, tuple):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {s}")
    return value

def list_type(s):
    try:
        # Safely evaluate the string as a tuple
        value = ast.literal_eval(s)
        if not isinstance(value, list):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid list value: {s}")
    return value

parser = argparse.ArgumentParser()

### Seed and device
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--device', type=str, default='cuda:4', help='')
parser.add_argument('--config_path',type=str,required=True,help='Model JSON configuration')
### Data and Pre-processing
parser.add_argument('--datafolder', type=str, required=True, help='Training dataset directory')
parser.add_argument('--result_path', type=str,required=True,help='Directory for metrics and checkpoints')
parser.add_argument('--save_interpret', type=bool,default=False,help='path to save results')

# For PDBBIND datasets - we train for 30K iteration 
parser.add_argument('--regression_task',type=bool, help='True if regression else False',default=True) 
# For any classification type - we train for 100 epochs (same as DrugBAN) [change --total_iters = None]
parser.add_argument('--classification_task',type=bool, help='True if classification else False') 
parser.add_argument('--mclassification_task',type=int, help='number of multiclassification, 0 if no multiclass task')
parser.add_argument('--epochs', type=int, default=1, help='')
parser.add_argument('--evaluate_epoch',type=int,default=1)

parser.add_argument('--total_iters',type=int,default=None) 
parser.add_argument('--evaluate_step',type=int,default=500)


# optimizer params - only change this for PDBBind v2016
parser.add_argument('--lrate', type=float, default=1e-4, help='learning rate for PSICHIC') # change to 1e-5 for LargeScaleInteractionDataset
parser.add_argument('--eps', type=float, default=1e-8, help='higher = closer to SGD') # change to 1e-5 for PDBv2016
parser.add_argument('--betas',type=tuple_type, default="(0.9,0.999)")  # change to (0.9,0.99) for PDBv2016
# batch size
parser.add_argument('--batch_size',type=int,default=64)
# sampling method - only used for pretraining large-scale interaction dataset ; allow self specified weights to the samples
parser.add_argument('--sampling_col',type=str,default='') 
parser.add_argument('--trained_model_path',type=str,default='',help='This does not need to be perfectly aligned, as you can add prediction head for some other tasks as well!')
# parser.add_argument('--finetune_modules',type=list_type,default=None)
parser.add_argument('--finetune_path',type=str,default=None)
# notebook mode?
parser.add_argument('--nb_mode',type=bool,default=False)
# dose need?
parser.add_argument('--dose_mode',type=bool,default=False)
parser.add_argument('--set_layer', type=str, default="SetRep")
parser.add_argument('--early_stopping_epochs', type=int, default=500)


args = parser.parse_args()

with open(args.config_path,'r') as f:
        config = json.load(f)

# overwrite 
config['optimizer']['lrate'] = args.lrate
config['optimizer']['eps'] = args.eps
config['optimizer']['betas'] = args.betas
config['tasks']['regression_task'] = args.regression_task
config['tasks']['classification_task'] = args.classification_task
config['tasks']['mclassification_task'] = args.mclassification_task
config['params']['set_layer'] = args.set_layer


# device
device = torch.device(args.device)
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

model_path = os.path.join(args.result_path,'save_model_seed{}'.format(args.seed))
if not os.path.exists(model_path):
    os.makedirs(model_path)

interpret_path = os.path.join(args.result_path,'interpretation_result_seed{}'.format(args.seed))
if not os.path.exists(interpret_path):
    os.makedirs(interpret_path)

if args.epochs is not None and args.total_iters is not None:
    print('If epochs and total iters are both not None, then we only use iters.')
    args.epochs = None


print(args)
with open(os.path.join(args.result_path, 'model_params.txt'), 'w') as f:
    f.write(str(args))


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(args.seed)


## import files
train_df = pd.read_csv(os.path.join(args.datafolder,'train.csv'))
test_df = pd.read_csv(os.path.join(args.datafolder,'test.csv'))

valid_path = os.path.join(args.datafolder,'valid.csv')
valid_df = 1 #None
if os.path.exists(valid_path):
    valid_df = pd.read_csv(valid_path)
    ligand_smiles = list(set(train_df['smiles'].tolist()+test_df['smiles'].tolist()+valid_df['smiles'].tolist())) 
else:
    ligand_smiles = list(set(train_df['smiles'].tolist()+test_df['smiles'].tolist())) 


ligand_path = os.path.join(args.datafolder,'ligand.pt')
if os.path.exists(ligand_path):
    print('Loading Ligand Graph data...')
    ligand_dict = torch.load(ligand_path)
else:
    print('Initialising Ligand SMILES to Ligand Graph...')
    ligand_dict = ligand_init(ligand_smiles)
    torch.save(ligand_dict,ligand_path)

torch.cuda.empty_cache()
##TODO: drop any invalid smiles

##TODO: drop any invalid smiles

## 
## training loader
train_shuffle = True
train_sampler = None

if args.sampling_col:
    train_weights = torch.from_numpy(train_df[args.sampling_col].values)

    def sampler_from_weights(weights):
        sampler = CustomWeightedRandomSampler(weights, len(weights), replacement=True)
        
        return sampler 

    train_shuffle = False
    train_sampler = sampler_from_weights(train_weights)
if train_sampler is not None:
    print('shuffle should be False: ',train_shuffle)


if args.dose_mode == False:
    train_dataset = MotifMoleculeDataset(train_df, ligand_dict, device=args.device)
    test_dataset = MotifMoleculeDataset(test_df, ligand_dict, device=args.device)
else:
    train_dataset = MotifMoleculeDataset_dose(train_df, ligand_dict, device=args.device, cache_transform=True)
    test_dataset = MotifMoleculeDataset_dose(test_df, ligand_dict, device=args.device, cache_transform=True)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_shuffle,
                        sampler=train_sampler, follow_batch=['mol_x', 'clique_x']) 

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            follow_batch=['mol_x', 'clique_x']) 


valid_dataset, valid_loader = None, None
if valid_df is not None:
    if args.dose_mode == False:
        valid_dataset = MotifMoleculeDataset(valid_df, ligand_dict, device=args.device)
    else:
        valid_dataset = MotifMoleculeDataset_dose(valid_df, ligand_dict, device=args.device, cache_transform=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                follow_batch=['mol_x', 'clique_x']
                                )

    
degree_path = os.path.join(args.result_path,'save_model_seed{}'.format(args.seed),'degree.pt')
if not os.path.exists(degree_path):
    print('Computing training data degrees for PNA')
    mol_deg, clique_deg = compute_pna_degrees(train_loader)
    degree_dict = {'ligand_deg':mol_deg, 'clique_deg':clique_deg}
    torch.save(degree_dict, degree_path)
else:
    degree_dict = torch.load(degree_path)
    mol_deg, clique_deg = degree_dict['ligand_deg'], degree_dict['clique_deg']


model = net(mol_deg,
            # MOLECULE
            mol_in_channels=config['params']['mol_in_channels'], 
            hidden_channels=config['params']['hidden_channels'],                 
            num_layers=config['params']['num_layers'],
            # heads=config['params']['heads'],
            clique_num_timesteps=config['params']['clique_num_timesteps'],
            num_timesteps=config['params']['num_timesteps'],
            n_hidden_sets=config['params']['n_hidden_sets'],
            n_elements=config['params']['n_elements'],
            dropout=config['params']['dropout'],
            # dropout_attn_score=config['params']['dropout_attn_score'],
            # output
            regression_head=config['tasks']['regression_task'],
            classification_head=config['tasks']['classification_task'] ,
            multiclassification_head=config['tasks']['mclassification_task'],
            set_layer=config['params']['set_layer'],
            dose_mode=args.dose_mode,
            device=device).to(device)


model.reset_parameters()
if args.trained_model_path:
    param_dict = os.path.join(args.trained_model_path,'model.pt')
    model.load_state_dict(torch.load(param_dict, map_location=args.device),strict=False)
    print('Pretrained model loaded!')

nParams = sum([p.nelement() for p in model.parameters()])
print('Model loaded with number of parameters being:', str(nParams))

with open(os.path.join(args.result_path,'save_model_seed{}'.format(args.seed),'config.json'),'w') as f:
    json.dump(config,f, indent=4)

## Evaluation metric type depends on task
if config['tasks']['regression_task']:
    evaluation_metric = 'rmse' #pearson
elif config['tasks']['classification_task']:
    evaluation_metric = 'roc'
elif config['tasks']['mclassification_task']:
    evaluation_metric = 'macro_f1'
else:
    raise Exception("no valid interaction property prediction task...")


engine = Trainer(model=model, lrate=config['optimizer']['lrate'], min_lrate=config['optimizer']['min_lrate'],
                    wdecay=config['optimizer']['weight_decay'], betas=config['optimizer']['betas'], 
                    eps=config['optimizer']['eps'], amsgrad=config['optimizer']['amsgrad'],
                    clip=config['optimizer']['clip'], steps_per_epoch=len(train_loader), 
                    num_epochs=args.epochs,total_iters = args.total_iters, 
                    warmup_iters=config['optimizer']['warmup_iters'], 
                    lr_decay_iters=config['optimizer']['lr_decay_iters'],
                    schedule_lr=config['optimizer']['schedule_lr'], regression_weight=1, classification_weight=1, 
                    evaluate_metric=evaluation_metric, result_path=args.result_path, runid=args.seed, 
                    finetune_path=args.finetune_path, dose_mode=args.dose_mode, early_stopping_epochs=args.early_stopping_epochs,
                    device=device)

print('-'*50)
print('start training model')
if args.epochs:
    engine.train_epoch(train_loader, val_loader = valid_loader, test_loader = test_loader, evaluate_epoch = args.evaluate_epoch)
else:
    engine.train_step(train_loader, val_loader = valid_loader, test_loader = test_loader, evaluate_step = args.evaluate_step)

print('finished training model')
print('-'*50)

print('loading best checkpoint and predicting test data')
print('-'*50)
model.load_state_dict(torch.load(os.path.join(args.result_path,'save_model_seed{}'.format(args.seed),'model.pt'))) #save the best model

if args.dose_mode == False:
    screen_df, eval_dict, attention_dict = eval_test(test_df, model, test_loader,
                    result_path=os.path.join(args.result_path, "interpretation_result_seed{}".format(args.seed)), 
                    save_interpret=args.save_interpret, 
                    ligand_dict=ligand_dict, device=args.device) #, dose_mode=args.dose_mode
else:
    screen_df, eval_dict, attention_dict = eval_test_dose(test_df, model, test_loader,
                    result_path=os.path.join(args.result_path, "interpretation_result_seed{}".format(args.seed)), 
                    save_interpret=args.save_interpret, 
                    ligand_dict=ligand_dict, device=args.device)



# mol_embedding = attention_dict["mol_feature"].detach().cpu().numpy()
# # save mol_embedding
# np.save(os.path.join(args.result_path,'mol_embedding_seed{}.npy'.format(args.seed)), mol_embedding)

screen_df.to_csv(os.path.join(args.result_path,'test_prediction_seed{}.csv'.format(args.seed)),index=False)

with open(os.path.join(args.result_path,'eval_dict_seed{}.json'.format(args.seed)), 'w') as f:
    json.dump(eval_dict, f)
