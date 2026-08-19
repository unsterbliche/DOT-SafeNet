from model import *
from utils import *
import numpy as np
from sklearn.model_selection import KFold
import torch
import os,sys
import warnings
warnings.filterwarnings("ignore")
import copy
from torch.utils.tensorboard import SummaryWriter
import random
import pathlib
import argparse

import warnings
warnings.filterwarnings("ignore")

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):

    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    
    torch.cuda.set_device(int(args.device.split(":")[-1]))
    device = torch.device(args.device) #args.gpu = int(os.environ['LOCAL_RANK'])
    same_seeds(args.seed)
    parmeter =  'bach'+str(args.train_batch_size) + 'LR'+ str(args.lr) + 'random' + str(args.seed) + args.protein_embedding
    dataset_root = pathlib.Path(args.data_root).expanduser().resolve() / args.dataset
    dataset_path = str(dataset_root / "train") + os.sep
    model_file_dir = str(pathlib.Path(args.output_root).expanduser().resolve() / args.dataset) + os.sep
    embedding_path = str(dataset_root / args.protein_embedding) + os.sep
    max_length = int((dataset_root / "max_length.txt").read_text().strip())
    best_model_name = model_file_dir + parmeter + '.pt'
    model_file_dir_epoch = model_file_dir + 'train_epoch/'
    log_dir = model_file_dir+ 'logs/' 
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=parmeter)
   
    if not os.path.exists(model_file_dir):
        pathlib.Path(model_file_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(log_dir):
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(model_file_dir_epoch):
        pathlib.Path(model_file_dir_epoch).mkdir(parents=True, exist_ok=True)
    

    model = CPI_classification(device,
                            emb_size=args.emb_size,
                            max_length=2000, #esm是2000
                            dropout=args.dropout_global,
                            modulator_emb_dim=args.modulator_emb_dim,
                            ppi_emb_dim=args.ppi_emb_dim,
                            h_dim=args.h_dim,
                            n_heads=args.n_heads)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=1)
    
    
    # load all valid entries to generate train_val set
    raw_fold = eval(open(dataset_path + 'valid_entries.txt','r').read())
    # np.random.seed(args.seed)
    random_entries = np.random.permutation(raw_fold)
    ptr = int(args.train_val_ratio*len(random_entries))
    train_val = [random_entries[:ptr],random_entries[ptr:]]
    train_data, valid_data = create_dataset_for_train(dataset_path, embedding_path, train_val)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True,
                                                collate_fn=collate,num_workers=16)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.test_batch_size, shuffle=False,
                                                collate_fn=collate,num_workers=16)

    
    train_losses = []
    val_losses = []
    epoch_num = []
    
    train_start_time = time.time()
    # print('epoch\ttime\ttrain_loss\tval_loss\tAUROC')
    train_epoch = 0
    best_auc = args.best_auc
    for epoch in range(args.num_epochs):
        train_epoch += 1
        start_time = time.time()
        print('Start training at epoch: {}'.format(epoch + 1))
        train_loss = train(model, device, train_loader, optimizer, epoch + 1)
        T, S, val_loss = evaluate(model, device, valid_loader)
        scheduler.step(val_loss)
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Valid/Loss', val_loss, epoch)
        epoch_num.append(epoch+1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)


        current_roc_auc, current_aupr, precision, accuracy, recall, f1, specificity, mcc, bacc, pred_labels, best_threshold = performance_evaluation(S, T)
        AUCS = [str(epoch+1),str(format(time.time()-start_time, '.1f')),str(format(train_loss, '.4f')),str(format(val_loss, '.4f')),str(format(current_roc_auc, '.4f'))]
        print('epoch\ttime\ttrain_loss\tval_loss\tAUROC')
        print('\t'.join(map(str, AUCS)))
        print('Val AUC:\t{}'.format(current_roc_auc))
        print('Val AUPR:\t{}'.format(current_aupr))
        print('Val BACC:\t{}'.format(bacc))
        
        # Save the model only in the rank 0 process
        
        if current_roc_auc >= best_auc:
            stop_epoch = 0
            best_auc = current_roc_auc
            best_epoch = epoch + 1
            best_model = copy.deepcopy(model)
            best_model_state = {'net': best_model.state_dict(),'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(best_model_state, best_model_name)
        else:
            stop_epoch += 1

      
        # Save the model every 20 epochs
        model_file_dir_epoch_nowname = model_file_dir_epoch + 'model_epoch_' + str(epoch+1) + '.pt'
        if (epoch + 1) % 2 == 0: #本来是10
            # Save the model only in the rank 0 process
            model_state = {'net': model.state_dict(),'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save( model_state, model_file_dir_epoch_nowname)
            
                
            # torch.save(model.state_dict(), model_file_dir_epoch_nowname)

        if stop_epoch == args.early_stop:
            print('(EARLY STOP) No improvement since epoch ', best_epoch, '; best_test_AUC', best_auc)
            break

        torch.cuda.empty_cache()

    print('Finish training!')
    print('Total training time: {:.5f} hours'.format((time.time()-train_start_time)/3600))
    loss_df = pd.DataFrame({'epoch':epoch_num,'train_loss':train_losses,'val_loss':val_losses})
    loss_df.to_csv(log_dir + 'loss.csv',index=False)

    start_time = time.time()
    print('Last epoch test results: {}'.format(train_epoch))
    T, S, val_loss = evaluate(model, device, valid_loader)
    roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, bacc, pred_labels, best_threshold = performance_evaluation(S, T)
    print('AUC:\t{}'.format(roc_auc))
    print('AUPR:\t{}'.format(aupr))
    print('precision:\t{}'.format(precision))
    print('accuracy:\t{}'.format(accuracy))
    print('recall:\t{}'.format(recall))
    print('f1:\t{}'.format(f1))
    print('specificity:\t{}'.format(specificity))
    print('mcc:\t{}'.format(mcc))
    print('bacc:\t{}'.format(bacc))
    print('')
    print('Took {:.5f}s.'.format(time.time() - start_time))
  
    start_time = time.time()
    print('Best epoch test results: {}'.format(best_epoch))
    T, S, val_loss = evaluate(best_model, device, valid_loader)
    roc_auc, aupr, precision, accuracy, recall, f1, specificity, mcc, bacc, pred_labels, best_threshold = performance_evaluation(S, T)
    print('AUC:\t{}'.format(roc_auc))
    print('AUPR:\t{}'.format(aupr))
    print('precision:\t{}'.format(precision))
    print('accuracy:\t{}'.format(accuracy))
    print('recall:\t{}'.format(recall))
    print('f1:\t{}'.format(f1))
    print('specificity:\t{}'.format(specificity))
    print('mcc:\t{}'.format(mcc))
    print('bacc:\t{}'.format(bacc))
    print('Took {:.5f}s.'.format(time.time() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:1',type=str,help='device id (0,1,2,3)')
    parser.add_argument('--dataset', type=str, default='CPI_data_cls',choices=['BindingDB_cls','CPI_data_cls','kinases'])
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=100) #100
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dropout_global', type=float, default=0.2)
    parser.add_argument('--train_val_ratio', type=float, default=0.9)
    parser.add_argument('--early_stop', type=int, default=10) #10
    parser.add_argument('--stop_epoch', type=int, default=0)
    parser.add_argument('--best_epoch', type=int, default=-1)
    parser.add_argument('--best_auc', type=float, default=0.5)
    parser.add_argument('--last_epoch', type=int, default=1)
    parser.add_argument('--best_model', type=str, default=None)
    parser.add_argument('--modulator_emb_dim', type=int, default=256)
    parser.add_argument('--ppi_emb_dim', type=int, default=512)
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--protein_embedding', type=str, default='esm2',choices=['onehot','tape', 'esm1b', 'esm2','esm2_1024'])
    parser.add_argument('--emb_size', type=int, default=1280,choices=[20, 768, 1280, 1280])
    parser.add_argument('--data-root', required=True, help='Directory containing one subdirectory per pretraining dataset')
    parser.add_argument('--output-root', required=True, help='Directory in which checkpoints and logs are written')
    args = parser.parse_args()
    
    main(args)

# nohup python -u train.py >train_pretrain_cpi.log 2>&1 &
