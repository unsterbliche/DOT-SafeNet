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
from tqdm import tqdm

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
    parmeter =  'ratio_' +str(args.train_val_ratio)+'batch'+str(args.train_batch_size) + 'LR_1e-4'+'random_0_' + args.protein_embedding
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
                            max_length=2000,
                            dropout=args.dropout_global,
                            modulator_emb_dim=args.modulator_emb_dim,
                            ppi_emb_dim=args.ppi_emb_dim,
                            h_dim=args.h_dim,
                            n_heads=args.n_heads,
                            n_output=1)
    
    model.to(device)
    
   
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=1)
    
    train_data, valid_data = create_dataset_for_train(dataset_path, embedding_path)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True,
                                                collate_fn=collate,num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.test_batch_size, shuffle=False,
                                                collate_fn=collate,num_workers=4)

    
    train_losses = []
    val_losses = []
    epoch_num = []
    
    train_start_time = time.time()
    train_epoch = 0
    best_pearson = args.best_pearson
    best_rmse = args.best_rmse
    stop_epoch = args.stop_epoch
    for epoch in tqdm(range(args.num_epochs)):
        train_epoch += 1
        start_time = time.time()
        print('Start training at epoch: {}'.format(epoch + 1))

        train_loss = train(model, device, train_loader, optimizer, epoch + 1)
        val_loss, T_pK, S_pK, T_pAC50, S_pAC50= evaluate(model, device, valid_loader)
        
        scheduler.step(val_loss)
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Valid/Loss', val_loss, epoch)
        epoch_num.append(epoch+1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        current_rmse_pK, current_mse_pK, current_pearson_pK, current_spearman_pK, current_r2_pK, current_rm2_pK, current_ci_pK, current_cindex_pK = performance_evaluation(T_pK, S_pK)
        current_rmse_pAC50, current_mse_pAC50, current_pearson_pAC50, current_spearman_pAC50, current_r2_pAC50, current_rm2_pAC50, current_ci_pAC50, current_cindex_pAC50 = performance_evaluation(T_pAC50, S_pAC50)
        current_rmse = (current_rmse_pK + current_rmse_pAC50) / 2
        current_pearson = (current_pearson_pK + current_pearson_pAC50) / 2
        METRICS = [str(epoch+1),str(format(time.time()-start_time, '.1f')),str(format(train_loss, '.4f')),str(format(val_loss, '.4f')),str(format(current_rmse, '.4f')),str(format(current_pearson, '.4f'))]
        print('epoch\ttime\ttrain_loss\tval_loss\tRMSE_all\tPearson_all')
        print('\t'.join(map(str, METRICS)))
        print('Val_pK RMSE:\t{}'.format(current_rmse_pK))
        print('Val_pK Spearman:\t{}'.format(current_spearman_pK))
        print('Val_pK R2:\t{}'.format(current_r2_pK))
        print('Val_pK RM2:\t{}'.format(current_rm2_pK))
        print('Val_pK CI_index:\t{}'.format(current_cindex_pK))
        print('Val_pAC50 RMSE:\t{}'.format(current_rmse_pAC50))
        print('Val_pAC50 Spearman:\t{}'.format(current_spearman_pAC50))
        print('Val_pAC50 R2:\t{}'.format(current_r2_pAC50))
        print('Val_pAC50 RM2:\t{}'.format(current_rm2_pAC50))
        print('Val_pAC50 CI_index:\t{}'.format(current_cindex_pAC50))
        
        if current_rmse <= best_rmse:
            stop_epoch = 0
            best_rmse = current_rmse
            best_epoch = epoch + 1
            best_model = copy.deepcopy(model)
            best_model_state = {'net': best_model.state_dict(),'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(best_model_state, best_model_name)
        else:
            stop_epoch += 1

      
        
        model_file_dir_epoch_nowname = model_file_dir_epoch + 'model_epoch_' + str(epoch+1) + '.pt'
        if (epoch + 1) %  20 == 0: #10
            # Save the model only in the rank 0 process
            model_state = {'net': model.state_dict(),'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(model_state, model_file_dir_epoch_nowname)
            

        if stop_epoch == args.early_stop:
            print('(EARLY STOP) No improvement since epoch ', best_epoch, '; best_test_pearson', best_pearson)
            break

        torch.cuda.empty_cache()

    print('Finish training!')
    print('Total training time: {:.5f} hours'.format((time.time()-train_start_time)/3600))
    loss_df = pd.DataFrame({'epoch':epoch_num,'train_loss':train_losses,'val_loss':val_losses})
    loss_df.to_csv(log_dir + 'loss.csv',index=False)

    # Record the model result of the last epoch
    start_time = time.time()
    print('Last epoch test results: {}'.format(train_epoch))
    val_loss, T_pK, S_pK, T_pAC50, S_pAC50 = evaluate(model, device, valid_loader)
    rmse_pK, mse_pK, pearson_pK, spearman_pK, r2_pK, rm2_pK, ci_pK, cindex_pK = performance_evaluation(T_pK, S_pK)
    rmse_pAC50, mse_pAC50, pearson_pAC50, spearman_pAC50, r2_pAC50, rm2_pAC50, ci_pAC50, cindex_pAC50 = performance_evaluation(T_pAC50, S_pAC50)
    rmse = (rmse_pK + rmse_pAC50) / 2
    mse = (mse_pK + mse_pAC50) / 2
    pearson = (pearson_pK + pearson_pAC50) / 2
    spearman = (spearman_pK + spearman_pAC50) / 2
    r2 = (r2_pK + r2_pAC50) / 2
    rm2 = (rm2_pK + rm2_pAC50) / 2
    ci = (ci_pK + ci_pAC50) / 2
    cindex = (cindex_pK + cindex_pAC50) / 2
    # rmse, mse, pearson, spearman, r2, rm2, ci, cindex = performance_evaluation(T, S)
    print('RMSE:\t{}'.format(rmse))
    print('MSE:\t{}'.format(mse))
    print('Pearson:\t{}'.format(pearson))
    print('Spearman:\t{}'.format(spearman))
    print('R2:\t{}'.format(r2))
    print('RM2:\t{}'.format(rm2))
    print('Ci:\t{}'.format(ci))
    print('Cindex:\t{}'.format(cindex))
    print('Took {:.5f}s.'.format(time.time() - start_time))

    # Record the results of the best epoch model
    start_time = time.time()
    print('Best epoch test results: {}'.format(best_epoch))
    # val_loss, T, S = evaluate(best_model, device, valid_loader)
    val_loss, T_pK, S_pK, T_pAC50, S_pAC50 = evaluate(best_model, device, valid_loader)
    rmse_pK, mse_pK, pearson_pK, spearman_pK, r2_pK, rm2_pK, ci_pK, cindex_pK = performance_evaluation(T_pK, S_pK)
    rmse_pAC50, mse_pAC50, pearson_pAC50, spearman_pAC50, r2_pAC50, rm2_pAC50, ci_pAC50, cindex_pAC50 = performance_evaluation(T_pAC50, S_pAC50)
    rmse = (rmse_pK + rmse_pAC50) / 2
    mse = (mse_pK + mse_pAC50) / 2
    pearson = (pearson_pK + pearson_pAC50) / 2
    spearman = (spearman_pK + spearman_pAC50) / 2
    r2 = (r2_pK + r2_pAC50) / 2
    rm2 = (rm2_pK + rm2_pAC50) / 2
    ci = (ci_pK + ci_pAC50) / 2
    cindex = (cindex_pK + cindex_pAC50) / 2
    # rmse, mse, pearson, spearman, r2, rm2, ci, cindex = performance_evaluation(T, S)
    print('RMSE:\t{}'.format(rmse))
    print('MSE:\t{}'.format(mse))
    print('Pearson:\t{}'.format(pearson))
    print('Spearman:\t{}'.format(spearman))
    print('R2:\t{}'.format(r2))
    print('RM2:\t{}'.format(rm2))
    print('Ci:\t{}'.format(ci))
    print('Cindex:\t{}'.format(cindex))
    print('Took {:.5f}s.'.format(time.time() - start_time))


#['GPCR','IonChannel','Enzyme','Kinase','NHR','Transporter','Other']
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:0',type=str,help='device id (0,1,2,3)')
    parser.add_argument('--dataset', type=str, default='Other',choices=['GPCR','IonChannel','Enzyme','Kinase','NHR','Transporter','Other'])
    parser.add_argument('--train_batch_size', type=int, default=128) #64
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=200) #100
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dropout_global', type=float, default=0)
    parser.add_argument('--train_val_ratio', type=float, default=0.9)
    parser.add_argument('--early_stop', type=int, default=25) #10
    parser.add_argument('--stop_epoch', type=int, default=0)
    parser.add_argument('--best_epoch', type=int, default=-1)
    parser.add_argument('--best_pearson', type=float, default=0)
    parser.add_argument('--best_rmse', type=float, default=float('inf'))
    parser.add_argument('--last_epoch', type=int, default=1)
    parser.add_argument('--best_model', type=str, default=None)
    parser.add_argument('--modulator_emb_dim', type=int, default=256)
    parser.add_argument('--ppi_emb_dim', type=int, default=512)
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--protein_embedding', type=str, default='esm2',choices=['onehot','tape', 'esm1b', 'esm2','esm2_1024'])
    parser.add_argument('--emb_size', type=int, default=1280,choices=[20, 768, 1280, 1280])
    parser.add_argument('--data-root', required=True, help='Directory containing the seven target-family datasets')
    parser.add_argument('--output-root', required=True, help='Directory in which checkpoints and logs are written')
    args = parser.parse_args()
  
    main(args)


# nohup python -u train_reg_train.py --dataset GPCR --device cuda:0 > train_reg_train_GPCR.log 2>&1 &
# nohup python -u train_reg_train.py --dataset IonChannel --device cuda:6 > train_reg_train_IonChannel.log 2>&1 &
# nohup python -u train_reg_train.py --dataset Enzyme --device cuda:0 > train_reg_train_Enzyme.log 2>&1 &
# nohup python -u train_reg_train.py --dataset Kinase --device cuda:0 > train_reg_train_Kinase.log 2>&1 &
# nohup python -u train_reg_train.py --dataset NHR --device cuda:3 > train_reg_train_NHR.log 2>&1 &
# nohup python -u train_reg_train.py --dataset Transporter --device cuda:4 > train_reg_train_Transporter.log 2>&1 &
# nohup python -u train_reg_train.py --dataset Other --device cuda:3 > train_reg_train_Other.log 2>&1 &
