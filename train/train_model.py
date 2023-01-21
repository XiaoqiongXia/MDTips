import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import SequentialSampler
from torch import nn 

#from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
from time import time
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, log_loss
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
import pickle 
import copy
from prettytable import PrettyTable
from data.data_process import data_process_loader
from model.model import DTI
from utils.utils import roc_curve, prauc_curve
from torch.utils.tensorboard import SummaryWriter

# 模型初始化
def model_initialize(**config):
    model = MDTips(**config)
    return model


class MDTips:
    '''
        Drug Target interaction 
    '''
    def __init__(self, **config):
        super(MDTips, self).__init__()
        self.config = config
        self.result_folder = config['result_folder']
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        self.binary = config['binary']
        if 'num_workers' not in self.config.keys():
            self.config['num_workers'] = 0
        if 'decay' not in self.config.keys():
            self.config['decay'] = 0
        if 'cuda_id' in self.config:
            if self.config['cuda_id'] is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device('cuda:' + str(self.config['cuda_id']) if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config['device'] = self.device
        # a MDTips model
        self.model = DTI(**config)
    
    def dgl_collate_func(self,x):
        import dgl
        if self.config['data_type'] == 'MDTips':
            d_s,p_s,mask,d_kg,p_kg,d_expr,p_expr, y = zip(*x)
            d_s = dgl.batch(d_s) # 将graph打包成batch
            return d_s,torch.tensor(p_s),torch.tensor(mask), torch.tensor(d_kg),torch.tensor(p_kg),torch.tensor(d_expr),torch.tensor(p_expr),torch.tensor(y)
        
        if self.config['data_type'] == 'Structure_KG':
            d_s,p_s,mask,d_kg,p_kg, y = zip(*x)
            d_s = dgl.batch(d_s) # 将graph打包成batch
            return d_s,torch.tensor(p_s),torch.tensor(mask), torch.tensor(d_kg),torch.tensor(p_kg),torch.tensor(y)
        
        if self.config['data_type'] == 'Structure_Expr':
            d_s,p_s,mask,d_expr,p_expr, y = zip(*x)
            d_s = dgl.batch(d_s) # 将graph打包成batch
            return d_s,torch.tensor(p_s),torch.tensor(mask), torch.tensor(d_expr),torch.tensor(p_expr),torch.tensor(y)
        
        if self.config['data_type'] == 'Structure':
            d_s,p_s,mask, y = zip(*x)
            d_s = dgl.batch(d_s) # 将graph打包成batch
            return d_s,torch.tensor(p_s),torch.tensor(mask),torch.tensor(y)
    
    def cal_score(self,feat):
        if self.config['data_type'] == 'MDTips':
            d_s = feat[0].to(self.device)
            p_s = feat[1].to(self.device)
            mask = feat[2].to(self.device)
            d_kg = feat[3].to(torch.float32).to(self.device)
            p_kg = feat[4].to(torch.float32).to(self.device)
            d_expr = feat[5].to(torch.float32).to(self.device)
            p_expr = feat[6].to(torch.float32).to(self.device)
            label = feat[7].to(self.device)
            score = self.model((d_s,p_s,mask,d_kg,p_kg,d_expr,p_expr))
            
        if self.config['data_type'] == 'Structure_KG':
            d_s = feat[0].to(self.device)
            p_s = feat[1].to(self.device)
            mask = feat[2].to(self.device)
            d_kg = feat[3].to(torch.float32).to(self.device)
            p_kg = feat[4].to(torch.float32).to(self.device)
            label = feat[5].to(self.device)
            score = self.model((d_s,p_s,mask,d_kg,p_kg))
        
        if self.config['data_type'] == 'Structure_Expr':
            d_s = feat[0].to(self.device)
            p_s = feat[1].to(self.device)
            mask = feat[2].to(self.device)
            d_expr = feat[3].to(torch.float32).to(self.device)
            p_expr = feat[4].to(torch.float32).to(self.device)
            label = feat[5].to(self.device)
            score = self.model((d_s,p_s,mask,d_expr,p_expr))
        
        if self.config['data_type'] == 'KG_Expr':
            d_kg = feat[0].to(torch.float32).to(self.device)
            p_kg = feat[1].to(torch.float32).to(self.device)
            d_expr = feat[2].to(torch.float32).to(self.device)
            p_expr = feat[3].to(torch.float32).to(self.device)
            label = feat[4].to(self.device)
            score = self.model((d_kg,p_kg,d_expr,p_expr))
        
        if self.config['data_type'] == 'Structure':
            d_s = feat[0].to(self.device)
            p_s = feat[1].to(self.device)
            mask = feat[2].to(self.device)
            label = feat[3].to(self.device)
            score = self.model((d_s,p_s,mask))
        
        if self.config['data_type'] == 'KG':
            d_kg = feat[0].to(torch.float32).to(self.device)
            p_kg = feat[1].to(torch.float32).to(self.device)
            label = feat[2].to(self.device)
            score = self.model((d_kg,p_kg))
        
        if self.config['data_type'] == 'Expr':
            d_expr = feat[0].to(torch.float32).to(self.device)
            p_expr = feat[1].to(torch.float32).to(self.device)
            label = feat[2].to(self.device)
            score = self.model((d_expr,p_expr))
        
        return score, label
        

    def test_(self, data_generator, model, repurposing_mode = False, test = False):
        y_pred = []
        y_label = []
        model.eval()
        for i, feat in enumerate(data_generator):
            score,label = self.cal_score(feat)

            if self.binary:
                m = torch.nn.Sigmoid()
                logits = torch.squeeze(m(score)).detach().cpu().numpy()
            else:
                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1)
                loss = loss_fct(n, Variable(torch.from_numpy(np.array(label)).float()).to(self.device))
                logits = torch.squeeze(score).detach().cpu().numpy()
            
            label_ids = Variable(label)
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
        model.train()
        if self.binary:
            if repurposing_mode:
                return y_pred
            ## ROC-AUC curve
            if test:
                roc_auc_file = os.path.join(self.result_folder, "roc-auc.jpg")
                plt.figure(0,dpi=600)
                roc_curve(y_pred, y_label, roc_auc_file, 'MDTips')
                plt.figure(1,dpi=600)
                pr_auc_file = os.path.join(self.result_folder, "pr-auc.jpg")
                prauc_curve(y_pred, y_label, pr_auc_file, 'MDTips')

            return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), log_loss(y_label, outputs), y_pred
        else:
            if repurposing_mode:
                return y_pred
            return mean_squared_error(y_label, y_pred), pearsonr(y_label, y_pred)[0], pearsonr(y_label, y_pred)[1], concordance_index(y_label, y_pred), y_pred, loss

    def train(self, train, val = None, test = None, verbose = True):
        lr = self.config['LR']
        decay = self.config['decay']
        BATCH_SIZE = self.config['batch_size']
        train_epoch = self.config['train_epoch']
        
        loss_history = []

        # support multiple GPUs
        if torch.cuda.device_count() > 1:
            if verbose:
                print("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
            self.model = nn.DataParallel(self.model, dim = 0)
        elif torch.cuda.device_count() == 1:
            if verbose:
                print("Let's use " + str(torch.cuda.device_count()) + " GPU!")
        else:
            if verbose:
                print("Let's use CPU/s!")
        # Future TODO: support multiple optimizers with parameters
        opt = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = decay)

        if verbose:
            print('--- Data Preparation ---')
        params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': self.config['num_workers'],
                'drop_last': False}
        if ('Structure' in self.config['data_type']) | (self.config['data_type'] == 'MDTips'):
            params['collate_fn'] = self.dgl_collate_func
        training_generator = data.DataLoader(data_process_loader(train, **self.config), **params)
        if val is not None:
            validation_generator = data.DataLoader(data_process_loader(val, **self.config), **params)
        if test is not None:
            testing_generator = data.DataLoader(data_process_loader(test, **self.config), **params)
        # early stopping
        if self.binary:
            max_auc = 0
        else:
            max_MSE = 10000
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []
        valid_metric_header = ["# epoch"] 
        if self.binary:
            valid_metric_header.extend(["AUROC", "AUPRC", "F1"])
        else:
            valid_metric_header.extend(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
        table = PrettyTable(valid_metric_header)
        float2str = lambda x:'%0.4f'%x
        if verbose:
            print('--- Go for Training ---')
        self.model = self.model.to(self.device)
        self.model.train()
        writer = SummaryWriter()
        t_start = time() 
        iteration_loss = 0
        for epo in range(train_epoch):
            for i, feat in enumerate(training_generator):
                score,label = self.cal_score(feat)
                
                if self.binary:
                    loss_fct = torch.nn.BCELoss()
                    m = torch.nn.Sigmoid()
                    n = torch.squeeze(m(score), 1)
                    loss = loss_fct(n, label)
                else:
                    loss_fct = torch.nn.MSELoss()
                    n = torch.squeeze(score, 1)
                    loss = loss_fct(n, label)
                loss_history.append(loss.item())
                writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()

                if verbose:
                    if (i % 500 == 0):
                        t_now = time()
                        print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + \
                            ' with loss ' + str(loss.cpu().detach().numpy())[:7] +\
                            ". Total time " + str(int(t_now - t_start)/3600)[:7] + " hours") 
                        ### record total run time
            if val is not None:
                ##### validate, select the best model up to now 
                with torch.set_grad_enabled(False):
                    if self.binary:  
                        ## binary: ROC-AUC, PR-AUC, F1, cross-entropy loss
                        auc, auprc, f1, loss, logits = self.test_(validation_generator, self.model)
                        lst = ["epoch " + str(epo)] + list(map(float2str,[auc, auprc, f1]))
                        valid_metric_record.append(lst)
                      
                        if verbose:
                            print('Validation at Epoch '+ str(epo + 1) + ', AUROC: ' + str(auc)[:7] + \
                            ' , AUPRC: ' + str(auprc)[:7] + ' , F1: '+str(f1)[:7] + ' , Cross-entropy Loss: ' + \
                            str(loss)[:7])
                        if auc > max_auc:
                            model_max = copy.deepcopy(self.model)
                            max_auc = auc
                            es = 0
                        else:
                            es += 1
                            print("Counter {} of 5".format(es))
                            if es > 4:
                                print("Early stopping with best_auc: ", str(max_auc)[:7], "and auc for this epoch: ", str(auc)[:7], "...")
                                break

                    else:  
                        ### regression: MSE, Pearson Correlation, with p-value, Concordance Index  
                        mse, r2, p_val, CI, logits, loss_val = self.test_(validation_generator, self.model)
                        lst = ["epoch " + str(epo)] + list(map(float2str,[mse, r2, p_val, CI]))
                        valid_metric_record.append(lst)
                        
                        if verbose:
                            print('Validation at Epoch '+ str(epo + 1) + ' with loss:' + str(loss_val.item())[:7] +', MSE: ' + str(mse)[:7] + ' , Pearson Correlation: '\
                            + str(r2)[:7] + ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI)[:7])
                            writer.add_scalar("valid/mse", mse, epo)
                            writer.add_scalar("valid/pearson_correlation", r2, epo)
                            writer.add_scalar("valid/concordance_index", CI, epo)
                            writer.add_scalar("Loss/valid", loss_val.item(), iteration_loss)
                        if mse < max_MSE:
                            model_max = copy.deepcopy(self.model)
                            max_MSE = mse
                            es = 0
                        else:
                            es += 1
                            print("Counter {} of 5".format(es))
                            if es > 4:
                                print("Early stopping with best_mse: ", str(max_MSE)[:7], "and mse for this epoch: ", str(mse)[:7], "...")
                                break

                table.add_row(lst)
            else:
                model_max = copy.deepcopy(self.model)

        # load early stopped model
        self.model = model_max
        
        if val is not None:
            #### after training 
            prettytable_file = os.path.join(self.result_folder, "valid_markdowntable.txt")
            with open(prettytable_file, 'w') as fp:
                fp.write(table.get_string())

        if test is not None:
            if verbose:
                print('--- Go for Testing ---')
            if self.binary:
                auc, auprc, f1, loss, logits = self.test_(testing_generator, self.model, test = True)
                test_table = PrettyTable(["AUROC", "AUPRC", "F1"])
                test_table.add_row(list(map(float2str, [auc, auprc, f1])))
                if verbose:
                    print('Validation at Epoch '+ str(epo + 1) + ' , AUROC: ' + str(auc)[:7] + \
                      ' , AUPRC: ' + str(auprc)[:7] + ' , F1: '+str(f1)[:7] + ' , Cross-entropy Loss: ' + \
                      str(loss)[:7])                
            else:
                mse, r2, p_val, CI, logits, loss_test = self.test_(testing_generator, self.model)
                test_table = PrettyTable(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
                test_table.add_row(list(map(float2str, [mse, r2, p_val, CI])))
                if verbose:
                    print('Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2) 
                      + ' with p-value: ' + str(f"{p_val:.2E}") +' , Concordance Index: '+str(CI))
            np.save(os.path.join(self.result_folder, 'test'  + '_logits.npy'), np.array(logits))                
    
            ######### learning record ###########

            ### 1. test results
            prettytable_file = os.path.join(self.result_folder, "test_markdowntable.txt")
            with open(prettytable_file, 'w') as fp:
                fp.write(test_table.get_string())

        ### 2. learning curve 
        fontsize = 16
        iter_num = list(range(1,len(loss_history)+1))
        plt.figure(3,dpi=600)
        plt.plot(iter_num, loss_history, "bo-")
        plt.xlabel("iteration", fontsize = fontsize)
        plt.ylabel("loss value", fontsize = fontsize)
        pkl_file = os.path.join(self.result_folder, "loss_curve_iter.pkl")
        with open(pkl_file, 'wb') as pck:
            pickle.dump(loss_history, pck)

        fig_file = os.path.join(self.result_folder, "loss_curve.png")
        plt.savefig(fig_file)
        if verbose:
            print('--- Training Finished ---')
            writer.flush()
            writer.close()
          

    def predict(self, df_data):
       
        print('predicting...')
        self.model.to(self.device)
        score = self.test_(df_data, self.model, repurposing_mode = True)
        return score

    def save_dict(self, path, obj):
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def save_model(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        torch.save(self.model.state_dict(), path_dir + '/model.pt')
        self.save_dict(path_dir, self.config)

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        state_dict = torch.load(path, map_location = torch.device('cpu'))
        # state_dict = torch.load(path,map_location = torch.device('cuda'/cuda:0))
        
        # to support training from multi-gpus data-parallel:
        
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)

        self.binary = self.config['binary']



