import os
import pandas as pd
import pickle
import train.train_model as models
from data.data_process import process,data_process_loader

for i in range(1,11):
    config = {
        'data_type':'MDTips', # MDTips, Structure_KG, Structure_Expr,KG_Expr,Structure,KG,Expr
        'binary':True,
        'KG_path':'./CV10/KG_Space/sample2/fold'+str(i)+'/KG',
        'landmark':1,
        'result_folder':'./result/KG_Space/sample2/fold'+str(i)+'/Structure',
        'LR':1e-3,
        'decay':0,
        'batch_size':128,
        'train_epoch':100
    }
    train = pd.read_csv('./CV10/KG_Space/sample2/fold'+str(i)+'/train.txt',sep='\t')
    test = pd.read_csv('./CV10/KG_Space/sample2/fold'+str(i)+'/test.txt',sep='\t')
    val = pd.read_csv('./CV10/KG_Space/sample2/fold'+str(i)+'/val.txt',sep='\t')
    train, val, test = process(train,val,test,**config)
    MDTips = models.model_initialize(**config)
    MDTips.train(train,val,test)
    MDTips.save_model(config['result_folder'])








