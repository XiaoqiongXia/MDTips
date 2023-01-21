import pandas as pd
import numpy as np
from torch.utils import data
import pickle
import train.train_model as models
from data.data_process import process,data_process_loader,process_data_final_model,protein2emb_encoder


# drug info: Structure, KG
DTP = pd.read_csv('./CV10/DTP_n88439.txt',sep='\t')
drugs = DTP['head'].unique() # 6766
genes = DTP['tail'].unique()

# 
gene_list = pd.read_csv('./data/Structure/gene_info.txt',sep='\t')
gene_list = gene_list.astype(str)
gene_list['entity'] = ['::'.join(['Gene',ID]) for ID in gene_list['GeneID']]


for i in range(0,len(drugs)):
    drug = drugs[i]
    DTP_pre = pd.DataFrame({'head':np.tile(drug, (len(gene_list),)),'tail':gene_list['entity']})
    DTP_pre['Label'] = np.zeros(len(DTP_pre))
    # DTP_pre['UP'] = np.ones(len(DTP_pre))
    model_type = 'Structure_KG'
    config = {
        'data_type':model_type, 
        'binary':True,
        'KG_path':'./Final_models/KG_Space/KG',
        'landmark':1,
        'result_folder':'./Final_models/KG_Space/'+model_type,
        'LR':1e-3,
        'decay':0,
        'batch_size':128,
        'train_epoch':4
    }
    DTP = process_data_final_model(DTP_pre,**config)
    MDTips = models.model_initialize(**config)
    MDTips.load_pretrained('./Final_models/KG_Space/'+model_type+'/model.pt')
    params = {'batch_size': 64,
                'shuffle': False,
                'num_workers':0,
                'drop_last': False}
    if ('Structure' in MDTips.config['data_type']) | (MDTips.config['data_type'] == 'MDTips'):
        params['collate_fn'] = MDTips.dgl_collate_func
    data_generator = data.DataLoader(data_process_loader(DTP, **MDTips.config), **params)
    score = MDTips.predict(data_generator)
    DTP = DTP['df'][['head','tail']]
    DTP['score'] = score
    DTP = DTP.sort_values("score",ascending=False)
    DTP = pd.merge(DTP,gene_list[['Symbol','entity']],left_on='tail',right_on='entity')
    DTP = DTP[['head','tail','score','Symbol']]
    DTP.to_csv('./screeing/'+model_type+'/'+drug+'_DTP.csv')
    DTP_sig = DTP[DTP['score']>0.9]
    print(drug)
    print(len(DTP_sig))
