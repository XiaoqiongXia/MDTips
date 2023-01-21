import os 
import pandas as pd
import numpy as np
import math
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# KG representation
# KG Space
path = './CV10/KG_Space/sample2/fold1/KG'
train = pd.read_csv(path+'/train.txt',sep='\t') 
test = pd.read_csv(path+'/test.txt',sep='\t')
val = pd.read_csv(path+'/val.txt',sep='\t')
KG = pd.concat([train,val],axis=0)  

# best_epoch = [51,56,51,45,46,61,62,71,71,43]  #mean=56

KG_folder = './Final_models/KG_Space/KG'
KG.to_csv(KG_folder+'/KG.txt',sep='\t',index=False)
test.to_csv(KG_folder+'/test.txt',sep='\t',index=False)


training = TriplesFactory.from_path(KG_folder+'/KG.txt',create_inverse_triples=True)
testing = TriplesFactory.from_path(KG_folder+'/test.txt',
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,
    create_inverse_triples=True
    )
pipeline_result = pipeline(  
    training = training,
    testing = testing,
    validation = None,
    model = 'ConvE',
    model_kwargs = {
        "embedding_dim": 200,
        "input_channels": 1,
        "output_channels": 32,
        "embedding_height": 10,
        "embedding_width": 20,
        "kernel_height": 3,
        "kernel_width": 3,
        "input_dropout": 0.2,
        "feature_map_dropout": 0.2,
        "output_dropout": 0.3,
        "apply_batch_normalization": True,
        "entity_initializer": "xavier_normal",
        "relation_initializer": "xavier_normal"
    },
    loss = 'BCEAfterSigmoidLoss',
    loss_kwargs = {
        "reduction": "mean"
    },
    optimizer = 'Adam',
    optimizer_kwargs = {
        "lr": 0.001
    },
    training_loop = 'LCWA',
    training_kwargs = {
    "num_epochs": 56,
    "batch_size": 256,
    "label_smoothing": 0.1,
    "checkpoint_name":KG_folder+'/checkpoint.pt',
    },
    evaluator_kwargs = {
    "filtered": True
    },
    use_tqdm = True,
    random_seed = 2022,
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    evaluation_fallback = True
)
pipeline_result.save_to_directory(KG_folder)


K_model = torch.load('./Final_models/KG_Space/KG/trained_model.pkl')
entity_representation = K_model.entity_embeddings().detach().cpu().numpy()
np.save('./Final_models/KG_Space/KG/entity_representation.npy',entity_representation)

import os
import pandas as pd
import pickle
import sys
import numpy as np
import math
import train.train_model as models
from data.data_process import process_data_final_model,data_process_loader


def get_best_epoch(path,model_type):
    best_epoch = []
    for i in range(1,11):
            result_folder =path+str(i)+'/'+model_type+'/valid_markdowntable.txt'
            res = pd.read_csv(result_folder)
            roc = []
            for j in range(2,len(res)-1):
                roc.append(float(res.iloc[j,0].split('|')[2]))
            best_epoch.append(pd.DataFrame(roc).rank().iloc[-1].values[0])
    best_epoch = math.ceil(np.mean(best_epoch))
    return best_epoch
## KG Space
# Structure # 34
# best_epoch = get_best_epoch('./result/KG_Space/sample2/fold','Structure') #34

config = {
    'data_type':'Structure', 
    'binary':True,
    'landmark':1,
    'KG_path':'',
    'result_folder':'./Final_models/KG_Space/Structure',
    'LR':1e-3,
    'decay':0,
    'batch_size':128,
    'train_epoch':30
}
train = pd.read_csv('./CV10/KG_Space/sample2/fold1/train.txt',sep='\t')
test = pd.read_csv('./CV10/KG_Space/sample2/fold1/test.txt',sep='\t')
val = pd.read_csv('./CV10/KG_Space/sample2/fold1/val.txt',sep='\t')
df = pd.concat([train,val,test],axis=0)
df = process_data_final_model(df,**config)
MDTips = models.model_initialize(**config)
MDTips.train(df)
MDTips.save_model(config['result_folder'])

# KG Space
# Structure_KG
# best_epoch = get_best_epoch('./result/KG_Space/sample2/fold','Structure_KG') #6
config = {
    'data_type':'Structure_KG', 
    'binary':True,
    'landmark':1,
    'KG_path':'./Final_models/KG_Space/KG',
    'result_folder':'./Final_models/KG_Space/Structure_KG',
    'LR':1e-3,
    'decay':0,
    'batch_size':128,
    'train_epoch':4
}
train = pd.read_csv('./CV10/KG_Space/sample2/fold1/train.txt',sep='\t')
test = pd.read_csv('./CV10/KG_Space/sample2/fold1/test.txt',sep='\t')
val = pd.read_csv('./CV10/KG_Space/sample2/fold1/val.txt',sep='\t')
df = pd.concat([train,val,test],axis=0)
df = process_data_final_model(df,**config)
MDTips = models.model_initialize(**config)
MDTips.train(df)
MDTips.save_model(config['result_folder'])

## LINCS Space
# MDTips
# best_epoch = get_best_epoch('./result/LINCS_Space/sample2/fold','MDTips') #7

config = {
    'data_type':'MDTips', 
    'binary':True,
    'KG_path':'./Final_models/LINCS_Space/KG',
    'landmark':1,
    'result_folder':'./Final_models/LINCS_Space/MDTips',
    'LR':1e-3,
    'decay':0,
    'batch_size':128,
    'train_epoch':7
}
train = pd.read_csv('./CV10/LINCS_Space/sample2/fold1/train.txt',sep='\t')
test = pd.read_csv('./CV10/LINCS_Space/sample2/fold1/test.txt',sep='\t')
val = pd.read_csv('./CV10/LINCS_Space/sample2/fold1/val.txt',sep='\t')
df = pd.concat([train,val,test],axis=0)
df = process_data_final_model(df,**config)
MDTips = models.model_initialize(**config)
MDTips.train(df)
MDTips.save_model(config['result_folder'])


