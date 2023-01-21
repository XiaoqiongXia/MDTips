
import pandas as pd
import numpy as np
from torch.utils import data
import pickle
import train.train_model as models
from data.data_process import process,data_process_loader,process_data_final_model,protein2emb_encoder


# Structure model
def get_d_structure(SMILES):
    from dgllife.utils import smiles_to_bigraph, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
    drug_node_featurizer = AttentiveFPAtomFeaturizer()
    drug_bond_featurizer = AttentiveFPBondFeaturizer(self_loop=True)
    from functools import partial
    fc = partial(smiles_to_bigraph, add_self_loop=True)
    D_Structure = fc(smiles = SMILES, node_featurizer = drug_node_featurizer, edge_featurizer = drug_bond_featurizer)
    return D_Structure

def get_p_structure(Seq):
    P_Structure = protein2emb_encoder(Seq)
    return P_Structure

drugs = pd.read_csv('./Dataset/Chemical/ChEMBL/ChEMBL_process_n2304874.tsv',sep='\t')
gene_list = pd.read_csv('./data/Structure/gene_info.txt',sep='\t')
gene_list = gene_list.astype(str)
gene_list['entity'] = ['::'.join(['Gene',ID]) for ID in gene_list['GeneID']]

P_Structure = [get_p_structure(seq) for seq in gene_list['Seq']]
model_type = 'Structure'
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
# DTP = process_data_final_model(DTP_pre,**config)

MDTips = models.model_initialize(**config)
MDTips.load_pretrained('./Final_models/KG_Space/'+model_type+'/model.pt')
params = {'batch_size': 128,
            'shuffle': False,
            'num_workers':0,
            'drop_last': False}
if ('Structure' in MDTips.config['data_type']) | (MDTips.config['data_type'] == 'MDTips'):
    params['collate_fn'] = MDTips.dgl_collate_func

for i in range(1,2304874):
    drug = drugs['ChEMBL ID'][i]
    DTP_pre = pd.DataFrame({'head':np.tile(drug, (len(gene_list),)),'tail':gene_list['entity']})
    DTP = {}
    DTP['Seq'] = gene_list['Seq']
    DTP['df'] = DTP_pre
    DTP['Label'] = np.zeros(len(DTP_pre))
    DTP['D_Structure'] = np.tile(get_d_structure(drugs['SMILES'][i]), (len(gene_list),))
    DTP['P_Structure'] = P_Structure
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