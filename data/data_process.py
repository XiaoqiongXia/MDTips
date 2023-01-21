import pandas as pd
import numpy as np
import pickle
from torch.utils import data
from time import time
import torch
import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer,CanonicalBondFeaturizer
import cmapPy
from cmapPy.pandasGEXpress.parse import parse
import codecs
from subword_nmt.apply_bpe import BPE

vocab_path = "./data/Structure/ESPF/protein_codes_uniprot_2000.txt"
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
#sub_csv = pd.read_csv(dataFolder + '/subword_units_map_protein.csv')
sub_csv = pd.read_csv("./data/Structure/ESPF/subword_units_map_uniprot_2000.csv")
idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

def protein2emb_encoder(x):
    max_p = 545
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
   
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p
        
    return i, np.asarray(input_mask)

def get_d_structure(drug,drug_info):
    from dgllife.utils import smiles_to_bigraph, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
    drug_node_featurizer = AttentiveFPAtomFeaturizer()
    drug_bond_featurizer = AttentiveFPBondFeaturizer(self_loop=True)
    from functools import partial
    fc = partial(smiles_to_bigraph, add_self_loop=True)
    drug = drug.split('::')[1]
    if drug in list(drug_info.keys()):
        D_data = drug_info[drug]
        # get graph structure info of drugs, node feats: 39, bond feats: 11
        try:
            D_data['structure'] = fc(smiles = D_data['SMILES'], node_featurizer = drug_node_featurizer, edge_featurizer = drug_bond_featurizer)
        except:
            D_data['structure'] = ''
    else:
        D_data = {}
        D_data['structure'] = ''
    return D_data['structure']


def get_p_structure(protein,protein_info):
    protein = protein.split('::')[1]
    if protein in list(protein_info.keys()):
        P_data = protein_info[protein]
        P_data['Structure'] = protein2emb_encoder(P_data['Sequence'])
    else:
        P_data = {}
        P_data['Structure'] = ''
    return P_data['Structure']


def get_SMILES(drug,drug_info):
    D_data = drug_info[drug.split('::')[1]]
    return D_data['SMILES']

def get_AAS(protein,protein_info):
    if protein.split('::')[1] in protein_info.keys():
        P_data = protein_info[protein.split('::')[1]]
    return P_data['Sequence']

def get_kg_embs(**config):
    entity = pd.read_csv(config['KG_path']+'/training_triples/entity_to_id.tsv.gz',sep='\t')
    entity_dic = dict(zip(entity['label'],entity['id']))
    entity_representation = np.load(config['KG_path']+'/entity_representation.npy')
    return entity_dic, entity_representation

def get_d_kg(entity_dic, entity_representation, drug,**config):
    if drug in entity_dic.keys():
        kg_id = entity_dic[drug]
        kg = entity_representation[kg_id]
    else:
        kg = np.zeros((200,))
    return kg

def get_p_kg(entity_dic, entity_representation, protein,**config):
    if protein in entity_dic.keys():
        kg_id = entity_dic[protein]
        kg = entity_representation[kg_id]
    else:
        kg = np.zeros((200,))
    return kg


def get_d_profiles(**config):
    expr = pd.read_csv('./data/LINCS/consensi-drugbank.tsv',sep='\t',index_col=0)
    drug_info = expr.index.values
    geneinfo = pd.read_csv('./data/LINCS/geneinfo_beta.txt',sep='\t')
    rid = geneinfo['gene_id'][geneinfo['feature_space']=='landmark'].astype(str)
    if config['landmark'] == 1:
        expr = expr[rid.values]
    return expr,drug_info

def get_d_expr(drug,expr,drug_info,**config):
    drug = drug.split('::')[1]
    n_genes = expr.shape[1]
    if drug in drug_info:
        d_expr = np.array(expr.loc[drug])
    else:
        d_expr =  np.zeros((n_genes, ))
    return d_expr

def get_p_profiles(**config):
    xpr_sig = pd.read_csv('./data/LINCS/consensi-knockdown.tsv',sep='\t',index_col=0)
    oe_sig = pd.read_csv('./data/LINCS/consensi-overexpression.tsv',sep='\t',index_col=0)
    xpr_sig.index = [str(id) for id in xpr_sig.index.values]
    xpr_info = xpr_sig.index.values
    oe_sig.index = [str(id) for id in oe_sig.index.values]
    oe_info = oe_sig.index.values
    geneinfo = pd.read_csv('./data/LINCS/geneinfo_beta.txt',sep='\t')
    rid = geneinfo['gene_id'][geneinfo['feature_space']=='landmark'].astype(str)
    if config['landmark'] == 1:
        xpr_sig = xpr_sig[rid.values]
        oe_sig = oe_sig[rid.values]
    return xpr_sig,oe_sig, xpr_info,oe_info

def get_p_expr(protein,xpr_sig,oe_sig, xpr_info,oe_info,UP,**config):
    protein = protein.split('::')[1]
    n_genes = xpr_sig.shape[1]
    if UP == 1:
        if protein in oe_info:
            p_expr = np.array(oe_sig.loc[protein])
        else:
            p_expr =  np.zeros((n_genes, ))
    if UP == 0:
        if protein in xpr_info:
            p_expr = np.array(xpr_sig.loc[protein])
        else:
            p_expr =  np.zeros((n_genes, ))
    return p_expr



def process(train, val, test,**config):
    data_type = config['data_type']
    train_pro = {}
    val_pro = {}
    test_pro = {}
    t_start = time() 
    df = pd.concat([train,val,test])

    # get drug graph
    with open('./data/Structure/gene_info_dics.pkl', 'rb') as fp:
        protein_info = pickle.load(fp)

    with open('./data/Structure/drug_info_dics.pkl', 'rb') as fp:
        drug_info = pickle.load(fp)

    print('prepare D_graph')
    unique = []
    D_str_item = []
    for item in df['head'].unique():
        d_graph = get_d_structure(item,drug_info)
        unique.append(d_graph)
        if d_graph:
            if d_graph.edata:
                D_str_item.append(item)
    unique_dict_d = dict(zip(df['head'].unique(), unique))

    # get protein graph
    print('prepare P_graph')
    unique = []
    P_str_item = []
    for item in df['tail'].unique():
        p_graph = get_p_structure(item,protein_info)
        unique.append(p_graph)
        if p_graph:
            P_str_item.append(item)
    unique_dict_p = dict(zip(df['tail'].unique(), unique))

    keep_D_item = D_str_item
    keep_P_item = P_str_item
    keep_D_item = pd.DataFrame(keep_D_item,columns=['head'])
    keep_P_item = pd.DataFrame(keep_P_item,columns=['tail'])

    print('keep drugs:' + str(len(keep_D_item)))
    print('keep proteins:' + str(len(keep_P_item)))

    df = pd.merge(df,keep_D_item,on='head')
    df = pd.merge(df,keep_P_item,on='tail')
    train = pd.merge(train,keep_D_item,on='head')
    train_pro['df'] = pd.merge(train,keep_P_item,on='tail')
    val = pd.merge(val,keep_D_item,on='head')
    val_pro['df'] = pd.merge(val,keep_P_item,on='tail')
    test = pd.merge(test,keep_D_item,on='head')
    test_pro['df'] = pd.merge(test,keep_P_item,on='tail')

    print('train drugs:' + str(len(train_pro['df']['head'].unique())))
    print('train proteins:' + str(len(train_pro['df']['tail'].unique())))
    print('train DTPs:' + str(len(train_pro['df'])))

    print('val drugs:' + str(len(val_pro['df']['head'].unique())))
    print('val proteins:' + str(len(val_pro['df']['tail'].unique())))
    print('val DTPs:' + str(len(val_pro['df'])))

    print('test drugs:' + str(len(test_pro['df']['head'].unique())))
    print('test proteins:' + str(len(test_pro['df']['tail'].unique())))
    print('test DTPs:' + str(len(test_pro['df'])))

    if ('Structure' in data_type) | (data_type == 'MDTips'):
        # get drug graph
        print('prepare final D_Structure')
        train_pro['D_Structure'] = [unique_dict_d[i] for i in train_pro['df']['head']]
        val_pro['D_Structure'] = [unique_dict_d[i] for i in val_pro['df']['head']]
        test_pro['D_Structure'] = [unique_dict_d[i] for i in test_pro['df']['head']]

        # get protein graph
        print('prepare final P_graph')
        train_pro['P_Structure'] = [unique_dict_p[i] for i in train_pro['df']['tail']]
        val_pro['P_Structure'] = [unique_dict_p[i] for i in val_pro['df']['tail']]
        test_pro['P_Structure'] = [unique_dict_p[i] for i in test_pro['df']['tail']]

    # get SMILES and AAC
    if config['data_type'] == 'Seq':
        unique = []
        for item in df['head'].unique():
            unique.append(get_SMILES(item,drug_info))
        d_unique_dict = dict(zip(df['head'].unique(), unique))

        unique = []
        for item in df['tail'].unique():
            unique.append(get_AAS(item,protein_info))
        p_unique_dict = dict(zip(df['tail'].unique(), unique))

        train_pro['SMILES'] = [d_unique_dict[i] for i in train_pro['df']['head']]
        val_pro['SMILES'] = [d_unique_dict[i] for i in val_pro['df']['head']]
        test_pro['SMILES'] = [d_unique_dict[i] for i in test_pro['df']['head']]
        train_pro['AAS'] = [p_unique_dict[i] for i in train_pro['df']['tail']]
        val_pro['AAS'] = [p_unique_dict[i] for i in val_pro['df']['tail']]
        test_pro['AAS'] = [p_unique_dict[i] for i in test_pro['df']['tail']]
    
    if ('KG' in data_type) | (data_type == 'MDTips'):
        # get drug kg
        print('prepare D_KG')
        entity_dic, entity_representation = get_kg_embs(**config)
        unique = []
        for item in df['head'].unique():
            unique.append(get_d_kg(entity_dic, entity_representation,item,**config))
        unique_dict = dict(zip(df['head'].unique(), unique))
        train_pro['D_kg'] = [unique_dict[i] for i in train_pro['df']['head']]
        val_pro['D_kg'] = [unique_dict[i] for i in val_pro['df']['head']]
        test_pro['D_kg'] = [unique_dict[i] for i in test_pro['df']['head']]

        # get protein kg
        print('prepare P_kg')
        unique = []
        for item in df['tail'].unique():
            unique.append(get_p_kg(entity_dic, entity_representation,item,**config))
        unique_dict = dict(zip(df['tail'].unique(), unique))
        train_pro['P_kg'] = [unique_dict[i] for i in train_pro['df']['tail']]
        val_pro['P_kg'] = [unique_dict[i] for i in val_pro['df']['tail']]
        test_pro['P_kg'] = [unique_dict[i] for i in test_pro['df']['tail']]

    if ('Expr' in data_type) | (data_type == 'MDTips'):
        # get drug expr
        print('prepare D_expr')
        expr,drug_info = get_d_profiles(**config)
        D_Expr = []
        D_expr_item = []
        for item in df['head'].unique():
            expr1 = get_d_expr(item,expr,drug_info,**config)
            D_Expr.append(expr1)
            if expr1.any():
                D_expr_item.append(item)
        D_Expr_dic = dict(zip(df['head'].unique(), D_Expr))

        train_pro['D_expr'] = [D_Expr_dic[i] for i in train_pro['df']['head']]
        val_pro['D_expr'] = [D_Expr_dic[i] for i in val_pro['df']['head']]
        test_pro['D_expr'] = [D_Expr_dic[i] for i in test_pro['df']['head']]

        # get protein expr
        print('prepare P_expr')
        xpr_sig,oe_sig, xpr_info,oe_info = get_p_profiles(**config)
        
        def get_p_exprs(df,xpr_sig,oe_sig, xpr_info,oe_info,**config):
            P_expr = []
            for i in range(0,len(df)):
                tail = df['tail'].iloc[i]
                UP = df['UP'].iloc[i]
                P_expr.append(get_p_expr(tail,xpr_sig,oe_sig, xpr_info,oe_info,UP,**config))
            return P_expr

        train_pro['P_expr'] = get_p_exprs(train_pro['df'],xpr_sig,oe_sig, xpr_info,oe_info,**config)
        val_pro['P_expr'] = get_p_exprs(val_pro['df'],xpr_sig,oe_sig, xpr_info,oe_info,**config)
        test_pro['P_expr'] = get_p_exprs(test_pro['df'],xpr_sig,oe_sig, xpr_info,oe_info,**config)
    
    # get Label
    train_pro['Label'] = torch.tensor(train_pro['df']['Label'].values).to(torch.float32)
    val_pro['Label'] = torch.tensor(val_pro['df']['Label'].values).to(torch.float32)
    test_pro['Label'] = torch.tensor(test_pro['df']['Label'].values).to(torch.float32)
    
    t_now = time()
    print("Data process cost time " + str(int(t_now - t_start)/3600)[:7] + " hours")
    return train_pro, val_pro, test_pro


def process_data_final_model(train,**config):
    data_type = config['data_type']
    train_pro = {}
    t_start = time() 
    df = train
    # get drug graph
    with open('./data/Structure/gene_info_dics.pkl', 'rb') as fp:
        protein_info = pickle.load(fp)

    with open('./data/Structure/drug_info_dics.pkl', 'rb') as fp:
        drug_info = pickle.load(fp)

    
    #if config['data_type'] == 'Structure':
    print('prepare D_graph')
    unique = []
    D_str_item = []
    for item in df['head'].unique():
        d_graph = get_d_structure(item,drug_info)
        unique.append(d_graph)
        if d_graph:
            if d_graph.edata:
                D_str_item.append(item)
    unique_dict_d = dict(zip(df['head'].unique(), unique))

    # get protein graph
    print('prepare P_graph')
    unique = []
    P_str_item = []
    for item in df['tail'].unique():
        p_graph = get_p_structure(item,protein_info)
        unique.append(p_graph)
        if p_graph:
            P_str_item.append(item)
    unique_dict_p = dict(zip(df['tail'].unique(), unique))

    keep_D_item = D_str_item
    keep_P_item = P_str_item
    keep_D_item = pd.DataFrame(keep_D_item,columns=['head'])
    keep_P_item = pd.DataFrame(keep_P_item,columns=['tail'])

    print('keep drugs:' + str(len(keep_D_item)))
    print('keep proteins:' + str(len(keep_P_item)))

    df = pd.merge(df,keep_D_item,on='head')
    df = pd.merge(df,keep_P_item,on='tail')
    train = pd.merge(train,keep_D_item,on='head')
    train_pro['df'] = pd.merge(train,keep_P_item,on='tail')


    print('train drugs:' + str(len(train_pro['df']['head'].unique())))
    print('train proteins:' + str(len(train_pro['df']['tail'].unique())))
    print('train DTPs:' + str(len(train_pro['df'])))

    if ('Structure' in data_type) | (data_type == 'MDTips'):
        # get drug graph
        print('prepare final D_Structure')
        train_pro['D_Structure'] = [unique_dict_d[i] for i in train_pro['df']['head']]
        
        # get protein graph
        print('prepare final P_graph')
        train_pro['P_Structure'] = [unique_dict_p[i] for i in train_pro['df']['tail']]
        
    # get SMILES and AAC
    if config['data_type'] == 'Seq':
        unique = []
        for item in df['head'].unique():
            unique.append(get_SMILES(item,drug_info))
        d_unique_dict = dict(zip(df['head'].unique(), unique))

        unique = []
        for item in df['tail'].unique():
            unique.append(get_AAS(item,protein_info))
        p_unique_dict = dict(zip(df['tail'].unique(), unique))

        train_pro['SMILES'] = [d_unique_dict[i] for i in train_pro['df']['head']]
        train_pro['AAS'] = [p_unique_dict[i] for i in train_pro['df']['tail']]
        
    if ('KG' in data_type) | (data_type == 'MDTips'):
        # get drug kg
        print('prepare D_KG')
        entity_dic, entity_representation = get_kg_embs(**config)
        unique = []
        for item in df['head'].unique():
            unique.append(get_d_kg(entity_dic, entity_representation,item,**config))
        unique_dict = dict(zip(df['head'].unique(), unique))
        train_pro['D_kg'] = [unique_dict[i] for i in train_pro['df']['head']]
        
        # get protein kg
        print('prepare P_kg')
        unique = []
        for item in df['tail'].unique():
            unique.append(get_p_kg(entity_dic, entity_representation,item,**config))
        unique_dict = dict(zip(df['tail'].unique(), unique))
        train_pro['P_kg'] = [unique_dict[i] for i in train_pro['df']['tail']]
    if ('Expr' in data_type) | (data_type == 'MDTips'):
        # get drug expr
        print('prepare D_expr')
        expr,drug_info = get_d_profiles(**config)
        D_Expr = []
        D_expr_item = []
        for item in df['head'].unique():
            expr1 = get_d_expr(item,expr,drug_info,**config)
            D_Expr.append(expr1)
            if expr1.any():
                D_expr_item.append(item)
        D_Expr_dic = dict(zip(df['head'].unique(), D_Expr))

        train_pro['D_expr'] = [D_Expr_dic[i] for i in train_pro['df']['head']]
        # get protein expr
        print('prepare P_expr')
        xpr_sig,oe_sig, xpr_info,oe_info = get_p_profiles(**config)
        
        def get_p_exprs(df,xpr_sig,oe_sig, xpr_info,oe_info,**config):
            P_expr = []
            for i in range(0,len(df)):
                tail = df['tail'].iloc[i]
                UP = df['UP'].iloc[i]
                P_expr.append(get_p_expr(tail,xpr_sig,oe_sig, xpr_info,oe_info,UP,**config))
            return P_expr

        train_pro['P_expr'] = get_p_exprs(train_pro['df'],xpr_sig,oe_sig, xpr_info,oe_info,**config)
    # get Label
    train_pro['Label'] = torch.tensor(train_pro['df']['Label'].values).to(torch.float32)
    
    t_now = time()
    print("Data process cost time " + str(int(t_now - t_start)/3600)[:7] + " hours")
    return train_pro

class data_process_loader(data.Dataset):

    def __init__(self, df, **config):
        'Initialization'
        self.df = df
        self.config = config
        self.list_IDs = np.array(range(0,len(self.df['df'])))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df['df'])

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        if self.config['data_type'] == 'MDTips':
            d_s = self.df['D_Structure'][index]
            p_s,mask = self.df['P_Structure'][index]
            d_kg = self.df['D_kg'][index]
            p_kg = self.df['P_kg'][index]
            d_expr = self.df['D_expr'][index]
            p_expr = self.df['P_expr'][index]
            y = self.df['Label'][index]
            return d_s, p_s,mask, d_kg,p_kg,d_expr,p_expr,y
        
        if self.config['data_type'] == 'Structure_KG':
            d_s = self.df['D_Structure'][index]
            p_s,mask = self.df['P_Structure'][index]
            d_kg = self.df['D_kg'][index]
            p_kg = self.df['P_kg'][index]
            y = self.df['Label'][index]
            return d_s, p_s,mask, d_kg,p_kg,y
        
        if self.config['data_type'] == 'Structure_Expr':
            d_s = self.df['D_Structure'][index]
            p_s,mask = self.df['P_Structure'][index]
            d_expr = self.df['D_expr'][index]
            p_expr = self.df['P_expr'][index]
            y = self.df['Label'][index]
            return d_s, p_s,mask, d_expr,p_expr, y
        
        if self.config['data_type'] == 'KG_Expr':
            d_kg = self.df['D_kg'][index]
            p_kg = self.df['P_kg'][index]
            d_expr = self.df['D_expr'][index]
            p_expr = self.df['P_expr'][index]
            y = self.df['Label'][index]
            return d_kg,p_kg,d_expr,p_expr, y
        
        if self.config['data_type'] == 'Structure':
            d_s = self.df['D_Structure'][index]
            p_s,mask = self.df['P_Structure'][index]
            y = self.df['Label'][index]
            return d_s, p_s,mask, y
        
        if self.config['data_type'] == 'KG':
            d_kg = self.df['D_kg'][index]
            p_kg = self.df['P_kg'][index]
            y = self.df['Label'][index]
            return d_kg,p_kg, y

        if self.config['data_type'] == 'Expr':
            d_expr = self.df['D_expr'][index]
            p_expr = self.df['P_expr'][index]
            y = self.df['Label'][index]
            return d_expr,p_expr, y




































