# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 20:53:17 2022

@author: toxicant
"""

import os
import pickle
from collections import OrderedDict
import random
import glob
import pandas as pd
#from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
import dgl
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import networkx as nx
import codecs
from subword_nmt.apply_bpe import BPE


def roc_curve(y_pred, y_label, figure_file, method_name):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py  
    '''
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)   
    fpr = dict()
    tpr = dict() 
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
    roc_auc[0] = auc(fpr[0], tpr[0])
    lw = 2
    plt.plot(fpr[0], tpr[0],
         lw=lw, label= method_name + ' (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    fontsize = 14
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig(figure_file)
    return 

def prauc_curve(y_pred, y_label, figure_file, method_name):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        reference: 
            https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    ''' 
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import auc
    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)    
#   plt.plot([0,1], [no_skill, no_skill], linestyle='--')
    plt.plot(lr_recall, lr_precision, lw = 2, label= method_name + ' (area = %0.2f)' % average_precision_score(y_label, y_pred))
    fontsize = 14
    plt.xlabel('Recall', fontsize = fontsize)
    plt.ylabel('Precision', fontsize = fontsize)
    plt.title('Precision Recall Curve')
    plt.legend()
    plt.savefig(figure_file)
    return 

def words2idx_d():
    vocab_path ="./data/Structure/ESPF/drug_codes_chembl_freq_1500.txt"
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    sub_csv = pd.read_csv("./data/Structure/ESPF/subword_units_map_chembl_freq_1500.csv")
    idx2word = sub_csv['index'].values
    words2idx = dict(zip(idx2word, range(0, len(idx2word))))
    return dbpe, words2idx

def words2idx_p():
    vocab_path = "./data/Structure/ESPF/protein_codes_uniprot_2000.txt"
    bpe_codes_protein = codecs.open(vocab_path)
    pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
    sub_csv = pd.read_csv("./data/Structure/ESPF/subword_units_map_uniprot_2000.csv")
    idx2word = sub_csv['index'].values
    words2idx = dict(zip(idx2word, range(0, len(idx2word))))
    return pbpe, words2idx

def drug2espf(dbpe, words2idx,x):
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx[i] for i in t1])  # index
    except:
        i1 = np.array([0])
    v1 = np.zeros(len(words2idx),)
    v1[i1] = 1
    return v1

def protein2espf(pbpe, words2idx,x):
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx[i] for i in t1])  # index
    except:
        i1 = np.array([0])
    v1 = np.zeros(len(words2idx),)
    v1[i1] = 1
    return v1







