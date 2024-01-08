# MDTips: A Multimodal-data based Drug Target interaction prediction system fusing knowledge, gene expression profile and structural data
-----------------------------------------------------------------
Code by Xiaoqiong Xia in the Institutes of Biomedical Sciences at Fudan University.

## 1. Introduction
This repository contains source code (**MDTips**) and data for paper "[MDTips: A Multimodal-data based Drug Target interaction prediction system fusing knowledge, gene expression profile and structural data]

**MDTips** is a Python implementation of the knowledge-driven deep learning model, which integrates the available heterogeneous data (KG, graph/sequence, gene expression profile) into a unified workflow and offers a powerful tool for predicting candidate targets, side effects, and indications of the input drugs..

**MDTips**  yielded accurate and robust performance on DTI prediction tasks and was quickly used to predict related information of extensive drugs, only providing the drug's structural data. In summary, **MDTips** could be a powerful tool for drug repurposing and development.

## 2. Pipeline

![alt text](docs/Figure1.jpg "Pipeline")

The pipeline of building MDTips. (A) The data space of DTP_KS and DTP_KSE, negative samples are generated by under-sampling strategy. Positive DTPs and negative samples are mixed to form dataset_KS and dataset_KSE. (B) The 10-fold cross-validation strategy used in this study. The data were randomly divided into a training set, a validation set and a test set with the ratio of 8:1:1. The schematic workflow of (C) K_representation, (D) S_representaion, (E) E_representation, and (F) multimodal fusion and classifier. 

## 3. Installation

**MDTips** depends on the following packages. You must have them installed before using **MDTips**.

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch  
pip install pandas  
pip install prettytable  
pip install tensorboard  
pip install scikit-learn  
pip install matplotlib  
pip install lifelines  
conda install -c conda-forge rdkit  
conda install -c conda-forge notebook  
conda install git  
conda install -c rmg descriptastorus  
pip install pandas-flavor  
pip install DeepPurpose  
pip install pykeen==1.8.0  
pip install cmapPy  

## 5. Usage

### 5.1. Data

The DTP datasets used to train **MDTips** are located at folder ``MDTips/CV10/``
The heterogeneous data are located at folder ``MDTips/data/``

Download all data from https://doi.org/10.5281/zenodo.7560544

### 5.2. Training MDTips

### 5.2.1 10-fold cross-validation

The training codes (10-fold cross-validation) for **MDTips** is **train_MDTips.py**.
For example, you can run **train_MDTips.py** to train MDTips.

### 5.2.2 training final models

The training codes for **MDTips** is **train_final_model.py**.
For example, you can run **train_final_model.py** to train MDTips.

You must train the K model before training the fusion models.

### 5.3. predict drug-related information

Besides **MDTips** source code, we also publicize the prediction codes, located at folder ``MDTips/code/predict`` 

The codes for getting these information are located at folder ``MDTips/code/prediction``
For example, you can run **predict_MDTips.py** to predict drug-target interactions using MDTips.

## 6. References

## 7. Contact

**Xiaoqiong Xia** < 19111510052@fudan.edu.cn >

the Institutes of Biomedical Sciences, Fudan University, China

