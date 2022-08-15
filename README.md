# MDTips: A Multimodal-data based Drug Target interaction prediction system fusing knowledge, gene expression profile and structural data
-----------------------------------------------------------------
Code by Xiaoqiong Xia in the Institutes of Biomedical Sciences at Fudan University.

## 1. Introduction
This repository contains source code (**MDTips**) and data for paper "[MDTips: A Multimodal-data based Drug Target interaction prediction system fusing knowledge, gene expression profile and structural data]

**MDTips** is a Python implementation of the knowledge-driven deep learning model, which contains 4 DTI models: KL, K, SL, and S, and selects the optimal model to predict DTIs, the drug's side effects, indications and pharmacologic classes, and disease-related genes, diseases, symptoms, and drugs under different scenarios.

**MDTips**  yielded accurate and robust performance on DTI prediction tasks and was quickly used to predict related information of extensive drugs, only providing the drug's structural data. In summary, **MDTips** could be a powerful tool for drug repurposing and development.

## 2. Pipeline

![alt text](docs/Figure1.jpg "Pipeline")

The pipeline of building DTI models. (A) The overlapping statuses of DTPs, drugs and targets in the DRKG space and the LINCS space. (B) Data partitioning in 10-fold cross-validation. The data were randomly divided into a training set, a validation set and a test set in the ratio of 8:1:1. The schematic workflow of three single models that (C) K, (D) S, (E) L, and (F) the 4 fusion models. 

## 3. MDTips

![alt text](docs/Figure2.jpg "MDTips")

Figure 2: Overall architecture of **MDTips**. 
**DeepCE** is used for predicting DTIs, the drug's side effects, indications and pharmacologic classes, and disease-related genes, diseases, symptoms, and drugs under different scenarios.

## 4. Installation

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

The datasets used to train **MDTips** are located at folder ``MDTips/CV10/``

### 5.2. Training MDTips

### 5.2.1 10-fold cross-validation

The training codes (10-fold cross-validation) for **MDTips** are located at folder ``MDTips/code/cv10``

s1_Structure.py (S model)  
s2_L1000.py (L model)  
s3_KGE.py (K model)  
s4_Structure_L1000.py (SL model)  
s5_KG_Structure.py (KS model)  
s6_KG_L1000.py (KL model)  
s7_KG_Structure_L1000.py (KSL model)  

You need to train 3 single models before training 4 fusion models and compare the performance (AUROC & AUPR) of 7 models in LINCS Space. 
Then, you need to train and compare the S, K, and KS models in KG Space.

### 5.2.2 training final models

The training codes for **MDTips** are located at folder ``MDTips/code/train_final_model``

s1_Structure.py (S model)  
s2_KGE.py (K model)  
s3_Structure_L1000.py (SL model)  
s4_KG_L1000.py (KL model)  

You must train the S and K model in KG Space before training the SL and KL model in LINCS Space. The final models will be selected to predict the drug-related and disease-related information. 

### 5.3. predict drug-related and disease-related information

Besides **MDTips** source code, we also publicize the prediction codes and examples (e.g., melphalan, mavacamten, and ovarian cancer).

The codes for getting these information are located at folder ``MDTips/code/prediction``

## 6. References

## 7. Contact

**Xiaoqiong Xia** < 19111510052@fudan.edu.cn >

the Institutes of Biomedical Sciences, Fudan University, China

