#!/usr/bin/env python
# coding: utf-8

# In[1]:


from importlib import reload
import nbimporter
from collections import Counter
import ReadBedFiles
from ReadBedFiles import readJsonFile, readFiles2Vector, writeJsonFile, convertMat2document, readJsonFile
import glob
import pandas as pd
import numpy as np
import time
import copy
import gc
from gensim.models import Word2Vec
from sklearn.decomposition import PCA


# In[2]:


import numpy as np
from gensim.models import Word2Vec
import json


# In[3]:


from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.font_manager as font_manager


# In[4]:


def initialization(clas_type, label, path_train, path_test, path_universe, path_mat, meta_data, sample_of_interest, path_w2v_model):
    path_train = path_train.format(clas_type)
    path_test = path_test.format(clas_type)
    path_universe = path_universe.format(clas_type)
    path_mat = path_mat
    meta_data = pd.read_csv(meta_data.format(clas_type))
    meta_data = meta_data.loc[meta_data[label].isin(sample_of_interest)][['Experiment_ID', label]]
    model = Word2Vec.load(path_w2v_model.format(clas_type))
    
    return path_train, path_test, path_universe, path_mat, meta_data, model


# In[5]:


def embedding_avg(model, document):
    listOfWVs= []
    for word in document.split(' '):
        if word in model.wv.vocab:
            listOfWVs.append(model[word])
            
    if(len(listOfWVs) == 0):
        return np.zeros([100])
    return np.mean(listOfWVs, axis=0)

def document_embedding_avg(document_Embedding, model):
    document_Embedding_avg = {}
    for file, doc  in document_Embedding.items():
        document_Embedding_avg[file] = embedding_avg(model, doc)
    return document_Embedding_avg

def create_data4T_SNE(tdMatrix, labels, label, threshld, clm):
    tdMatrix =  {k.lower(): v for k, v in tdMatrix.items()}
#     print(len((set(labels.Experiment_ID))))
    labels.Experiment_ID = list(labels.Experiment_ID.str.lower())
#     print(labels.Experiment_ID[0:10])
    commonFiles = list(set(tdMatrix.keys())  & set(labels.Experiment_ID)) 
#     print(len(commonFiles))
    y = []
    for name in commonFiles:
        y.append(labels[labels[clm] == name][label].tolist()[0])
    list_of_frequent= y #removeElements(y, threshld)
#     print(len(list_of_frequent))
    data_X = []
    y = []
    for name in commonFiles:
        y_1 = labels[labels[clm] == name][label].tolist()[0]
        if(y_1 in list_of_frequent):
            data_X.append(tdMatrix[name])
            y.append(y_1)
    
    return np.array(data_X), y


# # Read data

# In[6]:


i = 0


# In[7]:


clas_types = ['antibody', 'cell', 'tissue']
labels = ['antibody', 'cell line', 'tissue']
sample_of_interest = [['h3k27ac', 'h3k4me3', 'h3k27me3', 'h3k4me1', 'h3k36me3', 'h3k9me3', 'h3k4me2'],
                     ['k562', 'mcf7', 'hek293', 'a549', 'hepg2', 'hct116','lovo', 'gm12878', 'lncap','hela'],
                     ['liver', 'peripheral blood', 'primary prostate cancer', 'blood', 'breast','bone marrow', 'kidney']
                     ]
PCA_flg = False
label = labels[i]
clas_type = clas_types[i]


# In[8]:


results = pd.DataFrame(columns = ['dataset','representation',  'pca', 'f1'])


# In[9]:


path = '/project/shefflab/data/ChIP-Atlas/'
path_train = path + '{}dataset/train/*'
path_test = path + '{}dataset/test/*'
path_universe = './representations/antibody_universe.txt'
path_mat = path + 'datasets/term_doc_mat/'
meta_data = path + 'meta_data_{}.csv'
path_w2v_model = './word2vec.model'

#path_w2v_model = path + 'word2vecmodels/word2vec_{}.model'


# In[10]:


path_train, path_test, path_universe, path_mat, meta_data, model = initialization(clas_type, label, path_train, path_test, path_universe, path_mat, meta_data, sample_of_interest[i], path_w2v_model)


# In[11]:


path_train


# In[12]:


get_ipython().run_cell_magic('time', '', "path_univ = './representations/antibody_universe.txt'\n\npca = PCA(n_components = 100)\n\nprint(path_univ)\ntrain_files , segmentation_df_train = readFiles2Vector(path_train, path_univ, numberofCores = 4, numOfFiles= 100, PATH = path_train)\nprint(len(train_files))\n\ntest_files, segmentation_df_test = readFiles2Vector(path_test, path_univ, numberofCores = 4, numOfFiles= 100, PATH = path_test)\nprint(len(test_files))\n\n    \ndocument_Embedding_train = convertMat2document(train_files, segmentation_df_train)\ndocument_Embedding_avg_train = document_embedding_avg(document_Embedding_train, model)\n\n\ndocument_Embedding_test = convertMat2document(test_files, segmentation_df_test)\ndocument_Embedding_avg_test = document_embedding_avg(document_Embedding_test, model)\n \n    \nX_train, y_train = create_data4T_SNE(document_Embedding_avg_train, (meta_data.loc[(meta_data[label] !='none') & (meta_data[label] !='nan') & (meta_data[label].notna())]), label, 0, 'Experiment_ID')\nX_test, y_test = create_data4T_SNE(document_Embedding_avg_test, (meta_data.loc[(meta_data[label] !='none') & (meta_data[label] !='nan') & (meta_data[label].notna())]), label, 0, 'Experiment_ID')\n\nprint(len(X_train), len(y_train))\nprint(len(X_test), len(y_test))\n\n# model for classification- this is what we could put into the randomsearch_cv\nclf = svm.SVC(kernel = 'linear')\nstart_time = time.time()\nclf.fit(X_train, y_train)\n\nf1 = f1_score((y_test), clf.predict(X_test), average = 'micro')\n# from clf to here would need to be in main of other file\n# wuld return f1 instead of scores for their recommendation \n\nprint(f1_score((y_test), clf.predict(X_test), average = 'micro'))\nresults = pd.concat([results, pd.DataFrame([[clas_type, 'Region-set2vec', PCA_flg, f1]], columns= ['dataset', 'representation',  'pca', 'f1'])], ignore_index=True)\n\nresults.to_csv('./results/F1_class{}_pca{}.csv'.format(clas_type, PCA_flg), index = False)")


# In[13]:


results.to_csv('./results/F1_class{}_pca{}.csv'.format(clas_type, PCA_flg), index = False)


# In[ ]:




