#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install --user import_ipynb


# In[ ]:


import json
import glob
import pandas as pd 
from multiprocessing import Pool
import nbimporter
from importlib import reload
import nbimporter
import import_ipynb
# import AIListFunc
import os
import json
import datetime


# In[ ]:


import AIListFunc


# # Reading Files-multiprocessing

# In[ ]:


def reader(listofPar):
    filename = listofPar[0]
    nofiles = listofPar[1]
    segmentation_df = listofPar[2]
    universefile = listofPar[3]
    PATH = listofPar[4]
    return AIListFunc.ailist_vectorize(filename, nofiles, segmentation_df, universefile, PATH)


# In[1]:


def readFiles2Vector(path_bed_files, universeFile_path, numberofCores = 4, numOfFiles= 100, PATH = ''):
    term_doc_matrix_all = {}
    segmentation_dfs = {}
    term_doc_matrix = {}

    file_list = []
    df_list = []
    term_doc_matrix = {}


    segmentation_df = pd.read_csv(universeFile_path, sep='\t', names=["chrom", "start", "end"])
    segmentation_df['word'] = segmentation_df['chrom'] + '_' + segmentation_df['start'].astype(str) + '_' + segmentation_df['end'].astype(str)

    print('Reading universe file: Done', datetime.datetime.now())

    pool = Pool(numberofCores)

    file_list = sorted(glob.glob(path_bed_files))

    listOfparameters = [file_list, [numOfFiles] * len(file_list), [segmentation_df]* len(file_list), [universeFile_path] * len(file_list),                        [PATH] * len(file_list)]


    #creates a list of vectors
    df_list = pool.map(reader, list(map(list, zip(*listOfparameters))))# (file_list, numOfFiles, segmentation_df))

    print('Reading bed files: Done', datetime.datetime.now())

    for i in range (0,len(file_list)):

        file2 = file_list[i].split('/')[-1].replace('.05.bed','').lower()
        if(sum(df_list[i])>0):
            term_doc_matrix[file2] = df_list[i]
            
    print('Converting to matrix: Done', datetime.datetime.now())
    
    pool.close()
    return term_doc_matrix, segmentation_df#, file_list


# In[ ]:


def convertMat2document(term_doc_matrix, df_seg):
    documents = {}
    for file in term_doc_matrix.keys():
        index = [index for index, value in enumerate(term_doc_matrix[file]) if value == 1]
        doc = ' '.join(df_seg.iloc[index]['word'])
        documents[file] = doc
    return documents
    


# In[ ]:


def writeJsonFile(term_doc_matrix, filename = "term_doc_matrix_bg.json"):
    json1 = json.dumps(term_doc_matrix)
    f = open(filename, "w")
    f.write(json1)
    f.close()


# In[ ]:


def readJsonFile(filename='term_doc_matrix_bg.json'):
    with open(filename, 'r') as j:
         term_doc_matrix = json.loads(j.read())
    return term_doc_matrix


# In[ ]:




