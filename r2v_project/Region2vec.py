#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gensim.models import Word2Vec
import json
import nbimporter
import random


# # Shuffling the documents

# In[2]:


def shuffling(document_universe, shuffle_repeat):
    common_text = [value.split(' ')  for key, value in document_universe.items()]
    training_samples = []
    training_samples.extend(common_text)

    for rn in range(shuffle_repeat):
        [(random.shuffle(l)) for l in common_text]
        training_samples.extend(common_text)
    return training_samples
    


# # Train Word2Vec model

# In[ ]:


def trainWord2Vec(training_samples, window, size, min_count):
    model = Word2Vec(training_samples, size=size, window=window, min_count=min_count)
    return model


# In[ ]:





# In[5]:


# file_oi = glob.glob(path_documents + clas_type)[0] # '/document_{}_universe-txt_tr*'.format(clas_type))[0]
# with open(file_oi) as j:
# #     print(glob.glob(path_documents + '/*')[3])
#     document_universe = json.loads(j.read())


# In[7]:


# model = Word2Vec.load("/project/shefflab/data/ChIP-Atlas/word2vecModel/word2vec_shuffle_document_tissue_universe-txt_train.json_19_size100_win100_mincnt_100.model")

