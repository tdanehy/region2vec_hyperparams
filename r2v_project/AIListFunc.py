#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import subprocess
import pandas as pd
import numpy as np


# # AIlist Functions

# In[2]:


get_ipython().run_line_magic('time', '')

def run_ailist(f, num_files, database_output, universeFile, PATH):
#     PATH = "/project/shefflab/data/ChIP-Atlas/dataset/train/"
    
    '''
    f:          the query file
    num_files:  the number of files used in the segmentation.
                this will be used to find the segmentation file in regionMap
    database_output: if the ailist results are the database regions (True) or the query regions (False)
    returns pandas dataframe of the database regions that overlapped the query regions
    you need ailist in your path
    '''

    if f.startswith('./'):
        query = f
        
    else:
        query = os.path.join(PATH, f)
    database = universeFile
    
    if database_output:
        #print(subprocess.check_output(["ailist", database, query]))
        ailist_result = subprocess.check_output(['/scratch/tld9kg/AIList/bin/ailist', database, query])
        ailist_result = [x.replace(':', '').split('\t ') for x in ailist_result.decode("utf-8").split('\n')[2:-3]]
    else:
        #print(subprocess.check_output(["ailist", query, database]))
        ailist_result = subprocess.check_output(['/scratch/tld9kg/AIList/bin/ailist', query, database])
        ailist_result = [x.replace(':', '').split('\t ') for x in ailist_result.decode("utf-8").split('\n')[2:-3]]
    
#     print((ailist_result))

    if (len(ailist_result)== 0):
        ailist_result = [['chr0', '0', '0', '0']]

    return pd.DataFrame.from_records(ailist_result)#pd.read_csv('ailist_results.csv', sep=',', header=None)


def ailist_vectorize(f, num_files, segmentation_df, universeFile, PATH):
    '''
    creates a vector from the ailist results
    '''
#     print(list(segmentation_df))
    ailist_df = run_ailist(f, num_files, False, universeFile, PATH)
    ailist_df.columns = ['chrom', 'start', 'end', 'overlaps']
    
    ailist_df.start = ailist_df.start.astype('int64')
    ailist_df.end = ailist_df.end.astype('int64')
    ailist_df.overlaps = ailist_df.overlaps.astype('int64')
    

    ailist_df['chrom'] = ailist_df['chrom'].apply(lambda x: x.strip(':'))

    vector_df = segmentation_df.merge(ailist_df, how='left', on=['chrom', 'start', 'end'])
    vector_df['overlaps'] = vector_df['overlaps'].apply(lambda x: 1 if x >= 1 else 0)

    if(sum(list(vector_df['overlaps'])) == 0):
        return list(np.zeros(len(segmentation_df)))

#     print(len(vector_df))
    return list(vector_df['overlaps'])


# In[ ]:





# In[ ]:





# In[ ]:




