#!/usr/bin/env python
# coding: utf-8

######## import functions from other files
import nbimporter
import ReadBedFiles
import Region2vec
import json
import argparse
import IPython
from IPython import get_ipython

######### Data path
#path_bed_files = "./bedfiles/*"
path_bed_files = "/project/shefflab/data/ChIP-Atlas/antibodydataset/train/*"
# Universe path
universeFile_path = "./representations/antibody_universe.txt"

######### Parameters 
numberofCores = 4
numberofFiles = 100
#window = 150 #8
#size = 10 #10, 50, 100, 200, 300
min_count = 100 
shuffle_repeat = 100 #100

######### Setting up the argument parser
# https://docs.python.org/3/library/argparse.html
parser = argparse.ArgumentParser(description='run region2vec')
parser.add_argument('-w', '--window', metavar='W', type=int,
                    help='window value')
parser.add_argument('-s','--size', metavar='S', type=int, 
                    help='size value')
 
args = parser.parse_args()
window = args.window
size = args.size


########## The following is commented out because we want don't want to re-do the conversion of term-doc matrix to corpus every time, so we save the variable documents and re-import it 

# Read bed files 
#term_doc_matrix, segmentation_df = ReadBedFiles.readFiles2Vector(path_bed_files, universeFile_path, numberofCores, numberofFiles)
# convert term-doc matrix to Corpus
#documents = ReadBedFiles.convertMat2document(term_doc_matrix, segmentation_df)
#f = open("documents.json","w")
#json1 = json.dumps(documents)
#f.write(json1)
#f.close()

with open('documents.json') as f:
    documents = json.load(f)

    
########## Shuffle documents for training
shuffeled_documents = Region2vec.shuffling(documents, shuffle_repeat)


########## Train word2Vec model
model = Region2vec.trainWord2Vec(shuffeled_documents, window, size, min_count )

model.save("./word2vec_.model")


######### Check to see that it worked
print(len(model.wv.vocab))
print(len(term_doc_matrix))
