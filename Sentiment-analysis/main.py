"""main file"""

from import_data import *
from preprocessing import *
from embedPreprocess import *
from paddingPreprocess import *

import pandas as pd
import numpy as np

############################
### importation des data ###
############################

data = importation("data/train.csv")
print(data.columns)
#print(data['text'].head())

#####################
### preprocessing ###
#####################

'''on va realiser une 1ere classification juste sur le sentiment analysis'''
df1 = data.drop('selected_text', axis = 1)
#print(df1.text.head())
tweets = np.array(df1.text)

nb_tweet = 10 # nombre de tweet que l'on prend en compte

## to lower ##
raw_text = lower_txt(tweets)
print('nombre de tweets: ', len(raw_text))

## build vocabulary dictionnary ##
vocabulary = build_vocabulary(raw_text)
#print(len(vocabulary))
vocabulary = {w:c for w,c in vocabulary.items() if c>70}
print('taille du vocabulary: ', len(vocabulary))

mTokenize = tokenize_matrix(raw_text[:nb_tweet])
print(mTokenize[:5])
#print(len(mTokenize))

###############################
### Embedding preprocessing ###
###############################

## Creation of the word_to_index ##
def give_final_word_to_index(vocabulary, path_to_merge_indexes, load_glove_bool=True, need_index_to_embedding_array=False):

    '''From the vocabulary built give us a .json file
    with the final word_to_index from the merge between our word_to_index
    and the the Glove word_to_index.
    Input:
        vocabulary: dict given by the build_vocabulary function
        path_to_merge_indexes: string name of the export file .json
        load_glove_bool: Bool to test the preprocess (speed the importation)
        need_index_to_embedding_array: Bool set on True if need the Glove array
    Output:
        export a .json file named path_to_merge_indexes
        '''

    export_json(vocabulary, 'vocabulary.json')

    word_to_idx = {w:i for i,w in enumerate(vocabulary)}
    #print(len(word_to_idx))
    export_json(word_to_idx, 'word_to_idx.json')

    #######################################################################
    ### on fait correspondre nos indices de word_to_idx a ceux de GloVe ###
    #######################################################################

    glove_filename = 'embedding_matrix/glove.twitter.27B.50d.txt'
    path_to_glove_index = 'word_to_idx_glove.json'

    if load_glove_bool == False:
        word_to_index_glove, index_to_embedding_array = export_glove_word_to_index(glove_filename,path_to_glove_index)
    else:
        if need_index_to_embedding_array == True:
            word_to_index_glove, index_to_embedding_array = load_glove_embedding(glove_filename,with_indexes=True)
        else:
            word_to_index_glove = importation(path_to_json, format = 'json')

    word_to_index_merge = index_mapping_embedding(word_to_idx,word_to_index_glove)
    #word_to_index_embedding c'est le dictionnaire avec les bons indices.
    export_json(word_to_index_merge, path_to_merge_indexes)

    if need_index_to_embedding_array == True:
        return index_to_embedding_array
    else:
        return

'''Pour recuperer le fichier "path_to_merge_indexes.json" du word_to_idx final,
mettre le Bool "run_give_final_word_to_index" sur True'''
run_give_final_word_to_index = False
load_glove_bool = True # True if glove already loaded
need_index_to_embedding_array = False # True if we need the Glove embedding matrix

path_to_merge_indexes = 'word_to_idx_merged.json'
if run_give_final_word_to_index == True:
    give_final_word_to_index(vocabulary,path_to_merge_indexes,load_glove_bool,need_index_to_embedding_array)

word_to_index_merge = importation(path_to_merge_indexes, format = 'json')


#####################################################
### Mapping tokenized matrix to the word_to_index ###
#####################################################

# mTokenize is our matrix of tweets, each tweet tokenised into list of word
mTokenizeInteger = from_word_to_integer(mTokenize,word_to_index_merge)
print(mTokenizeInteger[:5])
#print(len(mTokenizeInteger))

###############
### padding ###
###############

maxSize = max_size(mTokenizeInteger)
print(maxSize)

M = padding(mTokenizeInteger, maxSize)
print('M : shape = ',M.shape,'\n', M[:5])

#X_onehot = one_hot_post_padding(M, maxSize)

##############################################
### Affectation des datasets train et test ###
##############################################


#############
### Model ###
#############
