"""main file"""


from import_data import *
from preprocessing import *
from embedPreprocess import *
from paddingPreprocess import *

import pandas as pd
import numpy as np


data = importation("data/train.csv")
print(data.columns)
#print(data['text'].head())

## on realise une premiere classification juste sur juste le sentiment analysis
''' preprocessing '''

df1 = data.drop('selected_text', axis = 1)
#print(df1.text.head())
tweets = np.array(df1.text)

## to lower
raw_text = lower_txt(tweets)
print('nombre de tweets: ', len(raw_text))

## build vocabulary dictionnary
vocabulary = build_vocabulary(raw_text)
#print(len(vocabulary))
vocabulary = {w:c for w,c in vocabulary.items() if c>70}
print('taille du vocabulary: ', len(vocabulary))

export_json(vocabulary, 'vocabulary.json')

word_to_idx = {w:i for i,w in enumerate(vocabulary)}
#print(len(word_to_idx))
export_json(word_to_idx, 'word_to_idx.json')


''' on va faire correspondre nos indices de word_to_idx a ceux de l'embedding'''

glove_filename = 'embedding_matrix/glove.twitter.27B.50d.txt'
path_to_json = 'gloveWordtoIdx.json'


word_to_index_glove, index_to_embedding_array = export_glove_word_to_index(glove_filename,path_to_json)
#word_to_index_glove = importation(path_to_json, format = 'json')

word_to_index_embedding = index_mapping_embedding(word_to_idx,word_to_index_glove)
#word_to_index_embedding c'est le dictionnaire avec les bons indices.

''' on a notre index on passe maintenant a la tokenisation et padding'''

mTokenize = tokenize_matrix(raw_text[:10])
print(mTokenize[:5])
#print(len(mTokenize))

mTokenizeInteger = from_word_to_integer(mTokenize,word_to_index_embedding)
print(mTokenizeInteger[:5])

'''on fait le padding'''
maxSize = max_size(mTokenizeInteger)
print(maxSize)


''' padding a adapter et modifier'''

#M = padding(mTokenizeInteger,maxSize)
#print(M[:5])
