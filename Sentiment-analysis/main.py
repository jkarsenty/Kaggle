"""main file"""


from import_data import *
from preprocessing import *

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

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
print('nombre de tweets: ', raw_text)

## build vocabulary dictionnary
vocabulary = build_vocabulary(raw_text)
print('nombre de mot distincts: ', len(vocabulary))
export_json(vocabulary, 'vocab.json')

word_to_idx = {w:i for i,w in enumerate(vocabulary)}
#print(len(word_to_idx))

export_json(word_to_idx, 'wTi.json')

def tokenize_matrix(matrix):
    ''' from a matrix of tweet give a matrix of list of words (each te)'''
    newMatrix = matrix
    for i in range(len(matrix)):
        #print(matrix[i])
        l = word_tokenize(str(matrix[i]))
        #print(l)
        newMatrix[i] = l

    return newMatrix

mTokenize = tokenize_matrix(raw_text)
#print(mTokenize[:5])
print(len(mTokenize))

def preprocess_for_padding(matrix):
    '''
    Give us the list of list of list for the pad_sequences.
    But 1rst doing a onehot of all the categories (department_id).
    intput:
        dataframe with all the features
    output:
        list L for the padding
    '''

    data = dataframe

    L =[] #liste de liste de liste: liste client en liste d'orders en liste de department
    for client in data.groupby('user_id'):
        #print(client,'\n')
        client_id = client[0]
        client_df = client[1]

        sequences_client = [] #liste de listes department par order pour chaque user
        for order in client_df.groupby('order_id'):
            #print(client_id,': ',order[0],'\n')
            order_id = order[0]
            order_df = order[1]
            #print(order[1],'\n')
            department_ids = list(order_df['department_id'])
            #print(department_ids,'\n')

            sequences_client.append(department_ids)
        #print(sequences_client)
        L.append(sequences_client)

    #print(L)
    #print(len(L))
    return L

def max_size(matrix):
    '''from list of list give me the size of my longest list'''
    maxSize = 0
    for l in matrix:
        sizeList = len(l)
        if maxSize < sizeList:
            maxSize = sizeList
        else:
            maxSize = maxSize
        #print(maxSize,sizeList)
    return maxSize
maxSize = max_size(mTokenize)

from keras.preprocessing.sequence import pad_sequences
def padding(mSequence, sizeSequenceMax):
    '''
    Make the padding with keras.preprocessing.sequence.pad_sequences
    Intput:
        mSequences (list of list)
    Output:
        X matrix of padding sequence
    '''
    X = mSequence

    for i in range(len(mSequence)):
        #print(i)
        xi = pad_sequences(mSequence[i], maxlen=sizeSequenceMax, padding='pre', truncating='pre', value=0)
        #print(xi)
        t = min(nb_orders,xi.shape[0])
        xi = xi[:t]
        #print(xi)
        X[i] = xi

    return X

M = padding(mTokenize,maxSize)
print(M[:10])
