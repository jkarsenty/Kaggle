from keras.preprocessing.sequence import pad_sequences
import numpy as np


def max_size(matrix):
    '''from list of list give me the size of my longest list'''
    maxSize = 0
    for l in matrix:
        sizeList = max(l)
        if maxSize < sizeList:
            maxSize = sizeList
        else:
            maxSize = maxSize
        #print(maxSize,sizeList)
    return maxSize

def padding(mSequence, sizeSequenceMax):
    '''
    Make the padding with keras.preprocessing.sequence.pad_sequences
    Intput:
        mSequences (list of list)
    Output:
        X matrix of padding sequence
    '''
    X = pad_sequences(mSequence, maxlen=sizeSequenceMax, padding='pre', truncating='pre', value=0)

    return X


## Test fonctions ##

#mTokenizeInteger = [[4, 0, 0, 64, 0, 4, 74, 4, 377, 211], [0, 0, 4, 128, 292, 15, 229, 35, 0, 0, 9, 9, 9],
#[29, 0, 32, 0, 21, 355], [86, 0, 9, 758, 21, 0], [0, 39, 0, 0, 0, 0, 4, 164, 0, 0, 0, 109, 509, 228, 46, 13, 0, 80, 550, 0]]
#print(mTokenizeInteger)
#maxSize = max_size(mTokenizeInteger)

#mSequence = mTokenizeInteger
#sizeSequenceMax = maxSize
#print(len(mTokenizeInteger)
#nb_tweet = 5
#X = padding(mSequence, nb_tweet, sizeSequenceMax)
#print(X)
#print(X.shape)
