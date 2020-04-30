''''
Step 4
Functions for padding.
Last preprocessing just before creating of the model.
- max_size(matrix)
- padding(mSequence, sizeSequenceMax)
'''



from keras.preprocessing.sequence import pad_sequences
import numpy as np

def max_size(matrix):
    '''from list of list, give the size of the longest list'''
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
    X = pad_sequences(mSequence, maxlen=sizeSequenceMax, padding='post', truncating='pre', value=0)

    return X

def one_hot_post_padding(matrix, maxSize):
    '''
    Make a one hot matrix X after a padding
    Input:
        matrix after the padding
        maxSize = size of the longest tweet (size of each tweet)
    Output:
        a one hot matrix as an array
    '''

    X_onehot = [] #our new onehot list of list of list in output

    for tweets in matrix:
        '''for each tweets'''
        T = np.zeros(maxSize+1) #onehot of each tweets

        for word in tweets:
            '''for each word in each tweet'''

            t = int(word) #transform word from float to integer

            if t == 0:
                T[t] = 0
            else:
                T[t] = 1

        X_onehot.append(T) #append each tweet in one list

    return np.array(X_onehot)


## Test fonctions ##

#mTokenizeInteger = [[4, 0, 0, 64, 0, 4, 74, 4, 377, 211], [0, 0, 4, 128, 292, 15, 229, 35, 0, 0, 9, 9, 9],
#[29, 0, 32, 0, 21, 355], [86, 0, 9, 758, 21, 0], [0, 39, 0, 0, 0, 0, 4, 164, 0, 0, 0, 109, 509, 228, 46, 13, 0, 80, 550, 0]]
#print(mTokenizeInteger)
#maxSize = max_size(mTokenizeInteger)

#mSequence = mTokenizeInteger
#sizeSequenceMax = maxSize
#print(len(mTokenizeInteger))
#X = padding(mSequence,sizeSequenceMax)
#print(X)
#print(X.shape)

#matrix = X
#X_onehot= one_hot_post_padding(matrix, maxSize)
#print('\n', X_onehot)
