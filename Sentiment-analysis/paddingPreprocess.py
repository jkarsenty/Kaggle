from keras.preprocessing.sequence import pad_sequences

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
