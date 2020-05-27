''''
Step 4
Functions for padding.
Last preprocessing just before creating of the model.
- max_size(matrix)
- padding(mSequence, sizeSequenceMax)
'''

from import_data import importation

from keras.preprocessing.sequence import pad_sequences
import numpy as np

def from_word_to_integer_cat(matrix,pos_word_to_idx,neg_word_to_idx):

    print('Transformation Integer en cours ...')
    pos_word = list(pos_word_to_idx.keys())
    neg_word = list(neg_word_to_idx.keys())

    new_matrix = [] #matrix with each tweet as list of int
    for tweet in matrix:
        #print(tweet)
        int_tweet = [] #each tweet with list of integer (categorie) instead of word
        for t in tweet:
            #print(t)
            if t in neg_word:
                #print('neg')
                int_t = 0
            elif t in pos_word:
                #print('pos')
                int_t = 2
            else:
                #print('neut')
                int_t = 1

            int_tweet.append(int_t)

        new_matrix.append(int_tweet)
    print('Transformation done')
    return new_matrix

def categorize_tweet_word(mTokenizeInteger,nb_cat):

    new_mTokenizeInteger = []
    for tweet in mTokenizeInteger:
        #print(tweet)
        word = [] #new representation of tweet
        for w in tweet:
            #print(w)
            word_cat = np.zeros(nb_cat) #new representation of w
            #print(word_cat)
            word_cat[w]=1
            word.append(list(word_cat))
        #print(word)
        new_mTokenizeInteger.append(word)
    #print(mTokenizeInteger)

    return new_mTokenizeInteger

def max_size(matrix):
    '''from list of list, give the size of the longest list'''
    maxSize = 0
    for l in matrix:
        #sizeList = len(l) #to give the size max of our sequences
        sizeList = len(l) #to give the index max in all sequences
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
    X = pad_sequences(mSequence, maxlen=sizeSequenceMax, padding='post', truncating='post', value=0)

    return X

def padding_for_target(ySequence, sizeSequenceMax):
    '''
    Make the padding with keras.preprocessing.sequence.pad_sequences
    but we have our ySequence wich have two value ySequence = [[y1],[y2]]
    so we need to make 2 padding for each value
    also we want to keep the values == 0
    so we put the sizeSequenceMax value + 1 where there are no value
    Intput:
        mSequences (list of list)
    Output:
        X matrix of padding sequence
    '''
    Y = []
    Ystart = []
    Yend = []

    for i in range(len(ySequence)):
        Ystart.append([ySequence[i][0]])
        Yend.append([ySequence[i][1]])

    Ystart_seq = pad_sequences(Ystart, maxlen=sizeSequenceMax, padding='post', truncating='post', value=sizeSequenceMax)
    Yend_seq = pad_sequences(Yend, maxlen=sizeSequenceMax, padding='post', truncating='post', value=sizeSequenceMax)

    for i in range(len(ySequence)):
        Y.append([Ystart_seq[i],Yend_seq[i]])

    return np.array(Y)

def one_hot_post_padding(matrix, maxSize):
    '''
    Make a one hot matrix X after a padding
    Input:
        matrix with 2 list after the padding (so each list is padded)
        maxSize = size of the longest tweet (size of each list padded)
    Output:
        a Matrix of 2 onehot lists inside fo each element as an array
    '''

    Y_onehot = [] #our new onehot list of list of list in output

    for tweets in matrix:
        '''for each tweets'''
        W = [] #List of the 2 words (each word as a list of index)
        #print(tweets)

        for word in tweets:
            '''for each word (here a list of index) in each tweet'''
            M = np.zeros(maxSize) #onehot list of the word

            for i in range(maxSize):
                '''for each index in the list'''
                t = int(word[i]) #transform word from float to integer

                if t != maxSize:
                    M[t] = 1

            W.append(M) #append each (of the 2) word in one list (selected_text)

        Y_onehot.append(W) #append each tweet in one list

    return np.array(Y_onehot)


## Test fonctions ##

if (__name__ == "__main__"):

    M = np.array([list(['', 'i`d', 'have', 'responded,', 'if', 'i', 'were', 'going']),
    list(['', 'sooo', 'sad', 'i', 'will', 'miss', 'you', 'here', 'in', 'san', 'diego!!!']),
    list(['my', 'boss', 'is', 'bullying', 'me...']),
    list(['', 'what', 'interview!', 'leave', 'me', 'alone']),
    list(['', 'sons', 'of', '****,', 'why', 'couldn`t', 'they', 'put', 'them', 'on', 'the', 'releases', 'we', 'already', 'bought'])])
    #print(M)
    pos_word_to_idx = importation('Files_json/pos_word_to_idx.json',format='json')
    neg_word_to_idx = importation('Files_json/neg_word_to_idx.json',format='json')
    m2 = from_word_to_integer_cat(M,pos_word_to_idx,neg_word_to_idx)
    #print(m2)

    nb_cat = 3
    new_mTokenizeInteger = categorize_tweet_word(m2,nb_cat)
    #print(new_mTokenizeInteger)

    maxSize = max_size(new_mTokenizeInteger)
    print(maxSize)
    mSequence = new_mTokenizeInteger
    sizeSequenceMax = maxSize
    print(len(mSequence))

    X = padding(mSequence,sizeSequenceMax)
    print(list(X[0]))
    print(X.shape)

    #[[4, 0, 0, 64, 0, 4, 74, 4, 377, 211], [0, 0, 4, 128, 292, 15, 229, 35, 0, 0, 9, 9, 9],
    #[29, 0, 32, 0, 21, 355], [86, 0, 9, 758, 21, 0], [0, 39, 0, 0, 0, 0, 4, 164, 0, 0, 0, 109, 509, 228, 46, 13, 0, 80, 550, 0]]
    #matrix = X
    #X_onehot= one_hot_post_padding(matrix, maxSize)
    #print('\n', X_onehot)
