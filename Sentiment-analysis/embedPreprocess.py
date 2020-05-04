'''
Step 3
Functions for the 2nd preprocessing.
Functions to build our word to index from the Glove embedding matrix.
- load_glove_embedding(glove_filename, with_indexes=True)
- export_glove_word_to_index(glove_filename,path_to_json = 'word_to_idx_glove.json')
- index_mapping_embedding(word_to_idxA, word_to_idxEmbedding)
- from_word_to_integer(matrixTokenize, word_to_idx_embedding)
'''

from import_data import export_json

import numpy as np
from collections import defaultdict

def run_use_glove(GLOVE_DIM,glove_path,NB_WORDS,tokenizer):

    #######################################
    ### Creation of Glove Word to index ###
    #######################################
    print('Creating Glove word to index...')
    word_to_idx_glove = {}

    with open(glove_path,'r') as glove:
        for line in glove:
            '''line is each word in Glove linked to his EMBEDDING_DIM representation'''
            values = line.split()
            word = values[0] #the Word
            representation = np.asarray(values[1:], dtype='float32') #the representation
            word_to_idx_glove[word] = representation

    print('Glove word to index created')

    ##########################################
    ### Creation of Glove embedding_matrix ###
    ##########################################
    print('Creating Glove embedding_matrix ...')
    embedding_matrix = np.zeros((NB_WORDS, GLOVE_DIM))
    tk=tokenizer

    for w, i in tk.word_index.items():
        # The word_index contains a token for all words of the training data so we need to limit that
        if i < NB_WORDS:
            representation = word_to_idx_glove.get(w)
            # Check if the word from the training data is in the GloVe word_to_idx
            # Otherwise the vector is kept with only zeros
            if representation is not None:
                embedding_matrix[i] = representation
        else:
            break
    print('Glove embedding_matrix created')

    return word_to_idx_glove,embedding_matrix

def load_glove_embedding(glove_filename, with_indexes=True):
    """
    Read a GloVe txt file.
    If 'with_indexes=True'
        return a tuple of a dictionnaries and an array
        '(word_to_idx_dict, index_to_embedding_array)'
    Else
        return only a direct 'word_to_embedding_dict'
        a dictionnary mapping from a string to a numpy array.
    """
    if with_indexes:
        word_to_idx_dict = dict()
        index_to_embedding_array = []
    else:
        word_to_embedding_dict = dict()

    print("Loading embedding from disks...")

    with open(glove_filename, 'r') as glove_file:
        for (i, line) in enumerate(glove_file):

            split = line.split(' ')

            word = split[0]

            representation = split[1:]
            representation = np.array(
                [float(val) for val in representation]
            )

            if with_indexes:
                word_to_idx_dict[word] = i
                index_to_embedding_array.append(representation)
            else:
                word_to_embedding_dict[word] = representation

    _WORD_NOT_FOUND = [0.0]* len(representation)  # Empty representation for unknown words.
    if with_indexes:
        _LAST_INDEX = i + 1
        word_to_idx_dict = defaultdict(lambda: _LAST_INDEX, word_to_idx_dict)
        index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
        print("Embedding loaded from disks.")
        return word_to_idx_dict, index_to_embedding_array
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
        print("Embedding loaded from disks.")
        return word_to_embedding_dict

def export_glove_word_to_index(glove_filename,path_to_json = 'word_to_idx_glove.json'):
    '''load the embedding, export the glove word_to_idx_dict to a json file
    and return word_to_idx_dict and the array index_to_embedding_array '''

    word_to_idx_dict, index_to_embedding_array = load_glove_embedding(glove_filename, with_indexes=True)
    export_json(word_to_idx_dict, path_to_json)

    return word_to_idx_dict, index_to_embedding_array

def index_mapping_embedding(word_to_idxA, word_to_idxEmbedding):
    ''' map the index of the embedding matrix in our word_to_idx'''

    print('Mapping index en cours...')
    my_word_to_idx = word_to_idxA
    d1,d2 = my_word_to_idx, word_to_idxEmbedding

    for w,i in d1.items():
        #print('w: ',w)
        #print('i: ',i)
        print('mot:',i)
        for wrd,idx in d2.items():
            #print('embd: ', wrd,idx)
            if wrd == w:
                 #print('indice: ',idx)
                 i = idx
                 #print('i: ',i)
                 d1[wrd] = i

    print('Mapping index done')
    my_word_to_idx = d1
    return my_word_to_idx

def from_word_to_integer(matrixTokenize, word_to_idx):
    '''from our list of list of words give us a list of list of integer'''

    print('Transformation Integer en cours ...')
    mT = []
    dictEmb = word_to_idx
    KEYS = set(dictEmb.keys()) #cles de nos
    #print(KEYS)

    for i in range(len(matrixTokenize)):
        tweet = []
        for word in matrixTokenize[i]:
            if word in KEYS:
                word = dictEmb[word] #index du word dans le dict
            else:
                word = 0 #pour le moment les mots absent sont remplace par 0
            #print(word)
            tweet.append(word)
        #print(tweet)
        mT.append(tweet)
    print('Transformation done')
    return mT

def fit_embedding_matrix_to_my_vocab_size(embedding_matrix,word_to_idx_merged):
    '''Fit the embedding matrix to the size of our vocabulary
    Input:
        - index_to_embedding_array: the original matrix of Glove of shape (1M,50)
        - word_to_idx_merged: word to index with glove indexes so the only indexes
        we need
    Output:
        - embedding_matrix_resized: with size of our vocabulary shape (voc_dim,50)
    '''
    voc_dim = len(word_to_idx_merged) #size de mon vocab (nbre de mots distincts)
    EMBEDDING_DIM = embedding_matrix.shape[1] #dimension de representation
    new_shape = (voc_dim,EMBEDDING_DIM)
    #New Embedding Matrix
    embedding_matrix_resized = []

    print('Fit embedding_matrix from',embedding_matrix.shape,'to',new_shape,'...')

    for idx,wrd in enumerate(word_to_idx_merged):
        #print(idx)
        embedding_matrix_resized.append(list(embedding_matrix[idx]))
    print('Fit done')

    return embedding_matrix_resized

### test de fonctions

if (__name__ == "__main__"):

    glove_filename= 'embedding_matrix/glove.twitter.27B.50d.txt'
    #word_to_idx_dict, index_to_embedding_array = export_glove_word_to_index(glove_filename)
    #print(index_to_embedding[:10])

    #d1={'i':0,'love':1,'you':2}
    #d2={'ho':0,'he':1,'love':2,'me':3,'you':4,'i':5}
    #from import_data import importation
    #d2 = importation('gloveWordtoIdx.json', format = 'json')
    #my_word_to_idx = index_mapping_embedding(d1,d2)
    #print(my_word_to_idx)

    #m = [['i','love','love','you'],['i','you','chocolate']]
    #print(from_word_to_integer(m,my_word_to_idx))
