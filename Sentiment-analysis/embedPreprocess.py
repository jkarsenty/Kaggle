'''
Step 3
Functions for the 2nd preprocessing.
Functions to build our word to index from the Glove embedding matrix.
- load_glove_embedding(glove_filename, with_indexes=True)
- export_glove_word_to_index(glove_filename,path_to_json = 'word_to_idx_glove.json')
- index_mapping_embedding(word_to_idxA, word_to_idxEmbedding)
- from_word_to_integer(matrixTokenize, word_to_index_embedding)
'''

from import_data import export_json

import numpy as np
from collections import defaultdict

def load_glove_embedding(glove_filename, with_indexes=True):
    """
    Read a GloVe txt file.
    If 'with_indexes=True'
        return a tuple of a dictionnaries and an array
        '(word_to_index_dict, index_to_embedding_array)'
    Else
        return only a direct 'word_to_embedding_dict'
        a dictionnary mapping from a string to a numpy array.
    """
    if with_indexes:
        word_to_index_dict = dict()
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
                word_to_index_dict[word] = i
                index_to_embedding_array.append(representation)
            else:
                word_to_embedding_dict[word] = representation

    _WORD_NOT_FOUND = [0.0]* len(representation)  # Empty representation for unknown words.
    if with_indexes:
        _LAST_INDEX = i + 1
        word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
        index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
        print("Embedding loaded from disks.")
        return word_to_index_dict, index_to_embedding_array
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
        print("Embedding loaded from disks.")
        return word_to_embedding_dict

def export_glove_word_to_index(glove_filename,path_to_json = 'word_to_idx_glove.json'):
    '''load the embedding, export the glove word_to_index_dict to a json file
    and return word_to_index_dict and the array index_to_embedding_array '''
    word_to_index_dict, index_to_embedding_array = load_glove_embedding(glove_filename, with_indexes=True)
    export_json(word_to_index_dict, path_to_json)
    return word_to_index_dict, index_to_embedding_array

def index_mapping_embedding(word_to_idxA, word_to_idxEmbedding):
    ''' map the index of the embedding matrix in our word_to_idx'''

    print('Mapping index en cours...')
    my_word_to_index = word_to_idxA
    d1,d2 = my_word_to_index, word_to_idxEmbedding

    for i,w in enumerate(d1):
        #print('w: ',w)
        #print('i: ',i)
        for idx,wrd in enumerate(d2):
            #print('embd: ', wrd,idx)
            if wrd == w:
                 #print('indice: ',idx)
                 i = idx
                 #print('i: ',i)
                 d1[wrd] = i

    print('Mapping index done')
    my_word_to_index = d1
    return my_word_to_index

def from_word_to_integer(matrixTokenize, word_to_index_embedding):
    '''from our list of list of words give us a list of list of integer'''

    print('Transformation Integer en cours ...')
    mT = []
    dictEmb = word_to_index_embedding
    for i in range(len(matrixTokenize)):
        tweet = []
        for word in matrixTokenize[i]:
            KEYS = set(dictEmb.keys())
            #print(KEYS)
            if word in KEYS:
                word = dictEmb[word]
            else:
                word = 0
            #print(word)
            tweet.append(word)
        #print(tweet)
        mT.append(tweet)
    print('Transformation done')
    return mT

### test de fonctions

#glove_filename= 'embedding_matrix/glove.twitter.27B.50d.txt'
#word_to_index_dict, index_to_embedding_array = export_glove_word_to_index(glove_filename)
#print(index_to_embedding[:10])

#d1={'i':0,'love':1,'you':2}
#d2={'ho':0,'he':1,'love':2,'me':3,'you':4,'i':5}
#from import_data import importation
#d2 = importation('gloveWordtoIdx.json', format = 'json')
#my_word_to_index = index_mapping_embedding(d1,d2)
#print(my_word_to_index)

#m = [['i','love','love','you'],['i','you','chocolate']]
#print(from_word_to_integer(m,my_word_to_index))
