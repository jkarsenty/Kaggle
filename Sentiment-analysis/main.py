"""main file"""

from import_data import *
from preprocessing import *
from embedPreprocess import *
from paddingPreprocess import *
from train import *

import pandas as pd
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #permet d'utiliser le GPU du Mac

############################
### importation des data ###
############################

data = importation("data/train.csv")
print(data.columns)
#print(data.describe())
#print(data['text'].head())

#####################
### preprocessing ###
#####################

'''on va realiser une 1ere classification juste sur le sentiment analysis'''
df1 = data.drop('selected_text', axis = 1)
#print(df1.text.head())

nb_tweet = 10000 # nombre de tweet que l'on prend en compte

#notre matrice qui servira de X
tweets = np.array(df1.text)

## to lower ##
raw_text = lower_txt(tweets)
raw_text = raw_text[:nb_tweet]
print('nombre de tweets: ', len(raw_text))

## build vocabulary dictionnary ##
vocabulary = build_vocabulary(raw_text)
#print(len(vocabulary))
#vocabulary = {w:c for w,c in vocabulary.items() if c>70}
print('taille du vocabulary: ', len(vocabulary))
export_json(vocabulary, 'vocabulary.json') #on exporte notre vocabulary

mTokenize = tokenize_matrix(raw_text)
print(mTokenize[:5])
#print(len(mTokenize))

#notre matrice qui servira de Y
Y = target_vector(df1,'sentiment',True)
#Y = np.reshape(Y, (-1,1)) #pour avoir la taille (*,1)
Y = Y[:nb_tweet]
#print(Y[:10])


#################################
## Creation of our word_to_idx ##
#################################

path_to_word_to_idx = 'word_to_idx.json'
word_to_idx = {w:i for i,w in enumerate(vocabulary)}
#print(len(word_to_idx))
export_json(word_to_idx, path_to_word_to_idx)

###############################
### Embedding preprocessing ###
###############################

def give_final_wrdToIdx_embMtx(path_to_word_to_idx, path_to_merge_indexes, path_to_embedding_matrix, load_glove_bool=True, need_index_to_embedding_array=False):

    '''From the vocabulary built give us 2 .json files:
    - the final word_to_idx_merged from the merge between our word_to_idx
    and the the Glove word_to_idx_glove.
    - the Glove embedding matrix
    Input:
        path_to_word_to_idx: string name of .json word_to_idx file
        path_to_merge_indexes: string name of the export file .json
        path_to_embedding_matrix: string name of the export file .npy
        load_glove_bool: Bool to test the preprocess (speed the importation)
        need_index_to_embedding_array: Bool set on True if need the Glove array
    Output:
        export a .json file named path_to_merge_indexes
        export a .npy file named path_to_embedding_matrix
    '''

    ##########################################################################
    ### Loading of Glove, creation of word_to_idx_glove & embedding_matrix ###
    ##########################################################################

    glove_filename = 'embedding_matrix/glove.twitter.27B.50d.txt'
    path_to_glove_index = 'word_to_idx_glove.json'

    if load_glove_bool == False:
        '''if glove's never been loaded'''
        word_to_idx_glove, index_to_embedding_array = export_glove_word_to_index(glove_filename,path_to_glove_index)
        #export the embedding matrix in a .npy file
        np.save(path_to_embedding_matrix,index_to_embedding_array)

    else:
        '''if glove's already loaded once'''
        if need_index_to_embedding_array == True:
            '''if need the embedding matrix in a variable'''
            #word_to_idx_glove, index_to_embedding_array = load_glove_embedding(glove_filename,with_indexes=True)
            index_to_embedding_array = np.load(path_to_embedding_matrix,allow_pickle=True)
        else:
            word_to_idx_glove = importation(path_to_glove_index, format = 'json')

    #####################################################################
    ### Mapping our word_to_idx words to indexes of word_to_idx_glove ###
    #####################################################################

    word_to_idx = importation(path_to_word_to_idx,format = 'json')

    word_to_idx_merged = index_mapping_embedding(word_to_idx,word_to_idx_glove)
    #word_to_idx_embedding c'est le dictionnaire avec les bons indices.
    export_json(word_to_idx_merged, path_to_merge_indexes)

    if need_index_to_embedding_array == True:
        return index_to_embedding_array
    else:
        return

''' Mettre le Bool "run_give_final_wrdToIdx_embMtx" sur False si besoin de:
- recuperer le fichier "path_to_merge_indexes.json" du word_to_idx final.
- recuperer le fichier "embedding_matrix.npy" de la matrice qui servira de weight
a l'Embedding dans notre modele
'''

run_give_final_wrdToIdx_embMtx = True #True if export files already created
load_glove_bool = True # True if glove already loaded
need_index_to_embedding_array = True # True if we need the Glove embedding matrix

path_to_merge_indexes = 'word_to_idx_merged.json'
path_to_embedding_matrix = 'embedding_matrix.npy'

if need_index_to_embedding_array == True:
    '''if need the embedding_matrix'''

    if run_give_final_wrdToIdx_embMtx == False:
        '''if embedding_matrix file never been created'''
        embedding_matrix = give_final_wrdToIdx_embMtx(path_to_word_to_idx,path_to_merge_indexes,path_to_embedding_matrix,load_glove_bool,need_index_to_embedding_array)
    else:
        '''if embedding_matrix file already created'''
        embedding_matrix = np.load(path_to_embedding_matrix,allow_pickle=True)

#word to index of our vocab but with glove indexes
word_to_idx_merged = importation(path_to_merge_indexes, format = 'json')

### Resizing of the Embedding Matrix ###

embedding_matrix_resized= fit_embedding_matrix_to_my_vocab_size(embedding_matrix,word_to_idx_merged)
#print((len(embedding_matrix_resized),len(embedding_matrix_resized[0])))

#############################################################################
### Mapping word of the tokenized matrix to the word_to_idx index integer ###
#############################################################################

'''Set the bool to False if mTokenizeInteger never been created'''
word_to_integer_bool = True
path_to_mTokenizeInteger = 'mTokenizeInteger.npy'

# mTokenize is our matrix of tweets, each tweet tokenised into list of word
if word_to_integer_bool == False:
    '''if mTokenizeInteger never been created'''
    mTokenizeInteger = from_word_to_integer(mTokenize,word_to_idx)
    np.save(path_to_mTokenizeInteger,mTokenizeInteger)
else:
    '''else just need to load it'''
    mTokenizeInteger = np.load(path_to_mTokenizeInteger,allow_pickle=True)

print(mTokenizeInteger[:5])
#print(len(mTokenizeInteger))

###############
### padding ###
###############

### Stats sur nos tweets ###
slen = [len(s) for s in mTokenizeInteger]
print(np.min(slen),np.max(slen))
print(np.quantile(slen,q= [0.25,0.5,0.75]))

#maxSize = max_size(mTokenizeInteger) #donne l'indice max
maxSize = np.max(slen)
print('max_size: ',maxSize)

M = padding(mTokenizeInteger, maxSize)
print('M: shape: ',M.shape,'\n', M[:5])

# No need of one hot with embedding matrix
#X_onehot = one_hot_post_padding(M, maxSize) #pas de onehot lors d'un embedding

##############################################
### Affectation des datasets train et test ###
##############################################

print('Y: shape: ',Y.shape)
x_train, y_train, x_test, y_test = split_dataset(M,Y,train_ratio=0.8)
#print('xtrain :', x_train.shape)
#print('ytrain :', y_train.shape)

#############
### Model ###
#############

### Embedding Layer ###

embedding_matrix = np.matrix(embedding_matrix_resized)
print('Glove shape:',embedding_matrix.shape)
voc_dim = len(word_to_idx) #nombre de mots distincts dans mon word_to_idx
print('voc_dim:',voc_dim)
EMBEDDING_DIM = embedding_matrix.shape[1] #dim de representation
print('EMBEDDING_DIM:',EMBEDDING_DIM)
MAX_SEQUENCE_LENGTH = maxSize #tweet le plus long
print('MAX_SEQUENCE_LENGTH:',MAX_SEQUENCE_LENGTH)

### Test avec juste un Embedding Model ###

#embdLayer = my_embedding_model(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,embedding_matrix)
#embdLayer.summary()
#print(x_test.shape)
#print(embdLayer.predict(x_test))

### Our Model ###

outp = Y.shape[1] #le nombre de classes nos sentiments

'''
En 1er lieu on va faire un RNN mais a modifier ensuite

Faire varier les criteres:
- trainable (dans le fichier train.py )
- epochs
- Rajouter des Dense et Dropout (dans le fichier train.py)
- Mettre un LSTM ?
_ rajouter un batch_size dans le fit ?
'''

model = my_model(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,embedding_matrix,outp)
model.summary()

validation_data = (x_test,y_test)
loss_fct = 'binary_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']
epochs = 10
train_model(x_train,y_train,validation_data, model,loss_fct,optimizer,metrics,epochs)

###############################
### Evaluation of the Model ###
###############################

_,acc = model.evaluate(x_test,y_test)
print('Accuracy: %.2f' %acc)

y_pred = model.predict(x_test)
print(y_pred[:3])
