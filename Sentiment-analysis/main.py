"""main file"""

from import_data import *
from EDA import exploratory_data_analysis
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

#cText = data.text
#cSelectedText = data.selected_text
#cSentiment = data.sentiment

#################################
### Exploratory Data Analysis ###
#################################
'''Set run on True if need EDA with all the print'''
exploratory_data_analysis(dataframe=data,run=False)

#####################
### preprocessing ###
#####################

nb_tweet = 10000 # nombre de tweet que l'on prend en compte
data = data[:nb_tweet]

## to lower ##
data.text = lower_txt(data.text)
data.selected_text = lower_txt(data.selected_text)
#print(data.selected_text[:5])

## Tokenisation of text & selected text columns ##
split_text = tokenize_matrix(list(data.text))
split_selected = tokenize_matrix(list(data.selected_text))
#print(split_selected[:5])

## creation of new column in the dataframe ##
data['split_text']=split_text
data['split_selected'] = split_selected
df = data.drop(['text','selected_text'], axis = 1)
#print(df1.text.head())
#print(df1.columns)

### Creation of our Positive & Negative Vocabulary ###
'''We create our vocabularies from the selected text of each tweet'''

pos_text = df.split_selected[df.sentiment == 'positive']
neg_text = df.split_selected[df.sentiment == 'negative']
print('nombre de tweets positifs: ', len(pos_text))
print('nombre de tweets negatifs:',len(neg_text))
#print(neg_text[:5])

## build vocabulary dictionnary ##
pos_vocabulary = build_vocabulary(pos_text)
#print('pos voc_dim:',len(pos_vocabulary))
neg_vocabulary = build_vocabulary(neg_text)
#print('neg voc_dim:',len(neg_vocabulary))
#print(neg_vocabulary)

## We have to deal with words in both vocabulary ##
need_stats_common_word = False
ratio_common = 1.83

list_common_word,list_pos_word,list_neg_word = Flist_common_word(pos_vocabulary,neg_vocabulary,need_stats_common_word,ratio_common)
#print('common_word:',len(list_common_word))
#print('only pos_word:',len(list_pos_word))
#print('only neg_word:',len(list_neg_word))

pos_vocabulary = delete_word_in_voc(pos_vocabulary,list_common_word)
pos_vocabulary = delete_word_in_voc(pos_vocabulary,list_neg_word) #delete neg_word
print('pos voc_dim:',len(pos_vocabulary))
neg_vocabulary = delete_word_in_voc(neg_vocabulary,list_common_word)
neg_vocabulary = delete_word_in_voc(neg_vocabulary,list_pos_word) #delete pos_word
print('pos voc_dim:',len(neg_vocabulary))

## export our vocabulary (optionnal)
export_json(pos_vocabulary, 'Files_json/pos_vocabulary.json')
export_json(neg_vocabulary, 'Files_json/neg_vocabulary.json')

### Nos Matrices X et Y ###

## Y ##
Y = target_vector(df,y_column_name='sentiment',integer_value=True)
#Y = np.reshape(Y, (-1,1)) #pour avoir la taille (*,1)
#print(Y[:10])

## X (mTokenize) ##
raw_text = np.array(df.split_text)
mTokenize = raw_text
#print(mTokenize[:5])
#print(len(mTokenize))

### preprocess approfondi sur le X ###
'''Piste a explorer:
Supprimer les espaces, les virgules, et autre mot inutile de chaque tweet
for each word in tweet if word == "to_define" then drop
'''

#################################
## Creation of our word_to_idx ##
#################################

path_to_pos_word_to_idx = 'Files_json/pos_word_to_idx.json'
path_to_neg_word_to_idx = 'Files_json/neg_word_to_idx.json'
pos_word_to_idx = {w:i for i,w in enumerate(pos_vocabulary)}
neg_word_to_idx = {w:i for i,w in enumerate(neg_vocabulary)}
export_json(pos_word_to_idx, path_to_pos_word_to_idx)
export_json(neg_word_to_idx, path_to_neg_word_to_idx)

###############################
### Embedding preprocessing ###
###############################

def give_final_wrdToIdx_embMtx(path_to_word_to_idx, path_to_merge_indexes, path_to_embedding_matrix, load_glove_bool=True, need_index_to_embedding_array=False):

    '''From a word_to_idx built give us 2 .json files:
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
    path_to_glove_index = 'Files_json/word_to_idx_glove.json'
    word_to_idx_glove = {}

    if load_glove_bool == False:
        '''if glove's never been loaded'''
        word_to_idx_glove, index_to_embedding_array = export_glove_word_to_index(glove_filename,path_to_glove_index)
        ## export the embedding matrix in a .npy file
        np.save(path_to_embedding_matrix,index_to_embedding_array)

    else:
        '''if glove's already loaded once'''
        if need_index_to_embedding_array == True:
            '''if need the embedding matrix in a variable'''
            #word_to_idx_glove, index_to_embedding_array = load_glove_embedding(glove_filename,with_indexes=True)
            index_to_embedding_array = np.load(path_to_embedding_matrix,allow_pickle=True)
        else:
            word_to_idx_glove = importation(path_to_glove_index, format = 'json')

    #######################################################################
    ### Mapping our word_to_idx indexes to indexes of word_to_idx_glove ###
    #######################################################################

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

path_to_pos_merge_indexes = 'Files_json/pos_word_to_idx_merged.json'
path_to_neg_merge_indexes = 'Files_json/neg_word_to_idx_merged.json'
path_to_embedding_matrix = 'Files_array/embedding_matrix.npy'

if need_index_to_embedding_array == True:
    '''if need the embedding_matrix'''

    if run_give_final_wrdToIdx_embMtx == False:
        '''if embedding_matrix file never been created'''
        ##positive
        embedding_matrix = give_final_wrdToIdx_embMtx(path_to_pos_word_to_idx,path_to_pos_merge_indexes,path_to_embedding_matrix,load_glove_bool,need_index_to_embedding_array)
        ##negative
        embedding_matrix = give_final_wrdToIdx_embMtx(path_to_neg_word_to_idx,path_to_neg_merge_indexes,path_to_embedding_matrix,load_glove_bool,need_index_to_embedding_array)

    else:
        '''if embedding_matrix file already created'''
        embedding_matrix = np.load(path_to_embedding_matrix,allow_pickle=True)

else:
    '''if only need to create the word_to_idx and not Glove embedding_matrix'''
    ##positive
    give_final_wrdToIdx_embMtx(path_to_pos_word_to_idx,path_to_pos_merge_indexes,path_to_embedding_matrix,load_glove_bool,need_index_to_embedding_array)
    ##negative
    give_final_wrdToIdx_embMtx(path_to_neg_word_to_idx,path_to_neg_merge_indexes,path_to_embedding_matrix,load_glove_bool,need_index_to_embedding_array)

#############################################################################
### Mapping word of the tokenized matrix to the word_to_idx index integer ###
#############################################################################
'''
We will do another preprocessing on our X matrix named mTokenize.
Options :
- mapping each word to integer if they are negative,neutral or positive: [neg,neut,pos]
- Mapping the indexes of each word to Glove's indexes
'''
## word to index of our vocab with glove indexes
word_to_pos_idx_merged = importation(path_to_pos_merge_indexes, format = 'json')
word_to_neg_idx_merged = importation(path_to_neg_merge_indexes, format = 'json')

### Option 1: mapping du X selon le vocabulaire ###

#print(mTokenize[:5]) #our matrix of tweets (tweets tokenised into list of word)
'''Set the bool to False if mTokenizeInteger never been created'''
word_to_integer_bool = True
path_to_mTokenizeInteger1 = 'Files_array/mTokenizeInteger1.npy'

if word_to_integer_bool == False:
    '''if mTokenizeInteger never been created'''
    mTokenizeInteger1= from_word_to_integer_cat(mTokenize,pos_word_to_idx,neg_word_to_idx)
    np.save(path_to_mTokenizeInteger1,mTokenizeInteger1)
else:
    '''else just need to load it'''
    mTokenizeInteger1 = np.load(path_to_mTokenizeInteger1,allow_pickle=True)
#print(mTokenizeInteger1[:5])
#print(len(mTokenizeInteger1))

### Option 2: mapping to glove indexes ###

'''Set the bool to False if mTokenizeInteger never been created'''
word_to_integer_bool = True
path_to_mTokenizeInteger2 = 'Files_array/mTokenizeInteger2.npy'
word_to_idx_merged = importation('Files_json/word_to_idx_merged.json',format = 'json')

if word_to_integer_bool == False:
    '''if mTokenizeInteger never been created'''
    mTokenizeInteger2 = from_word_to_integer(mTokenize,word_to_idx_merged)
    np.save(path_to_mTokenizeInteger2,mTokenizeInteger2)
else:
    '''else just need to load it'''
    mTokenizeInteger2 = np.load(path_to_mTokenizeInteger2,allow_pickle=True)
#print(mTokenizeInteger2[:5])
#print(len(mTokenizeInteger2))

###############
### padding ###
###############

### Stats sur nos tweets ###
slen = [len(s) for s in mTokenizeInteger1]
print('min:',np.min(slen),'max:',np.max(slen))
print('quartiles:',np.quantile(slen,q= [0.25,0.5,0.75]))

maxSize = max_size(mTokenizeInteger1) #donne la taille max des tweets
maxSize = np.max(slen)
print('max_size: ',maxSize)

### padding of words of each tweet into the 3 sentiments ###
nb_cat = 3 #3 sentiments
new_mTokenizeInteger = categorize_tweet_word(mTokenizeInteger1,nb_cat)
print(new_mTokenizeInteger[:5])

M = padding(new_mTokenizeInteger, maxSize)
print('M: shape: ',M.shape)
#print('\n', M[:5])

# No need of one hot with embedding matrix
#X_onehot = one_hot_post_padding(M, maxSize) #pas de onehot lors d'un embedding

##############################################
### Affectation des datasets train et test ###
##############################################

print('Y: shape: ',Y.shape)

Xtrain, Ytrain, x_test, y_test = split_dataset(M,Y,train_ratio=0.9)
x_train,y_train,x_validate,y_validate = split_dataset(Xtrain,Ytrain,train_ratio=0.8)
print('xtrain :', x_train.shape)
print('ytrain :', y_train.shape)
print('xtest :', x_test.shape)
print('ytest :', y_test.shape)
print('xvalidate :', x_validate.shape)
print('y_validate :', y_validate.shape)

############
### Model ###
#############

### Embedding Layer ###

#embedding_matrix = np.matrix(embedding_matrix_resized)
#print('Glove shape:',embedding_matrix.shape)
#voc_dim = len(word_to_idx) #nombre de mots distincts dans mon word_to_idx
#print('voc_dim:',voc_dim)
#EMBEDDING_DIM = embedding_matrix.shape[1] #dim de representation
#print('EMBEDDING_DIM:',EMBEDDING_DIM)
MAX_SEQUENCE_LENGTH = maxSize #tweet le plus long
print('MAX_SEQUENCE_LENGTH:',MAX_SEQUENCE_LENGTH)

### Test avec juste un Embedding Model ###

#embdLayer = my_embedding_model(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,embedding_matrix)
#embdLayer.summary()
#print(x_test.shape)
#print(embdLayer.predict(x_test))

### Our Model ###
inpt = (M.shape[1],M.shape[2])
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

model = my_lstm_model(inpt,outp)
model.summary()

validation_data = (x_validate,y_validate)
loss_fct = 'categorical_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']
epochs = 10

train_model(x_train,y_train,validation_data, model,loss_fct,optimizer,metrics,epochs)

###############################
### Evaluation of the Model ###
###############################

from sklearn.metrics import accuracy_score, confusion_matrix
#_,acc = model.evaluate(x_test,y_test)
#print('Accuracy: %.2f' %acc)

p_test = model.predict(x_test)
y_test = y_test.argmax(axis = 1)
p_test = p_test.argmax(axis = 1)
print(y_test,p_test)
p_acc = accuracy_score(y_test,p_test)
conf_mat = confusion_matrix(y_test,p_test)

print('acc:\n',p_acc)
print('conf_mat:\n',conf_mat)
