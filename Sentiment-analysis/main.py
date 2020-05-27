"""
Main file for Selected text Prediction on Tweeter Sentiment Analysis.
"""

from import_data import importation,export_file
from EDA import exploratory_data_analysis
from preprocessing import lower_txt,tokenize_matrix,remove_stopwords,target_vector,recup_start_and_end
from embedPreprocess import *
from paddingPreprocess import *
from train import *
from evaluation import *

import pandas as pd
import numpy as np
import json

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #permet d'utiliser le GPU du Mac

############################
### importation des data ###
############################

data = importation("data/train.csv")
print(data.columns)
#print(data.describe())
#print(data['text'].head())

#################################
### Exploratory Data Analysis ###
#################################
'''Set run on True if need the EDA with all the print'''
exploratory_data_analysis(dataframe=data,run=False)

#####################
### preprocessing ###
#####################
''' On va realiser une prediction des selected_text '''
nb_tweet = 20 # nombre de tweet que l'on prend en compte
data = data[:nb_tweet]

## to lower ##
data.text = lower_txt(data.text)
data.selected_text = lower_txt(data.selected_text)
#print(data.selected_text[:5])

## Shuffle and Remove stopwords ##
RemoveStpWrd = False
if RemoveStpWrd == True:
    data = data.reindex(np.random.permutation(data.index))
    #remove_stopwords permet aussi de mettre les tweet en liste de mot
    data.text = data.text.apply(remove_stopwords)

## Remove textID column ##
df1 = data.drop(['textID'],axis=1)
#print(df1.text.head())
#print(df1.count())

##########################################
### Selected text index on text column ###
##########################################
''' On a vu lors de l'EDA que la taille du text et du selcted text jour un role.
On va donc identifier la position du selected_text dans le text pour predire
le text en question '''

# 1) mise des tweet en liste de mot
if RemoveStpWrd == False:
    '''si on ne l'a pas fait avant avec le remove_stopwords'''
    df1['tweet'] = df1.text.apply(lambda x: x.split())
    df1['selected_tweet'] = df1.selected_text.apply(lambda x: x.split())
#print(df1.selected_text)

# 2) recuperation dans une liste de start et end de chaque selected text
list_selected_tweet_ind = recup_start_and_end(df1)

## Notre nouvelle colonne Y
df1['selected_index'] = list_selected_tweet_ind
#print(df1.selected_index.head())

#print('Nombre valeur Null:',df1.selected_index.isna().sum())
if df1.selected_index.isna().sum()/len(df1) < 0.1:
    df1.dropna(inplace=True)
else:
    print('Taux de valeur Null:',df1.selected_index.isna().sum()/len(df1))
#print(df1.describe())

########################################
### Splitting datasets: train & test ###
########################################

## Y matrix (selected_index) ##
''' The target classes is a list of 2 numbers - startind and endind
we will need to onehot this vector '''
Y = df1.selected_index
print(Y[:3])
print('Y: shape: ',Y.shape)

## X matrix (text) ##
X = df1.text
print('nombre de tweets: ',len(X))

### Split the Dataset ###

#print(type(X),type(Y))
x_train, y_train, x_test, y_test = split_dataset(X,Y,train_ratio=0.9,custom=False)
print('xtrain :', x_train.shape)
print('xtest :', x_test.shape)
## make sure train_test_split is ok ##
assert x_train.shape[0] == y_train.shape[0]
assert x_test.shape[0] == y_test.shape[0]

####################################################
### Tokenisation and Converting words to numbers ###
####################################################
'''we will use the keras.preprocessing.text.Tokenizer to tokenize
to tokenize our Tweet and transform the tokenized word into numbers
To do so set tokenizer = 3 on the tokenize_matrix()'''

NB_WORDS = None # all words will be used
tk = tokenize_matrix(x_train,tokenizer=3)
x_train_seq = tk.texts_to_sequences(x_train)
x_test_seq = tk.texts_to_sequences(x_test)
#print(x_train_seq[:5])
#print(len(x_train_seq))

#config = tk.get_config()
#config = eval(config['word_counts'])
#export_file(config,"config.json",format='json')

###############
### padding ###
###############

## Stats sur nos tweets ##
train_len = [len(s) for s in x_train]
print('min:',np.min(train_len),'max:',np.max(train_len))
print('quartile:',np.quantile(train_len,q= [0.25,0.5,0.75]))

test_len = [len(s) for s in x_test]
print('min:',np.min(test_len),'max:',np.max(test_len))
print('quartile:',np.quantile(test_len,q= [0.25,0.5,0.75]))

maxSize = max(np.max(train_len),np.max(test_len))
print('max_size: ',maxSize)

## Stats sur notre Target ##
y_train_start_max = [s[0] for s in y_train]
print('indice start y_train max:',max(y_train_start_max))
y_test_start_max = [s[0] for s in y_test]
print('indice start y_test max:',max(y_test_start_max))
y_train_end_max = [s[1] for s in y_train]
print('indice end y_train max:',max(y_train_end_max))
y_test_end_max = [s[1] for s in y_test]
print('indice end y_test max:',max(y_test_end_max))

maxStart = max(np.max(y_train_start_max),np.max(y_test_start_max))
maxEnd = max(np.max(y_train_end_max),np.max(y_test_end_max))
maxStartEnd = max(maxStart,maxEnd)+1 #+1 cause we want the sequence with all index
print('max_size: ',maxStartEnd)


## Padding ##
x_train_pad = padding(x_train_seq, maxSize)
x_test_pad = padding(x_test_seq,maxSize)
print('x_train_pad: shape: ',x_train_pad.shape)
print('\n', x_train_pad[5])

y_train_pad = padding_for_target(np.array(y_train),maxStartEnd)
y_test_pad = padding_for_target(np.array(y_test),maxStartEnd)
print('y_train_pad: shape: ',y_train_pad.shape)
print('\n', y_train_pad[0])

############################
### One Hot of my Target ###
############################

y_train_oh = one_hot_post_padding(y_train_pad, maxStartEnd)
y_test_oh = one_hot_post_padding(y_test_pad, maxStartEnd)

print(y_train_oh[0])

##############################################
### Splitting datasets: train & validation ###
##############################################

x_train,y_train,x_validate,y_validate = split_dataset(x_train_pad,y_train_oh,train_ratio=0.8,custom=False)
assert x_validate.shape[0] == y_validate.shape[0]
assert x_train.shape[0] == y_train.shape[0]

print('Shape of validation set:',x_validate.shape

#######################
### Glove Embedding ###
#######################

use_glove_embedding_matrix = False
GLOVE_DIM = 50
NB_WORDS = 10000
glove_folder = 'embedding_matrix'
glove_filename = 'glove.twitter.27B.50d.txt'
glove_path = glove_folder+'/'+glove_filename

if use_glove_embedding_matrix == True:
    word_to_idx_glove,embedding_matrix = run_use_glove(GLOVE_DIM,glove_path,NB_WORDS,tokenizer=tk)
