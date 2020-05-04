"""
Main file for Binary Classification on Tweeter Sentiment Analysis.
"""

from import_data import *
from EDA import exploratory_data_analysis
from preprocessing import *
from embedPreprocess import *
from paddingPreprocess import *
from train import *
from evaluation import *

import pandas as pd
import numpy as np

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
''' On va realiser une 1ere classification Binaire juste sur le sentiment analysis
positive ou negative'''
nb_tweet = -1 # nombre de tweet que l'on prend en compte
data = data[:nb_tweet]

## to lower ##
data.text = lower_txt(data.text)
data.selected_text = lower_txt(data.selected_text)
#print(data.selected_text[:5])

## Shuffle and Remove stopwords ##
data = data.reindex(np.random.permutation(data.index))
data.text = data.text.apply(remove_stopwords)

## Keep only 2 sentiments ##
df1 = data.drop('selected_text', axis = 1)
df1 = df1[df1.sentiment.isin(['negative','positive'])]
#print(df1.text.head())
#print(df1.count())

########################################
### Splitting datasets: train & test ###
########################################

## Y matrix (sentiment) ##
Y = target_vector(df1,'sentiment',True)
#Y = np.reshape(Y, (-1,1)) #pour avoir la taille (*,1)
print(Y[:3])
print('Y: shape: ',Y.shape)

## X matrix (text) ##
X = df1.text
print('nombre de tweets: ',len(X))

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
#print(mTokenize[:5])
#print(len(mTokenize))

#print(tk.get_config())
#print(tk.sequences_to_texts([mTokenize[0]]))

###############
### padding ###
###############

### Stats sur nos tweets ###
train_len = [len(s) for s in x_train]
print('min:',np.min(train_len),'max:',np.max(train_len))
print('quartile:',np.quantile(train_len,q= [0.25,0.5,0.75]))

test_len = [len(s) for s in x_test]
print('min:',np.min(test_len),'max:',np.max(test_len))
print('quartile:',np.quantile(test_len,q= [0.25,0.5,0.75]))

maxSize = max(np.max(train_len),np.max(test_len))
print('max_size: ',maxSize)

x_train_pad = padding(x_train_seq, maxSize)
x_test_pad = padding(x_test_seq,maxSize)
print('x_train_pad: shape: ',x_train_pad.shape)
print('\n', x_train_pad[5])

### Converting the target classes to numbers ###
'''Already done but can be done also with sklearn.preprocessing.LabelEncoder
if we have multiclass, but we have only 2 categories'''
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#y_train_encode = le.fit_transform(y_train)
#y_test_encode = le.transform(y_test)
#y_train_onehot = to_categorical(y_train_encode)
#y_test_onehot = to_categorical(y_test_encode)

##############################################
### Splitting datasets: train & validation ###
##############################################

x_train,y_train,x_validate,y_validate = split_dataset(x_train_pad,y_train,train_ratio=0.9,custom=False)
assert x_validate.shape[0] == y_validate.shape[0]
assert x_train.shape[0] == y_train.shape[0]

print('Shape of validation set:',x_validate.shape)

#############
### Model ###
#############

### Our Binary Models ###

MAX_SEQUENCE_LENGTH = maxSize
voc_dim = 10000 # = NB_WORDS
EMBEDDING_DIM = 20 #dim de representation
outp = Y.shape[1] #le nombre de classes nos sentiments (binary)
print(outp)

## With Keras Embedding Layer ##
model = my_model_binary0(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,outp)
model.summary()

## Parameters of the Compile & Fit ##
validation_data = (x_validate,y_validate)
loss_fct = 'binary_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']
epochs = 10

model_history = train_model(x_train,y_train,validation_data, model,loss_fct,optimizer,metrics,epochs)
print(model_history.history['accuracy'][-1])

###############################
### Evaluation of the Model ###
###############################

## To plot our train metrics ##
eval_metric(model_history, 'accuracy')
eval_metric(model_history, 'loss')

## Test on new set ##
results = model.evaluate(x_test_pad, y_test)
print('Loss: %.3f' %results[0])
print('Accuracy: %.3f' %results[1])

## Accuracy_score & Confusion Matrix ##
from sklearn.metrics import accuracy_score, confusion_matrix
p_test = model.predict(x_test_pad)
p_test = np.round(p_test,0)
#print(y_test[:5],p_test[:5])
p_acc = accuracy_score(y_test,p_test)
conf_mat = confusion_matrix(y_test,p_test)
print('acc:',p_acc)
print('conf_mat:\n',conf_mat)