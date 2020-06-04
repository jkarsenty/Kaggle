"""
Main file for Selected text Prediction on Tweeter Sentiment Analysis.
"""

from import_data import *
from EDA import exploratory_data_analysis
from preprocessing import lower_txt,tokenize_matrix,remove_stopwords,target_vector,recup_start_and_end,recup_start_and_end2
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
nb_tweet = -1 # nombre de tweet que l'on prend en compte
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

## Keep only 2 sentiments ##
'''because for neutral sentiment we will keep all the text as selected_text'''
data1 = data[data.sentiment.isin(['negative','positive'])]
#print(data.head())
#print(df1.count())

## Remove textID column ##
df1 = data1.drop(['textID'],axis=1)
df1 = df1.reset_index()

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
#print(df1.selected_tweet)

# 2) recuperation dans une liste de start et end de chaque selected text
list_selected_tweet_ind = recup_start_and_end(df1)
list_start_ind,list_end_ind = recup_start_and_end2(df1)

## Notre nouvelle colonne Y
df1['selected_index'] = list_selected_tweet_ind
df1['startind'] = list_start_ind
df1['endind'] = list_end_ind
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
#Y = df1[['startind','endind']]
print(Y[:3])
print('Y: shape: ',Y.shape)

## X matrix (text) ##
X = df1.text
print('nombre de tweets: ',len(X))

### Split the Dataset ###

#print(type(X),type(Y))
x_train, y_train, x_test, y_test = split_dataset(X,Y,train_ratio=0.8,custom=False)
print('xtrain :', x_train.shape)
print('ytrain :', y_train.shape)
print('xtest :', x_test.shape)
print('ytest :', y_test.shape)
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
#print(x_test_seq[0])
#mot = tk.sequences_to_texts([[2717, 29, 15, 250, 7]])
#print(mot)

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
y_train_max = [max(s) for s in y_train]
print('indice start y_train max:',max(y_train_max))
y_test_max = [max(s) for s in y_test]
print('indice start y_test max:',max(y_test_max))

maxInd = max(np.max(y_train_max),np.max(y_test_max))+1 #+1 cause we want the sequence with all index
print('max_size: ',maxInd)

## Padding ##
x_train_pad = padding(x_train_seq, maxSize)
x_test_pad = padding(x_test_seq,maxSize)
print('x_train_pad: shape: ',x_train_pad.shape)
print('\n', x_train_pad[5])

y_train_pad = padding_for_target(np.array(y_train),maxInd)
y_test_pad = padding_for_target(np.array(y_test),maxInd)
print('y_train_pad: shape: ',y_train_pad.shape)
print('\n', y_train_pad[0])

############################
### One Hot of my Target ###
############################

y_train_oh = one_hot_post_padding(y_train_pad, maxInd)
y_test_oh = one_hot_post_padding(y_test_pad, maxInd)
print(y_train_oh[0])

##############################################
### Splitting datasets: train & validation ###
##############################################

x_train,y_train,x_validate,y_validate = split_dataset(x_train_pad,y_train_oh,train_ratio=0.8,custom=False)
assert x_validate.shape[0] == y_validate.shape[0]
assert x_train.shape[0] == y_train.shape[0]

print('Shape of validation set:',x_validate.shape)
print('Shape of validation set:',y_validate.shape)

################################################
### Fit Y to right shape for 2 output model  ###
################################################

def adapt_fit_target(Y):

    n = len(Y)
    y1 = [] #column of Y start
    y2 = [] #column of Y end
    for i in range(n):
        y1.append(Y[i][0])
        y2.append(Y[i][1])

    newY = [y1,y2]
    return newY
y_train_list = adapt_fit_target(y_train)
y_validate_list = adapt_fit_target(y_validate)
y_test_list = adapt_fit_target(y_test_oh)

#######################
### Glove Embedding ###
#######################

use_glove_embedding_matrix = True
GLOVE_DIM = 50
NB_WORDS = 10000
glove_folder = 'embedding_matrix'
glove_filename = 'glove.twitter.27B.50d.txt'
glove_path = glove_folder+'/'+glove_filename

if use_glove_embedding_matrix == True:
    word_to_idx_glove,embedding_matrix = run_use_glove(GLOVE_DIM,glove_path,NB_WORDS,tokenizer=tk)

#############
### Model ###
#############

### Our Multiclass Models ###

if use_glove_embedding_matrix == False:
    '''With keras Embedding layer'''

    MAX_SEQUENCE_LENGTH = maxSize #tweet le plus long
    print('MAX_SEQUENCE_LENGTH:',MAX_SEQUENCE_LENGTH)
    voc_dim = NB_WORDS #nombre de mots distincts ie:= NB_WORDS = embedding_matrix.shape[0]
    print('voc_dim:',voc_dim)
    EMBEDDING_DIM = 26 #dim de representation
    print('EMBEDDING_DIM:',EMBEDDING_DIM)
    outp = y_train.shape[2] #le nombre de classes nos sentiments
    print('outp:',outp)

    ## my model ##
    model = my_main_model(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,outp)

else:
    '''With Glove Embedding'''
    print('Glove shape:',embedding_matrix.shape)

    MAX_SEQUENCE_LENGTH = maxSize #tweet le plus long
    print('MAX_SEQUENCE_LENGTH:',MAX_SEQUENCE_LENGTH)
    voc_dim = embedding_matrix.shape[0] #nombre de mots distincts ie:= NB_WORDS = embedding_matrix.shape[0]
    print('voc_dim:',voc_dim)
    EMBEDDING_DIM = embedding_matrix.shape[1] #dim de representation
    print('EMBEDDING_DIM:',EMBEDDING_DIM)
    outp = y_train.shape[2] #le nombre de classes nos sentiments
    print('outp:',outp)

    ## my model ##
    model = my_main_glove_model(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,embedding_matrix,outp)

model.summary()

#############################
### Compile and Fit model ###
#############################

## Parameters of the Compile & Fit ##
validation_data = (x_validate,[y_validate_list[0],y_validate_list[1]])
loss_fct = 'categorical_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']
epochs = 100

model_history = train_my_model(x_train,y_train_list,validation_data, model,loss_fct,optimizer,metrics,epochs)
print('Start Ind accuracy:', model_history.history['dense_1_accuracy'][-1])
print('End Ind accuracy:',model_history.history['dense_2_accuracy'][-1])

###############################
### Evaluation of the Model ###
###############################

## To plot our train metrics ##

#eval_metric(model_history, 'dense_1_accuracy')
#eval_metric(model_history, 'dense_2_accuracy')
#eval_metric(model_history, 'dense_1_loss')
#eval_metric(model_history, 'dense_2_loss')

## Test on new set ##
results = model.evaluate(x_test_pad, y_test_list)
#print('Loss: %.3f' %results[0])
#print('Accuracy: %.3f' %results[1])
print('Evaluation:',results)

## Accuracy_score & Confusion Matrix ##
from sklearn.metrics import accuracy_score, confusion_matrix
p_test_list = np.array(model.predict(x_test_pad)) #shape (2, 5382, 33)
y_test_list = np.array(y_test_list) #shape (2, 5382, 33)
#print(np.array(p_test_list)[0][0],y_test_list[0][0])

y_test_start = y_test_list[0].argmax(axis = 1)
y_test_end = y_test_list[1].argmax(axis = 1)
p_test_start = p_test_list[0].argmax(axis = 1)
p_test_end = p_test_list[1].argmax(axis = 1)
print(y_test_start[0],p_test_start[0])

p_acc_start = accuracy_score(y_test_start,p_test_start)
p_acc_end = accuracy_score(y_test_end,p_test_end)
conf_mat_start = confusion_matrix(y_test_start,p_test_start)
conf_mat_end = confusion_matrix(y_test_end,p_test_end)
print('acc start:',p_acc_start)
print('acc end :',p_acc_end)

#####################################
### Post Processing of prediction ###
#####################################

def from_start_end_to_selected_text(X_pad,predict_start,predict_end):

    # From padding to sequence
    X_SEQ = [] # X seq (without the padding)
    for i in range(len(X_pad)):
        text = X_pad[i]
        new_text = []
        #print(text)
        for elt in text:
            #on parcourt les mots
            #print(elt)
            if elt != 0:
                new_text.append(elt)
            #print(new_text)
        #X_pad[i] = tk.sequences_to_texts([new_text])
        X_SEQ.append(new_text)

    # From sequence to Selected sequence
    SEL_TXT = []
    for i in range(len(X_SEQ)):
        text = X_SEQ[i]
        start = predict_start[i]
        end = predict_end[i]
        #print(text)

        # keep only text between start and end
        selected_text = text[start:end+1] #text[ind from start to end included]
        #print(selected_text)
        SEL_TXT.append(selected_text)

    #print(SEL_TXT)
    return SEL_TXT


SEL_TXT_p = from_start_end_to_selected_text(x_test_pad,p_test_start,p_test_end)
#print(SEL_TXT_p[:4])
SEL_TXT_p = tk.sequences_to_texts(np.array(SEL_TXT_p))
#print(SEL_TXT_p[:4])

#print(x_test_pad[2],y_test_start[2],y_test_end[2])
#SEL_TXT_y = from_start_end_to_selected_text(x_test_pad,y_test_start,y_test_end)
#SEL_TXT_y = tk.sequences_to_texts(np.array(SEL_TXT_y))
#print(SEL_TXT_y[:4])

##################
### Submission ###
##################

test_df = importation("data/test.csv")

def preprocess_data_test(df,maxSize):
    '''preprocess the test data to fit the shape of model and make prediction'''

    x = lower_txt(df.text)
    tk = tokenize_matrix(x,tokenizer=3)
    x_seq = tk.texts_to_sequences(x)
    x_pad = padding(x_seq, maxSize)

    return x_pad

x_pad = preprocess_data_test(test_df,maxSize)
#print(x_pad[:2])

predict_list = np.array(model.predict(x_pad))
p_start = predict_list[0].argmax(axis = 1)
p_end = predict_list[1].argmax(axis = 1)
selected_text_predict = from_start_end_to_selected_text(x_pad,p_start,p_end)
selected_text_predict = tk.sequences_to_texts(np.array(selected_text_predict))
print(selected_text_predict[:2])
print(np.array(selected_text_predict).shape)

def submission_df(selected_text_predict,df):
    '''return a column with textID of test_df and selected_text predict by model'''
    #on cree une nouvelle colonne dans notre df avec le selected_text_predict

    df['selected_text_predict'] = selected_text_predict
    #on garde le text par defaut et on met le selected_text_predict
    df['selected_text'] = df[['text']]
    for i in range(len(df)):
        sentiment = df.sentiment[i]
        if sentiment != 'neutral':
            df.selected_text[i]=df.selected_text_predict[i]

    return df

submit = submission_df(selected_text_predict,test_df)
#export_file(submit,'submission2.csv','csv')

submit_kaggle = submit[['textID','selected_text']]
#print(submit_kaggle.head(5))

## export file ##
#export_file(submit_kaggle,path_name='sample_sumbmission.csv',format='csv')
#output_file='sample_sumbmission2.csv'
#submission(output_file, test_df, selected_text_predict)
