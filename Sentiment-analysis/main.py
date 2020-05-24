"""
Main file for Selected text Prediction on Tweeter Sentiment Analysis.
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
''' On va realiser une prediction des selected_text '''
nb_tweet = 10 # nombre de tweet que l'on prend en compte
data = data[:nb_tweet]

## to lower ##
data.text = lower_txt(data.text)
data.selected_text = lower_txt(data.selected_text)
#print(data.selected_text[:5])

## Shuffle and Remove stopwords ##
ShuffleAndRemove = False
if ShuffleAndRemove == True:
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
''' On va identifier la position du selected_text dans le text pour predire
le text en question '''

# 1) mise des tweet en liste de mot
if ShuffleAndRemove == False:
    '''si on ne l'a pas fait avant avec le remove_stopwords'''
    df1['tweet'] = df1.text.apply(lambda x: x.split())
    df1['selected_tweet'] = df1.selected_text.apply(lambda x: x.split())


list_selected_tweet_ind = [] #column of indexes of the selected_text in text
for i in range(len(df1)):
    print('tweet',i,'de taille',len(df1.selected_tweet[i])-1)

    selected_tweet_ind = [] #list of index of each selected_text in each text
    for WRD_SLCT in df1.selected_tweet[i]:
        for ind,WRD_TWT in enumerate(df1.tweet[i]):
            if WRD_TWT == WRD_SLCT :
                print(ind,WRD_SLCT)
                selected_tweet_ind.append(ind)

    list_selected_tweet_ind.append(selected_tweet_ind)
    print('\n')
