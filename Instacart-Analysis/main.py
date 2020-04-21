#!/usr/bin/env python3
# coding: utf-8

from import_data import run_import_data
from preprocessing import preprocess_for_padding, padding, one_hot_post_padding
from train import split_dataset, lstm_model, train_model
from evaluate import evaluation

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

### Import des Data ###

''' Permet de run l'import et de recuperer fichier corect pour le preprocessing.
nombre de user et nom du fichier est a mettre en parametre '''
nb_users = 10000
nb_categories = 10
#run_import_data(nb_users, nb_categories, 'data/merge_df.csv')

data = pd.read_csv('data/merge_df.csv')

### Preprocessing ###

''' Permet d'avoir une liste de liste de lites pour le padding.
dataframe a mettre en parametre '''
L = preprocess_for_padding(data)
#print(L)
#print(len(L))

''' Permet de faire le Padding.
sequence, nb_users, nb_orders and nb_categories a mettre en parametre '''
nb_orders = 10
M = padding(L,nb_users,nb_orders,nb_categories)
print(M)

''' Permet de faire le Onehot apres le padding.
matrix after padding and max_categories_found a mettre en parametre '''
X_onehot = one_hot_post_padding(M,nb_categories)
#print('\n','_________','\n')
#print(X_onehot)

### Choix Dataset ###
Y = X_onehot[:,-1,:]
X = X_onehot[:,:-1,:]

#print(X)
#print(Y)

x_train, y_train, x_test, y_test = split_dataset(X,Y,0.8)
print('xtrain :', x_train.shape)
print('ytrain :', y_train.shape)

### Train du Model ###
''' Permet de train notre model et '''
inpt =(x_train.shape[1],x_train.shape[2]) #inpt = (2,22)
outp = y_train.shape[1] #outp = 22
model = lstm_model(outp,inpt)
model.summary()

train_model(x_train, y_train, (x_test,y_test), model,'mse','adam',['accuracy'])

''' On visualisera l'apprentissage sur Tensorboard.
Commande "tensorboard --logdir trainings" sur terminal.
"trainings" est le dossier dans lequel s'enregistrent les data d'apprentissage.
Puis ouvrir l'url que nous renvoie le terminal (http://localhost:6006/). '''

## EVALUATION ##
'''Permet d'evaluer notre modele
x_test,y_test,nb_categories,les K top score and model en parametre. '''
K_topscore = 5
p = evaluation(x_test,y_test,nb_categories,K_topscore, model)

#print(p_acc)
#print(conf_mat)
