#!/usr/bin/env python3
# coding: utf-8

''' Nous allons ici realiser un ANN, un modele lineaire de reconnaissance visuelle
sur le jeu de donnees MNIST propose par kaggle.
Il s'agit d'une base de donnees de chiffres de 0-9 ecrits a la main.

Ici nous sommes sur la page principale, on pourra voir l'arborescence generale
du code. '''

import os 

from requirements import *
from import_data import importation
from train import traitement_data, linear_model, entrainement
from test import evaluation, submission

## IMPORTATION DES DONNEES ET CREATION DES DATAFRAMES ##

df_train = importation("data/train.csv")
#print(df_train.head())

x_train = df_train.drop('label', axis =1) #matrice des data de train
#taille: (42000,784)
y_train = df_train['label'] #colonne y comportant les labels
#taille: (42000,)

## PRETRAITEMENT DE NOS DATA ##

y_train = np.reshape(y_train.values, (-1,1)) #pour avoir la taille (42000,1)

print('xtrain :', x_train.shape)
print('ytrain :', y_train.shape)
print('classes de ytrain :', np.unique(y_train))

# on normalise nos data x_train et on encode les y_train
# pour avoir une distribution de proba sur les classes
x_train, y_train = traitement_data(x_train, y_train)
print(y_train.shape)

## CREATION DU MODELE ##

outp1 = y_train.shape[1] # outp = 10
model1 = linear_model(outp1) # notre instance du modele
model1.summary()

## ENTRAINEMENT DU MODELE ##
# a partir des fonctions presente dans train.py

entrainement(x_train, y_train, model1,'categorical_crossentropy','adam',['accuracy'])

''' On visualisera l'apprentissage sur Tensorboard.
Pour cela il faut taper la commande "tensorboard --logdir trainings", car
trainings est le dossier dans lequel s'enregistrent les data d'apprentissage.

Puis il faut ouvrir l'url que nous renvoie le terminal.
(pour moi: http://localhost:6006/) '''

## EVALUATION ##
# a partir des fonctions presente dans test.py

p_train, train_acc, train_cmat =  evaluation(x_train, y_train, model1)
#print("Confusion train\n", train_cmat)
#print("Acc train\n", train_acc)

## TEST DE NOTRE MODELE SUR DE NOUVELLES DATA ##

x_test = importation("data/test.csv")
p_test = model1.predict(x_test)
submission('submission_ann.csv', p_test)
