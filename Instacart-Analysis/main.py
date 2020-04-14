#!/usr/bin/env python3
# coding: utf-8

from import_data import run_import_data
from preprocessing import preprocess_for_padding, padding, one_hot_post_padding

import pandas as pd

### Import des Data ###

''' Permet de run l'import et de recuperer fichier corect pour le preprocessing.
nombre de user et nom du fichier est a mettre en parametre '''
run_import_data(10, 'data/merge_df.csv')

data = pd.read_csv('data/merge_df.csv')

### Preprocessing ###

''' Permet d'avoir une liste de liste de lites pour le padding.
dataframe a mettre en parametre '''
L = preprocess_for_padding(data)
#print(L)
#print(len(L))

''' Permet de faire le Padding.
sequence, nb_users, nb_orders and nb_categories a mettre en parametre '''
X = padding(L,10,2,10)
#print(X)

''' Permet de faire le Onehot apres le padding.
matrix after padding and max_categories_found a mettre en parametre '''
X_onehot = one_hot_post_padding(X,21)
#print('\n','_________','\n')
#print(X_onehot)
