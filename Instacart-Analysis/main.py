#!/usr/bin/env python3
# coding: utf-8

from import_data import run_import_data
from preprocessing import preprocess_for_padding

import pandas as pd

run_import_data(10, 'data/merge_df.csv')
# Permet de run l'import et de recuperer fichier corect pour le preprocessing
# nombre de user et nom du fichier est a mettre en parametre

data = pd.read_csv('data/merge_df.csv')

L = preprocess_for_padding(data)
#print(L)
#print(len(L))

#import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# permet d'avoir une liste de liste de lites pour le padding
#run_preprocess1

seq = [[1,2,3],[4,5,6,7]]
#print(len(seq))
s = pad_sequences(seq)
#print(s)

seq2 = [[[1,2,3],[4,5,6,7]],[[3,9],[8,7,0,6,5]],[[2,2,3,0,4,5],[4,5]]]

U = 3 #nb of user
T = 3 #nb of paniers (orders_id)
C = 8 #nb of categories (department_id)

X = np.zeros((3,2,8))
for i in range(len(seq2)):
    xi = pad_sequences(seq2[i], maxlen=C, padding='pre', truncating='pre', value=0)
    t = min(T,xi.shape[0])
    xi = xi[:t]
    X[i] = xi

print(X)
