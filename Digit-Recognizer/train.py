# coding: utf-8

"""
On definit ici les fonctions utiles pour l'entrainement de nos data.

Liste des fonctions :
* traitement_data(X,Y)
* linear_model(out,inpt = 784)
* cnn_model()
* entrainement(xtrain, ytrain, model,loss_fct, optimizer, metrics)

"""

import numpy as np
import keras
from keras.utils import to_categorical, normalize
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.models import Model


def traitement_data(X, Y):
    ''' Fonction qui permet de normaliser les data X et d'encoder les labels Y
    pour qu'ils prennent la forme d'une distribution de probabilite
    sur les classes. '''

    # normalisation
    x_norm = normalize(X)

    #encoding des labels
    n = len(np.unique(Y)) #nombre de classes
    y_encode = to_categorical(Y, num_classes = n)

    return x_norm, y_encode


def linear_model(outp, inpt = 784):
    ''' Fonction qui renvoie un modele lineaire sur une representation vectorielle
    des images avec en entree: shape (*,inpt) et en sortie: shape (*, outp) '''

    x = Input(shape=(inpt,)) # en entree du neuronne on aura shape (*,inpt)
    h1 = Dense(64, activation = 'relu')(x) # activation par default est relu
    #shape de sortie (*,64)
    h2 = Dense(64, activation = 'relu')(h1)
    y = Dense(outp, activation='softmax')(h2)

    return Model(inputs=x, outputs=y)


def cnn_model(outp, inpt = (28,28,1)):
    ''' Fonction qui renvoie un modele cnn sur une representation vectorielle
    des images avec en entree: shape (*,inpt) et en sortie: shape (*, outp) '''

    # en entree du neuronne on aura shape (*,inpt)
    x = Input(shape = inpt) # si RGB alors 3 si Gray-scale alors 1 en dernier
    h1 = Conv2D(filters=32, kernel_size=(5,5), padding= 'same', activation='relu')(x)
    h2 = Conv2D(filters=32, kernel_size=(5,5), padding = 'same', activation='relu')(h1)
    h3 = MaxPool2D(pool_size = (2,2))(h2)
    h = Dropout(0.25)(h3)

    g1 = Conv2D(filters=64, kernel_size=(3,3), padding= 'same', activation='relu')(h)
    g2 = Conv2D(filters=64, kernel_size=(3,3), padding= 'same', activation='relu')(g1)
    g3 = MaxPool2D(pool_size = (2,2), strides = (2,2))(g2)
    g = Dropout(0.25)(g3)

    f1 = Flatten()(g)
    f2 = Dense(256, activation = 'relu')(f1)
    f = Dropout(0.5)(f2)

    y = Dense(outp, activation='softmax')(f)

    return Model(inputs=x, outputs=y)


def entrainement(xtrain, ytrain, model,loss_fct, optimizer, metrics, epochs=5):
    ''' Fonction qui permet d'entrainer notre modele.
    On a les 2 etapes :
    * compilation = amelioration du modele selon lost function et optimizer
    * fit = appliquer ce modele au data

    On definit :
    - xtrain : data d'entrainement
    - ytrain : labels des data d'entrainement

    - model : notre modele instancie
    - loss (string): la fonction de cout
    - optimizer (string) : optimiseur utilise pour amelioration lors du train
    - metrics ([liste de string]) : criteres de mesure/metriques
    '''

    M = model

    #compilation
    M.compile(loss = loss_fct, optimizer = optimizer, metrics = metrics)

    # Argument permettant la visualisation
    callback_1 = keras.callbacks.TensorBoard(log_dir='trainings/train-conv')
    callback_2 = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=5)
    callback_3 = keras.callbacks.ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    Callbacks = [callback_1, callback_2, callback_3]

    #entrainement
    M.fit(xtrain,ytrain, callbacks = Callbacks, epochs= epochs)

    return
