'''
Step 5
On definit ici les fonctions utiles pour l'entrainement de nos data.
Les differents modeles du LSTM.
- split_dataset(X,Y,ratio)
- my_embedding_layer(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,voc_dim,embedding_matrix)
- my_model(outp, inpt)
- train_model(xtrain, ytrain, validation_data, model,loss_fct, optimizer, metrics, epochs=1)
'''

from keras.layers import Input, Dense, LSTM, SimpleRNN, Embedding
from keras.models import Model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

def split_dataset(X,Y,train_ratio):
    ''' Give us the dataset of train and test from the intial dataset
    Input:
        X, Y the initial Dataset as numpy array
        train_ratio: ratio of element to keep for train
    Output :
        x_train, y_train, x_test, y_test
    '''

    t = len(X) #nb_tweet total

    n_train = train_ratio * t #le % de data en train
    if n_train < t-1: #truncature de n_train
        n_train = round(n_train)
    else:
        n_train = int(n_train)

    x_train = X[: n_train]
    x_test = X[n_train :]

    y_train = Y[: n_train]
    y_test = Y[n_train :]

    return x_train, y_train, x_test, y_test

def my_embedding_model(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,embedding_matrix):
    '''Fonction qui prend en entrée:
    Intput:
        MAX_SEQUENCE_LENGTH: taille des séquences en entrée du modèle (nbre de mot max d'1 tweet)
        voc_dim: la taille du vocabulaire (nbre mot distincts)
        EMBEDDING_DIM: la dimension de l'espace de représentation des mots
    '''
    x = Input(shape=(MAX_SEQUENCE_LENGTH,))
    w = Embedding(voc_dim, EMBEDDING_DIM, weights=[embedding_matrix],trainable=False)(x)

    return Model(inputs=x, outputs=w)

def my_embedding_layer(voc_dim,EMBEDDING_DIM,embedding_matrix):
    '''Fonction qui prend en entrée:
    Intput:
        seq_len: taille des séquences en entrée du modèle (nbre de mot max d'1 tweet)
        voc_dim: la taille du vocabulaire (nbre mot distincts)
        EMBEDDING_DIM: la dimension de l'espace de représentation des mots
    Output:
        embdLayer: la couche d'Embedding de mon modele
    '''
    embdLayer = Embedding(voc_dim, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)
    return embdLayer

def my_model(outp, MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,embedding_matrix):
    ''' Fonction qui renvoie mon modele
    Data: en entree shape (*,inpt) et en sortie shape (*, outp) '''

    # en entree du neuronne on aura shape (*,inpt)
    #x = Input(shape = inpt) #en input on a (*,nb_ordrs,nb_categories)
    #h = LSTM(64, activation = 'tanh')(x) #activation par default est tanh
    #shape de sortie (*,64)
    #y = Dense(outp, activation='softmax')(h2) #to have the right output size

    x = Input(shape=(MAX_SEQUENCE_LENGTH,)) #inpt = (MAX_SEQUENCE_LENGTH,)
    embdLayer = my_embedding_layer(voc_dim,EMBEDDING_DIM,embedding_matrix)(x)
    rnn = SimpleRNN(MAX_SEQUENCE_LENGTH)(embdLayer)
    y = Dense(outp, activation="sigmoid")(rnn)

    return Model(inputs=x, outputs=y)

def train_model(xtrain, ytrain, validation_data, model,loss_fct, optimizer, metrics, epochs=1):
    ''' Fonction qui permet d'entrainer notre modele.
    On a les 2 etapes :
    * compilation = amelioration du modele selon lost function et optimizer
    * fit = appliquer ce modele au data
    Input:
        xtrain : data d'entrainement
        ytrain : labels des data d'entrainement
        validation_data: nos data de test qui valide le train
        model : notre modele instancie
        loss (string): la fonction de cout
        optimizer (string) : optimiseur utilise pour amelioration lors du train
        metrics ([liste de string]) : criteres de mesure/metriques
    '''

    M = model

    #compilation
    M.compile(loss = loss_fct, optimizer = optimizer, metrics = metrics)

    # Argument permettant la visualisation
    callback_1 = TensorBoard(log_dir='trainings/train-conv')
    callback_2 = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=5)
    callback_3 = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    Callbacks = [callback_1, callback_2, callback_3]

    #entrainement
    M.fit(xtrain,ytrain, callbacks = Callbacks, epochs= epochs, validation_data = validation_data)

    return
