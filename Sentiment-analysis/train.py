'''
Step 5
On definit ici les fonctions utiles pour l'entrainement de nos data.
Les differents modeles du LSTM.
- split_dataset(X,Y,ratio)
- my_embedding_layer(MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,voc_dim,embedding_matrix)
- my_model(outp, inpt)
- train_model(xtrain, ytrain, validation_data, model,loss_fct, optimizer, metrics, epochs=1)
'''

from keras.layers import Input, Dense, LSTM, SimpleRNN, Embedding, Flatten, Dropout
from keras.layers import CuDNNLSTM, Bidirectional
from keras.models import Model,Sequential
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

def split_dataset(X,Y,train_ratio,custom=False):
    ''' Give us the dataset of train and test from the intial dataset
    Input:
        X, Y the initial Dataset as numpy array
        train_ratio: ratio of element to keep for train
    Output :
        x_train, y_train, x_test, y_test
    '''
    if custom == True:
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

    else:
        '''we will use sklearn train_test_split'''
        test_size = 1-train_ratio
        random_state = 26 #number (seed) used to random
        x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=test_size,random_state=random_state)

    return x_train, y_train, x_test, y_test

###################################
### Models with keras Embedding ###
###################################

def my_model_binary(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,outp):
    ''' Fonction qui renvoie mon modele
    Data: en entree shape (*,inpt) et en sortie shape (*, outp=2)
    Y has binary value so last activation = sigmoid
    '''
    x = Input(shape=(MAX_SEQUENCE_LENGTH,)) #inpt = (MAX_SEQUENCE_LENGTH,)
    embdLayer = Embedding(voc_dim,EMBEDDING_DIM)(x)
    #rnn = SimpleRNN(int(MAX_SEQUENCE_LENGTH))(embdLayer)
    f = Flatten()(embdLayer)
    y = Dense(outp, activation="sigmoid")(f)
    return Model(inputs=x, outputs=y)

def my_model(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,outp):
    ''' Fonction qui renvoie mon modele
    Data: en entree shape (*,inpt) et en sortie shape (*, outp)
    Y has multiclass value (3) so last activation = softmax
    '''
    x = Input(shape=(MAX_SEQUENCE_LENGTH,)) #inpt = (MAX_SEQUENCE_LENGTH,)
    embdLayer = Embedding(voc_dim,EMBEDDING_DIM)(x)
    #rnn = SimpleRNN(int(MAX_SEQUENCE_LENGTH))(embdLayer)
    f = Flatten()(embdLayer)
    y = Dense(outp, activation="softmax")(f)
    return Model(inputs=x, outputs=y)

###################################
### Models with Glove Embedding ###
###################################

def my_glove_model_binary(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,embedding_matrix,outp):
    ''' Fonction qui renvoie mon modele
    Data: en entree shape (*,inpt) et en sortie shape (*, outp = 2)
    Y has binary value so last activation = sigmoid
    '''
    x = Input(shape=(MAX_SEQUENCE_LENGTH,)) #inpt = (MAX_SEQUENCE_LENGTH,)
    embdLayer = my_embedding_layer(voc_dim,EMBEDDING_DIM,embedding_matrix)(x)
    d = Dropout(0.5)(embdLayer)
    f = Flatten()(d)
    y = Dense(outp, activation="sigmoid")(f)
    return Model(inputs=x, outputs=y)

def my_glove_model(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,embedding_matrix,outp):
    ''' Fonction qui renvoie mon modele
    Data: en entree shape (*,inpt) et en sortie shape (*, outp)
    Y has multiclass value (3) so last activation = softmax
    '''
    x = Input(shape=(MAX_SEQUENCE_LENGTH,)) #inpt = (MAX_SEQUENCE_LENGTH,)
    embdLayer = my_embedding_layer(voc_dim,EMBEDDING_DIM,embedding_matrix)(x)
    d = Dropout(0.5)(embdLayer)
    f = Flatten()(d)
    y = Dense(outp, activation="softmax")(f)
    return Model(inputs=x, outputs=y)

#################################
### Embedding Models or Layer ###
#################################

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
    embdLayer = Embedding(voc_dim, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)
    return embdLayer

#####################
### Others Models ###
#####################

def my_glove_lstm_model_binary1(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,embedding_matrix,outp):
    '''Fonction qui renvoie mon modele
    Data: en entree shape (*,inpt) et en sortie shape (*, outp = 2)
    Y has binary value so last activation = sigmoid
    Modele avec Glove Embeding'''

    x = Input(shape=(MAX_SEQUENCE_LENGTH,)) #inpt = (max_seq,nb_cat) pour X = (*,max_seq,nb_cat)
    embdLayer = Embedding(voc_dim,output_dim=EMBEDDING_DIM,weights=[embedding_matrix], trainable=True)(x)
    h1 = LSTM(units = (EMBEDDING_DIM,), return_sequences = True)(x)
    d1 = Dropout(0.25)(h1)
    #h2 = Dense(32)(d1)
    #d2 = Dropout(0.25)(h2)
    f = Flatten()(d1)
    y = Dense(outp, activation="sigmoid")(f)
    return Model(inputs=x, outputs=y)

def my_rnn_model_binary(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,embedding_matrix,outp):
    ''' Fonction qui renvoie mon modele
    Data: en entree shape (*,inpt) et en sortie shape (*, outp=2)
    Y has binary value so last activation = sigmoid
    '''
    x = Input(shape=(MAX_SEQUENCE_LENGTH,)) #inpt = (MAX_SEQUENCE_LENGTH,)
    embdLayer = Embedding(voc_dim,EMBEDDING_DIM)(x)
    rnn = SimpleRNN(int(MAX_SEQUENCE_LENGTH))(embdLayer)
    y = Dense(outp, activation="sigmoid")(rnn)
    return Model(inputs=x, outputs=y)

def my_lstm_model_bidirectionnal(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,embedding_matrix,outp):
    '''Fonction qui renvoie mon modele
    Data: en entree shape (*,inpt) et en sortie shape (*, outp = 2)
    Y has multiclass value so last activation = softmax
    Modele avec Glove Embeding'''

    x = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embdLayer = my_embedding_layer(voc_dim,EMBEDDING_DIM,embedding_matrix)(x)
    h1 = Bidirectional(LSTM(64, return_sequences = True))(embdLayer)
    d1 = Dropout(0.25)(h1)
    h2 = Bidirectional(LSTM(32, return_sequences = False))(d1)
    d2 = Dropout(0.2)(h2)
    y = Dense(outp, activation="tanh")(d2)

    return Model(inputs=x, outputs=y)

def my_lstm_model1(MAX_SEQUENCE_LENGTH,voc_dim,EMBEDDING_DIM,embedding_matrix,outp):
    ''' Fonction qui renvoie mon modele
    Data: en entree shape (*,inpt) et en sortie shape (*, outp) '''

    x = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embdLayer = my_embedding_layer(voc_dim,EMBEDDING_DIM,embedding_matrix)(x)
    h1 = Bidirectional(LSTM(64, return_sequences = True))(embdLayer)
    d1 = Dropout(0.25)(h1)
    h2 = Bidirectional(LSTM(32, return_sequences = False))(d1)
    d2 = Dropout(0.2)(h2)
    y = Dense(outp, activation="softmax")(d2)

    return Model(inputs=x, outputs=y)

######################################
### Training: Compile & Fit Models ###
######################################

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
    callback_2 = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=10)
    callback_3 = ModelCheckpoint(filepath='weights.hdf5', verbose=0, save_best_only=True)

    #Callbacks = [callback_1, callback_2, callback_3]
    Callbacks = [callback_2, callback_3]
    #entrainement
    history = M.fit(xtrain,ytrain, callbacks = Callbacks, epochs= epochs, validation_data = validation_data)

    return history
