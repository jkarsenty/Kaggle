#!/usr/bin/env python3
# coding: utf-8

'''
Step 2
all functions for the preprocessing of our data, in order to have what we need
to run the LSTM model in the next step.
- preprocess_for_padding(dataframe)
- padding(sequence, nb_users, nb_orders, nb_categories)
- one_hot_post_padding(matrix, max_categories_found)
'''

import numpy as np
from keras.preprocessing.sequence import pad_sequences


def preprocess_for_padding(dataframe):
    '''
    Give us the list of list of list for the pad_sequences.
    But 1rst doing a onehot of all the categories (department_id).

    intput:
        dataframe with all the features
    output:
        list L for the padding
    '''

    data = dataframe

    L =[] #liste de liste de liste: liste client en liste d'orders en liste de department
    for client in data.groupby('user_id'):
        #print(client,'\n')
        client_id = client[0]
        client_df = client[1]

        sequences_client = [] #liste de listes department par order pour chaque user
        for order in client_df.groupby('order_id'):
            #print(client_id,': ',order[0],'\n')
            order_id = order[0]
            order_df = order[1]
            #print(order[1],'\n')
            department_ids = list(order_df['department_id'])
            #print(department_ids,'\n')

            sequences_client.append(department_ids)
        #print(sequences_client)
        L.append(sequences_client)

    #print(L)
    #print(len(L))
    return L

def padding(sequence, nb_users, nb_orders, nb_categories):
    '''
    Make the padding with keras.preprocessing.sequence.pad_sequences
    Intput:
        sequences
        nb_users
        nb_orders - nombre de orders par user
        nb_categories - donnera la taille de chaque sequence
    Output:
        X matrix of padding sequence
    '''
    X = np.zeros((nb_users, nb_orders, nb_categories))

    for i in range(len(sequence)):
        #print(i)
        xi = pad_sequences(sequence[i], maxlen=nb_categories, padding='pre', truncating='pre', value=0)
        print(xi)
        t = min(nb_orders,xi.shape[0])
        xi = xi[:t]
        #print(xi)
        k = nb_orders - t #permet de bien assigne le padding dans l'ordre
        X[i][k:] = xi

    return X


def one_hot_post_padding(matrix, max_categories_found,):
    '''
    Make a one hot matrix X after a padding
    Input:
        matrix after the padding
        max_categories_found = max of categorie found in every baskets
    Output:
        a one hot matrix as an array
    '''

    X_onehot = [] #our new onehot list of list of list in output

    for user in matrix:
        '''for each user'''
        L = [] #onehot of each user

        for order in user:
            '''for each basket in each user'''
            L1 = np.zeros(max_categories_found + 1) #onehot of each basket/order by user
            #print(order)

            for categorie in order:
                '''for each categorie in each order'''
                #print(categorie)
                c = int(categorie) #transform categorie from float to integer

                if c == 0:
                    L1[c] = 0
                else:
                    L1[c] = 1

            L.append(list(L1)) #append each order to each user in the list

        X_onehot.append(L) #append each user in one list

    return np.array(X_onehot)


#### Test functions ###

#seq2 = [[[1,2,3],[4,5,6,7],[8,7,0,6,5]],[[1,2,3,4,5,6,7,8],[1,2,4]],[[2,2,3,0,4,5],[4,5]]]
#U = 3 #nb of user
#T = 5 #nb of paniers (orders_id)
#C = 8 #nb of categories (department_id)
#X = padding(seq2,U,T,C)
#print(X)

#X = one_hot_post_padding(X,15)
#print('\n','_________','\n')

#Y = X[:,-1,:]
#X = X[:,:-1,:]

#print(X)

#print('\n','_________','\n')
#print(Y)
