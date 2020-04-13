'''
Step 2
all functions for the preprocessing of our data, in order to have what we need
to run the LSTM model in the next step.
- preprocess_for_padding(dataframe)
'''

import pandas as pd

# Permet de run l'import et de recuperer fichier corect pour le preprocessing
# nom du fichier est a mettre en parametre
#run_import_data('data/merge_df.csv')

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
