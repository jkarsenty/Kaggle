"""
Step 2
Functions for 1rst preprocessing.
functions in order to build our vocabulary before the model.
- lower_txt(text_list) or lower_df_txt(dataframe, column_name)
- build_vocabulary(text_list)
- tokenize_matrix(matrix)
- bag_of_word(string, word_to_idx)
"""

from nltk.tokenize import word_tokenize
import numpy as np

def target_vector(dataframe,y_column_name,integer_value=False):
    '''
    Give us the Y vector of our target depending on the problem
    Input:
        dataframe
        y_column_name: string of our column of target
        integer_value: Bool if we need integer value (True)
            or initial value (False)
    Output:
        Y: array of target (integer or initial value depending on what needed)
    '''
    Y_initial_values = np.array(dataframe[y_column_name])

    if integer_value == True:
        '''if need a onehot vector'''
        y_unique_values = list(np.unique(Y_initial_values))
        t = len(y_unique_values)
        #print(t)
        Y_onehot = np.zeros((len(Y_initial_values),t)) #vector Y_onehot taille:(len(Y),len(unique_values))
        #print(Y_onehot.shape)

        for i in range(len(Y_initial_values)):
            index_in_y_unique_values = y_unique_values.index(Y_initial_values[i])
            Y_onehot[i][index_in_y_unique_values] = 1

        Y = Y_onehot

    else:
        '''if keep the initial values of Y'''
        Y = Y_initial_values

    #print(Y_initial_values),print(Y)

    return Y


def lower_txt(text_list):
    '''
    Input:
      - list of string (tweets)
    Output:
      - list of string (tweets en minuscules)
    '''
    for d in text_list:
        d = str(d).lower()

    return text_list

def lower_df_txt(dataframe, column_name):
    '''
    Input:
      - dataframe
      - column of the text (tweets)
    Output:
      - df_lower (dataframe avec la colonne tweets en minuscules)
    '''
    df = dataframe
    df[column_name] = df[column_name].str.lower() #transform en lower

    return df

def build_vocabulary(text_list):
    '''
    Input:
      - list of string (tweets)
    Output:
      - dictionary (words in the corpus)
    '''
    vocabulary = {}
    for d in text_list:
        for w in str(d).split(" "):
            try:
                vocabulary[w] += 1
            except:
                vocabulary[w] = 1
    return vocabulary

def tokenize_matrix(matrix):
    ''' from a matrix of tweet give a matrix of list of words (each tweet)'''
    newMatrix = matrix
    for i in range(len(matrix)):
        #print(matrix[i])
        l = word_tokenize(str(matrix[i]))
        #print(l)
        newMatrix[i] = l
    print('Tokenisation done')
    return newMatrix

#not need if embedding matrix
def bag_of_word(string, word_to_idx):
    '''
    Input:
      - string (critique)
      - word_to_idx (it's a dictionary) (index)
    Output:
      - numpy matrix x avec x[1] = 1 si le mot i est présent dans le tweet, 0 sinon.
    '''
    x = np.zeros(len(word_to_idx))
    for w in string.split(" "):
        if w in word_to_idx:
            x[word_to_idx[w]] = 1.0
    return x
