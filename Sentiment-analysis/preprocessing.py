"""
functions for preprocessing.
functions in order to build our vocabulary before the model.
- lower_txt
- build_vocabulary
-
"""


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
