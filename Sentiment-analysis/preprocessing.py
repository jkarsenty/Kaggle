'''
Step 2
Functions for 1rst preprocessing.
functions in order to build our vocabulary before the model.
- lower_txt(text_list) or lower_df_txt(dataframe, column_name)
- build_vocabulary(text_list)
- tokenize_matrix(matrix)
- bag_of_word(string, word_to_idx)
'''

from nltk.tokenize import word_tokenize,TweetTokenizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
import re #to split on multiple separators

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
    
    '''sklearn.preprocessing.LabelEncoder can also be used:
    le = LabelEncoder(), then le.fit_transform() and then to_categorical()'''

    Y_initial_values = np.array(dataframe[y_column_name])

    if integer_value == True:
        '''if need to transform the Y_initial_values into numbers'''

        y_unique_values = list(np.unique(Y_initial_values))
        t = len(y_unique_values)

        if t > 2:
            '''if need a onehot vector'''
            #print(t)
            Y_onehot = np.zeros((len(Y_initial_values),t)) #vector Y_onehot taille:(len(Y),len(unique_values))
            #print(Y_onehot.shape)
            for i in range(len(Y_initial_values)):
                index_in_y_unique_values = y_unique_values.index(Y_initial_values[i])
                Y_onehot[i][index_in_y_unique_values] = 1
            Y = Y_onehot
        else:
            '''if binary choice'''
            Y = np.zeros(len(Y_initial_values))
            for i in range(len(Y_initial_values)):
                if Y_initial_values[i] == y_unique_values[0]:
                    Y[i] = 0
                else:
                    Y[i]= 1
            Y = pd.DataFrame(Y)

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
    new_list = []
    for d in text_list:
        #print(d)
        d = str(d).lower()
        new_list.append(d)

    return new_list

def remove_stopwords(input_text):
    '''
    Function to remove English stopwords from a Pandas Series.

    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series
    '''
    stopwords_list = stopwords.words('english')
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split()
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1]
    return " ".join(clean_words)

def build_vocabulary(text_list):
    '''
    Input:
      - list of string (tweets)
    Output:
      - dictionary (words in the corpus)
    '''
    vocabulary = {}
    for d in text_list:
        #print(type(d))

        if type(d) is list:
            '''if tweet is a list, we need to loop into the word of the list'''
            for d2 in d:
                '''for each word in the list'''
                for w in str(d2).split(" "):
                    #print(w)
                    try:
                        vocabulary[w] += 1
                    except:
                        vocabulary[w] = 1

        elif type(d) is str:
            for w in str(d).split(" "):
                #print(w)
                try:
                    vocabulary[w] += 1
                except:
                    vocabulary[w] = 1

    return vocabulary

def Flist_common_word(pos_vocabulary,neg_vocabulary,need_stats_common_word = True,ratio_common = 2):
    '''delete from our vocabularies the common words depending of the ratio_common
    Input:
    - pos_vocabulary: the pos_vocabulary
    - neg_vocabulary: the neg_vocabulary
    - need_stats_common_word: True to choose best ratio then False
    - ratio_common: ratio between value of the common_word in both vocabulary
    if neg_value/pos_value < ratio_common then delete from neg_word
    if pos_value/neg_value < ratio_common then delete from pos_word
    else delete from both
    Output:
    - list_common_word to delete from both vocabulary
    - list_neg_word to delete in pos_vocabulary
    - list_pos_word to delete in neg_vocabulary
    '''
    pos_word = list(pos_vocabulary.keys())
    neg_word = list(neg_vocabulary.keys())

    list_common_word = []
    list_neg_word = []
    list_pos_word = []
    list_ratio = [] #to give us stats on common_word
    if need_stats_common_word == True:
        for neg in neg_word:
            if neg in pos_word:
                common_word = neg
                neg_value = neg_vocabulary[common_word]
                pos_value = pos_vocabulary[common_word]
                neg_ratio = float(neg_value/pos_value)
                pos_ratio = float(pos_value/neg_value)
                list_ratio.append(max(neg_ratio,pos_ratio))

        print('Stats on common words:')
        print(len(list_ratio))
        if len(list_ratio)==0:
            pass
        else:
            print('min:',np.min(list_ratio),'max:',np.max(list_ratio))
            print('quartiles:',np.quantile(list_ratio,[0.25,0.5,0.75]))

    else:
        #ratio_common = 1.8333333
        for neg in neg_word:
            if neg in pos_word:
                common_word = neg
                #print(common_word)
                neg_value = neg_vocabulary[common_word]
                pos_value = pos_vocabulary[common_word]
                neg_ratio = float(neg_value/pos_value)
                pos_ratio = float(pos_value/neg_value)
                if neg_ratio <= ratio_common and pos_ratio <=ratio_common:
                    #print(neg_vocabulary[common_word])
                    #print(pos_vocabulary[common_word])
                    list_common_word.append(common_word)

                elif neg_ratio > ratio_common:
                    list_neg_word.append(common_word)

                elif pos_ratio > ratio_common:
                    list_pos_word.append(common_word)

    return list_common_word,list_pos_word,list_neg_word

def delete_word_in_voc(vocabulary,list_of_delete_word):
    '''from a dict give us a new dict without the word in the list_of_word '''
    new_vocabulary = {w:c for w,c in vocabulary.items() if not w in list_of_delete_word}
    return new_vocabulary

def tokenize_matrix(matrix,tokenizer=0,NB_WORDS=10000):
    ''' From a matrix of tweet give a matrix of list of words (each tweet)
    tokenizer: 0: split, 1: nltk word_tokenize, 2: nltk TweetTokenizer class
    3: keras.preprocessing Tokenizer with use of NB_WORDS most frequent words.
    '''
    print('Tokenisation en cours ...')

    if tokenizer == 3:
        '''tokenize our matrix & retrun the tokenizer'''

        tk = Tokenizer(num_words=NB_WORDS,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True, split=" ")
        tk.fit_on_texts(matrix)
        #newMatrix = tk.texts_to_sequences(matrix)

        #newMatrix = matrix.copy()
        #for i in range(len(matrix)):
        #    tokenize_tweet = np.array(tk.texts_to_sequences(matrix))
            #print(tokenize_tweet)
            #print(newMatrix[i])
        #    newMatrix[i] = []
        #    for mot in tokenize_tweet:
        #        #print(mot)
        #        newMatrix[i].append(mot)

        print('Tokenisation done')
        return tk

    else:
        '''tokenize our matrix'''
        newMatrix = matrix.copy()
        for i in range(len(matrix)):
            #print(matrix[i])
            if tokenizer == 0:
                '''split by " " '''
                #tokenize_tweet = re.split("[,;\ ]",matrix[i])
                tokenize_tweet = matrix[i].split(" ")

            elif tokenizer == 1:
                '''split with nltk word_tokenize'''
                tokenize_tweet = word_tokenize(matrix[i])

            elif tokenizer == 2:
                '''split with mltk TweetTokenizer class'''
                tknzr = TweetTokenizer()
                tokenize_tweet = tknzr.tokenize(str(matrix[i]))

            #print(tokenize_tweet)
            #print(newMatrix[i])
            newMatrix[i] = []
            for mot in tokenize_tweet:
                #print(mot)
                newMatrix[i].append(mot)

        print('Tokenisation done')
        return newMatrix

###########################
### Optionnal Functions ###
###########################

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

#no need if embedding matrix
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


## Tests ##
if (__name__ == "__main__"):
    M = ["I`d have responded, if I were going",
    "Sooo SAD I will miss you here in San Diego!!!",
    "my boss is bullying me..., what interview! leave me alone",
    "Sons of ****, why couldn`t they put them on the releases we already bought",
    "I`d have responded, if I were going"]
    #print(M)
    raw_text = lower_txt(M)
    print(raw_text)
    split_text = tokenize_matrix(raw_text,3)
    print(split_text)
    voc = build_vocabulary(raw_text)
    print(voc)
