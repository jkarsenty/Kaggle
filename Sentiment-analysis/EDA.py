'''Step1
- exploratory_data_analysis(dataframe,run)
'''

from import_data import *
from preprocessing import tokenize_matrix
import pandas as pd
import numpy as np


#################################
### Exploratory Data Analysis ###
#################################

def exploratory_data_analysis(dataframe, run = True):
    '''
    run the exploratory data analysis with the print and all stats
    '''

    data = dataframe
    if run == True:

        print(data.info())
        data.dropna(inplace=True) # 1 ligne est vide on la supprime

        print(data.head(5))
        print(data.describe())

        ### Distribution de mes data ###

        sentiments = np.unique(data['sentiment']) #target column unique values
        print('Sentiments:',sentiments)
        print('-----------')

        countList = []
        for s in sentiments:
            c = data['sentiment'][data['sentiment'] == s].count()
            countList.append(c)

        distrib = pd.DataFrame(list(zip(sentiments,countList)),columns=['sentiment','count'])
        print(distrib)

        ### Comparaison text & selected_text ###

        #On split pour avoir le nombre de mot
        print('-----------')
        split_text = tokenize_matrix(np.array(data.text))
        split_selected = tokenize_matrix(np.array(data.selected_text))
        data['split_text']=split_text
        data['split_selected'] = split_selected

        for s in sentiments:
            print('-----------')
            print('Sentiment:',s)
            tweetlen = [len(t) for t in data['split_text'][data['sentiment'] == s] ] #len of tweets selon sentiment
            print('tweet_min:',np.min(tweetlen))
            print('tweet_max:',np.max(tweetlen))
            print('quartiles:',np.quantile(tweetlen,q= [0.25,0.5,0.75]))




        for s in sentiments:
            '''On compare la taille du text et du selected_text de chaque tweets'''
            print('-----------')
            print('Sentiment:',s)

            Difflen = []
            for txt,seltxt in zip(data.split_text[data.sentiment==s],data.split_selected[data.sentiment==s]):
                Difflen.append(len(txt) - len(seltxt))

            print('Diff_min:',np.min(Difflen))
            print('Diff_max:',np.max(Difflen))
            print('quartiles:',np.quantile(Difflen,q= [0.25,0.5,0.75]))

        '''
        On voit bien que entre le text et le selected_text on a sur 90% des tweets
        nous avons 0 mot de difference
        '''

    return

## Tests ##
if (__name__ == "__main__"):
    data = importation("data/train.csv")
    exploratory_data_analysis(data,True)
