'''Step1
- exploratory_data_analysis(dataframe,run)
'''

from import_data import *
from preprocessing import *
from preprocessing import tokenize_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

#################################
### Exploratory Data Analysis ###
#################################

def exploratory_data_analysis(dataframe, run = True):
    '''
    run the exploratory data analysis with the print and all stats
    '''

    data = dataframe
    if run == True:

        ''' Missing Value '''
        print(data.info())
        data.dropna(inplace=True) # 1 ligne est vide on peut la supprimer

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
            print('----- Taille Text -----')
            print('Sentiment:',s)
            tweetlen = [len(t) for t in data['split_text'][data['sentiment'] == s] ] #len of tweets selon sentiment
            print('tweet_min:',np.min(tweetlen))
            print('tweet_max:',np.max(tweetlen))
            print('quartiles:',np.quantile(tweetlen,q= [0.25,0.5,0.75]))
            sns.distplot(tweetlen, label = s,norm_hist=True, kde = False)
        plt.title('Distribution Text des tweet ')
        plt.xlabel('nbre de mot')
        plt.ylabel('% de tweet')
        plt.legend()
        plt.show()

        for s in sentiments:
            print('----- Taille Selected_text -----')
            print('Sentiment:',s)
            tweetlen = [len(t) for t in data['split_selected'][data['sentiment'] == s] ] #len of tweets selon sentiment
            print('tweet_min:',np.min(tweetlen))
            print('tweet_max:',np.max(tweetlen))
            print('quartiles:',np.quantile(tweetlen,q= [0.25,0.5,0.75]))
            sns.distplot(tweetlen, label = s,norm_hist=True, kde = False)
        plt.title('Distribution Selected_text des tweet ')
        plt.xlabel('nbre de mot')
        plt.ylabel('% de tweet')
        plt.legend()
        plt.show()


        for s in sentiments:
            '''On compare la taille du text et du selected_text de chaque tweets'''
            print('----- Difference de taille -----')
            print('Sentiment:',s)

            Difflen = []
            for txt,seltxt in zip(data.split_text[data.sentiment==s],data.split_selected[data.sentiment==s]):
                Difflen.append(len(txt) - len(seltxt))

            print('Diff_min:',np.min(Difflen))
            print('Diff_max:',np.max(Difflen))
            print('quartiles:',np.quantile(Difflen,q= [0.25,0.5,0.75]))
            sns.distplot(Difflen, label = s,norm_hist=True, kde= False)
        plt.title('Distribution de la Difference de taille ')
        plt.xlabel('nbre de mot')
        plt.ylabel('% de tweet')
        plt.legend()
        plt.show()

        ### Correlation taille selected_text et text ###

        corr=[]
        for s in sentiments:
            text_pos = data.text[data.sentiment==s].astype(str).map(lambda x : len(x.split()))
            sel_pos = data.selected_text[data.sentiment==s].astype(str).map(lambda x : len(x.split()))
            corr.append(scipy.stats.pearsonr(text_pos,sel_pos)[0])
        plt.bar(sentiments,corr,color='blue',alpha=.7)
        plt.gca().set_title("Correlation nbre mot in text & selected text")
        plt.gca().set_ylabel("correlation")
        plt.show()

        print("\n","CONCLUSION:")
        print("Sur 90% des tweets le text et le selected_text sont les memes",'\n',
        "Le nbre de mot du text semble etre un facteur important pour predire le selected_text")

    return

## Tests ##
if (__name__ == "__main__"):
    data = importation("data/train.csv")
    exploratory_data_analysis(data,True)
