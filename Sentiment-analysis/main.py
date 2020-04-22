"""main file"""


from import_data import importation
from preprocessing import *

import pandas as pd
import numpy as np

data = importation("data/train.csv")
print(data.columns)
#print(data['text'].head())

## on realise une premiere classification juste sur juste le sentiment analysis
''' preprocessing '''

df1 = data.drop('selected_text', axis = 1)
#print(df1.text.head())
tweets = np.array(df1.text)

## to lower
raw_text = lower_txt(tweets)
#print(raw_text[:5])

## build vocabulary dictionnary
vocabulary = build_vocabulary(raw_text)
print(len(vocabulary))

#vocabulary = {w:c for w,c in vocabulary.items() if c>50}
#print(len(vocabulary))
