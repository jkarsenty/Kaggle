"""main file"""


from import_data import importation

data = importation("data/train.csv")
print(data.columns)
#print(data['text'].head())

## on realise une premiere classification juste sur juste le sentiment analysis
df1 = data.drop('selected_text', axis = 1)
print(df1.head())
