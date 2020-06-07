# Twitter Sentiment Analysis  

### main_v0.py : Binary Classification (positive and negative) on Twitter Sentiment Analysis.  
- With keras Embedding: Accuracy of 0.86 (0.98 on train data)
- With Glove Embedding: Accuracy of 0.88 (0.96 on train data)

### main_v1.py : Multiclass Classification on Twitter Sentiment Analysis.   
- With keras Embedding: Accuracy of 0.64 (0.97 on train data)
- With Glove Embedding: Accuracy of 0.69 (0.84 on train data)

### main.py: Main file for the Kaggle competition  Tweeter Sentiment Analysis
model predict the indexes of start word and end word of selected_text in the text.
model with one Input (our text embedded and pad_sequenced) and two outputs (onehot index for start word and index for end word)
- prediction with simple embedding model : Accuracy of 0.53 for start_word and 0.4 for end_word 
==> Need to improve the model because of overfitting (0.88 and 0.82)
==> Future work to test in order to improve :
- dropout 
- test with Glove embedding matrix
- improve the embedding matrix and wich word must be kept or dropped as stop_word in our case
- LSTM model must be more suitable for this.
