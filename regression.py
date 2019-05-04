import csv
from utils import Indexer, get_indexer, get_data_from_csv, DataPoint, read_word_embeddings
from nltk.tokenize import TweetTokenizer
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

embeddings_path = "data/glove.6B.50d-relativized.txt"
word_embeddings = read_word_embeddings(embeddings_path)

def form_input(words):
    text_embeddings = np.zeros(shape=(word_embeddings.get_embedding_length(),))
    for word in words:
        text_embeddings += word_embeddings.get_embedding(word)

    embed = np.divide(text_embeddings, len(words))
    
    return embed


tknr = TweetTokenizer()
indexer = get_indexer()
tweets = shuffle(get_data_from_csv())
X = []
y = []
for data in tweets:
    words = tknr.tokenize(data.text)
    if len(words) == 0:
        continue

    X.append(form_input(words))
    y.append(data.label)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

score = logisticRegr.score(x_train, y_train)
print("train set score: ", score)
print()
score = logisticRegr.score(x_test, y_test)
print("test set score: ", score)
print()
