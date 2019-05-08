import csv
from utils import Indexer, get_indexer, get_train_data_from_csv, DataPoint, read_word_embeddings, get_dev_data_from_csv
from nltk.tokenize import TweetTokenizer
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

embeddings_path = "data/glove.6B.300d-relativized.txt"
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
print("read data")
X = []
y = []
for data in tweets:
    words = tknr.tokenize(data.text)

    X.append(form_input(words))
    y.append(data.label)

train_data = get_train_data_from_csv()
shuffle(train_data)
print('read train data from csv')

dev_data = get_dev_data_from_csv()
shuffle(train_data)
print('read dev data from csv')

print('len of training data:', len(train_data))
print('len of dev data:', len(dev_data))

x_train = [d.text for d in train_data]
x_test = [d.text for d in dev_data]
y_train = [d.label for d in train_data]
y_test = [d.label for d in dev_data]
print("created train/test splits")

logisticRegr = LogisticRegression()

print("training..")
logisticRegr.fit(x_train, y_train)
print("trained model")

score = logisticRegr.score(x_train, y_train)
print("train set score: ", score)
print()
score = logisticRegr.score(x_test, y_test)
print("test set score: ", score)
print()
