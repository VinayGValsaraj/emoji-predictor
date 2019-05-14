import csv
from utils import Indexer, get_indexer, get_train_data_from_csv, DataPoint, read_word_embeddings, get_dev_data_from_csv, get_test_data_from_csv
from nltk.tokenize import TweetTokenizer
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

embeddings_path = "data/glove.twitter.27B.200d.txt"
word_embeddings = read_word_embeddings(embeddings_path)

def form_input(words):
    text_embeddings = np.zeros(shape=(word_embeddings.get_embedding_length(),))
    for word in words:
        text_embeddings += word_embeddings.get_embedding(word)

    embed = np.divide(text_embeddings, len(words))
    
    return embed


tknr = TweetTokenizer()
indexer = get_indexer('indexer_15_dups.csv')

train_data = get_train_data_from_csv('data/train_15_ds.csv')
shuffle(train_data)

dev_data = get_dev_data_from_csv('data/dev_15_ds.csv')
shuffle(train_data)

test_data = get_test_data_from_csv('data/test_15_ds.csv')
shuffle(train_data)

print('len of training data:', len(train_data))
print('len of dev data:', len(dev_data))

x_train = []
x_dev = []
x_test
y_train = [d.label for d in train_data]
y_dev = [d.label for d in dev_data]
y_test = [d.label for d in test_data]

for d in train_data:
    words = tknr.tokenize(d.text)
    embed = form_input(words)
    x_train.append(embed)

for d in dev_data:
    words = tknr.tokenize(d.text)
    embed = form_input(words)
    x_dev.append(embed)

for d in test_data:
    words = tknr.tokenize(d.text)
    embed = form_input(words)
    x_test.append(embed)


print("created train/dev/test splits")

logisticRegr = LogisticRegression()

print("training..")
logisticRegr.fit(x_train, y_train)
print("trained model")

y_dev_preds = logisticRegr.predict(x_dev)
y_test_preds = logisticRegr.predict(x_test)

score = logisticRegr.score(x_dev, y_dev)
print("dev set score: ", score)
print()
score = logisticRegr.score(x_test, y_test)
print("test set score: ", score)
print()

print("dev test classification report:")
print(classification_report(y_dev, y_dev_preds))

print()
print("test test classification report:")
print(classification_report(y_test, y_test_preds))

