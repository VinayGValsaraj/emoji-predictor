import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import json
import pandas as pd
from utils import get_train_data_from_csv, get_dev_data_from_csv, get_test_data_from_csv, Indexer, get_indexer
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import classification_report

include_test = True

tknr = TweetTokenizer()
indexer = get_indexer('indexer_15_dups.csv')
word_indexer = Indexer()
word_indexer.add_and_get_index("UNK")

train_data = get_train_data_from_csv('data/train_15_ds.csv')[0:100]
dev_data = get_dev_data_from_csv('data/dev_15_ds.csv')[:20]
test_data = get_test_data_from_csv('data/test_15_ds.csv')[0:20]

X_train = []
Y_train = []
X_dev = []
Y_dev = []
Y_dev_true = []
X_test = []
Y_test = []
Y_test_true = []

for d in train_data:
    words = tknr.tokenize(d.text)
    vector = []
    for word in words:
        idx= word_indexer.add_and_get_index(word)
        vector.append(idx)
    X_train.append(vector)
    y = d.label
    y_onehot = to_categorical(y, len(indexer), dtype='float32')
    Y_train.append(y_onehot)

print("indexed training data")

for d in dev_data:
    words = tknr.tokenize(d.text)
    vector = []
    for word in words:
        idx= word_indexer.add_and_get_index(word)
        vector.append(idx)
    X_dev.append(vector)
    y = d.label
    y_onehot = to_categorical(y, len(indexer), dtype='float32')
    Y_dev.append(y_onehot)
    Y_dev_true.append(d.label)

print("indexed dev data")

for d in test_data:
    words = tknr.tokenize(d.text)
    vector = []
    for word in words:
        idx= word_indexer.add_and_get_index(word)
        vector.append(idx)
    X_test.append(vector)
    y = d.label
    y_onehot = to_categorical(y, len(indexer), dtype='float32')
    Y_test.append(y_onehot)
    Y_test_true.append(d.label)

if include_test:
    print("indexed test data")

ix = len(X_train)
dix = len(X_dev)
X = X_train + X_dev + X_test
X = np.array(X)
X = pad_sequences(X)

X_train = np.array(X[:ix])
X_dev = np.array(X[ix:ix+dix])
X_test = np.array(X[ix+dix:])
Y_train = np.array(Y_train)
Y_dev = np.array(Y_dev)
Y_test = np.array(Y_test)

print("shape of train", X_train.shape)
print("shape of dev", X_dev.shape)
if include_test:
    print("shape of test", X_test.shape)

vocab_size = len(word_indexer)
embed_dim = 300
lstm_out = 300
batch_size= 128
drop = 0.1
num_epochs = 5

##Buidling the LSTM network

model = Sequential()
model.add(Embedding(vocab_size, embed_dim,input_length = X.shape[1]))
model.add(Dropout(drop))
model.add(LSTM(lstm_out, dropout_U=drop, dropout_W=drop))
model.add(Dense(len(indexer),activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

#Here we train the Network.

#model.fit(X_train, Y_train, batch_size =batch_size, verbose=5, epochs=num_epochs)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs)

print("hyperparameters:")
print("embed dim:", embed_dim)
print("lstm_out:", lstm_out)
print("batch_size", batch_size)
print("dropout:", drop)
print("num epochs:", num_epochs)
print()

# Measuring score and accuracy on validation set
print("evaluating dev set..")

score, acc = model.evaluate(X_dev, Y_dev, batch_size = batch_size)

# calculating precision and recall
Y_preds = model.predict_classes(X_dev)

model_name = 'model_15_ds_' + str(int(100 * round(acc, 2))) + '.h5'

model.save(model_name)

print("Logloss score: %.4f" % (score))
print("Validation set Accuracy: %.4f" % (acc))

print(classification_report(Y_dev_true, Y_preds))

print()
print()


if include_test:
    print("evaluating test set..")
    # Measuring score and accuracy on test set

    score, acc = model.evaluate(X_test, Y_test, batch_size = batch_size)

    # calculating precision and recall
    Y_preds = model.predict_classes(X_test)

    print("Logloss score: %.4f" % (score))
    print("Test set Accuracy: %.4f" % (acc))
    print(classification_report(Y_test_true, Y_preds))