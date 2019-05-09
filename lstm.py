import numpy as np
import collections
from collections import Counter
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch
import random
from utils import Indexer
import torch.nn.functional as F
import time
from utils import Indexer, get_indexer, get_train_data_from_csv, get_dev_data_from_csv, DataPoint, read_word_embeddings
from nltk.tokenize import TweetTokenizer
from random import shuffle

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")

class RNNClassifier(ConsonantVowelClassifier):
    def __init__(self, rnn, indexer):
        self.rnn = rnn
        self.indexer = indexer

    def predict(self, ex):
        log_probs = self.rnn.forward(ex)
        prediction = torch.argmax(log_probs)
        return prediction.item()

class recurrentNeuralNetwork(nn.Module):
    def __init__(self, num_classes, vocab_size, indexer, emb_dim=32, n_hidden_units=10):
        super(recurrentNeuralNetwork, self).__init__()
        embeddings_path = "data/glove.twitter.27B.200d.txt"
        self.word_embeddings = read_word_embeddings(embeddings_path)
        #self.word_embeddings = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, n_hidden_units)
        self.n_hidden_units = n_hidden_units
        self.linear = nn.Linear(self.n_hidden_units, num_classes)
        self.batch_size = 1
        self.indexer = indexer
        self.soft_max = nn.LogSoftmax(dim=0)
        self.tokenizer = TweetTokenizer()
        

    def forward(self, input):
        
        #print("input", input)
        embeds = self.form_input(input)
        #print("embed shape:", embeds.shape)
        x = embeds.view(1, self.batch_size, -1)
        #print(x.shape)
        #x = embeds.view(len(input), self.batch_size , -1)
        lstm_out, self.hidden = self.lstm(x)
        y  = self.linear(lstm_out[-1].view(self.batch_size, -1))
        log_probs = self.soft_max(y.view(-1))
        return log_probs

    def form_input(self, text):
        text_embeddings = np.zeros(shape=(self.word_embeddings.get_embedding_length(),))
        words = self.tokenizer.tokenize(text)
        for word in words:
            text_embeddings += self.word_embeddings.get_embedding(word)

        if len(words) == 0:
            print("no words")
            encoded = torch.from_numpy(np.divide(text_embeddings, 1.0)).float()
        else:
            encoded = torch.from_numpy(np.divide(text_embeddings, len(words))).float()
        
        return encoded

def get_vocab_size(dataset):

    word_cnts = Counter()
    tknr = TweetTokenizer()
    for data in dataset:
        tokenized = tknr.tokenize(data.text)
        for word in tokenized:
            word_cnts[word] += 1
    
    return len(word_cnts)


    

def train_rnn_classifier(train_data, vocab_size, indexer):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    start = time.time()

    embedding_size = 200
    hidden_units = 100
    num_classes = len(indexer)
    num_epochs = 10
    initial_learning_rate = 0.001

    print("embedding size", embedding_size)
    print("hidden units", hidden_units)
    print("num epochs", num_epochs)
    print("learning rate", initial_learning_rate)
    print()
    
    rnn = recurrentNeuralNetwork(num_classes, vocab_size, indexer, emb_dim=embedding_size, n_hidden_units=hidden_units)
    optimizer = optim.Adam(rnn.parameters(), lr=initial_learning_rate)

    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(train_data))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            y = train_data[idx].label
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
        
            rnn.zero_grad()
            probs = rnn.forward(train_data[idx].text)
            loss = torch.neg(probs).dot(y_onehot)
            total_loss += loss
            loss.backward()

            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))

    
    end = time.time()

    print("Total time:", end - start)
    return RNNClassifier(rnn, indexer)

def print_evaluation(test_data, model):
    """
    Runs the classifier on the given text
    :param text:
    :param lm:
    :return:
    """
    num_correct = 0
    for ex in test_data:
        if model.predict(ex.text) == ex.label:
            num_correct += 1
   
    num_total = len(test_data)
    print("%i correct / %i total = %.3f percent accuracy" % (num_correct, num_total, float(num_correct)/num_total * 100.0))


def main():

    train_data = get_train_data_from_csv('data/train_15_dns.csv')
    #shuffle(train_data)
    train_data = train_data[0:50000]

    dev_data = get_dev_data_from_csv('data/dev_15_dns.csv')
    #shuffle(dev_data)
    dev_data = dev_data[0:10000]

    print('len of training data:', len(train_data))
    print('len of dev data:', len(dev_data))

    vocab_size = get_vocab_size(train_data)
    print('calculated vocab size:', vocab_size)

    indexer = get_indexer('indexer_15_dups.csv')

    model = train_rnn_classifier(train_data, vocab_size, indexer)

    print_evaluation(dev_data, model)


if __name__ == "__main__":
    main()




