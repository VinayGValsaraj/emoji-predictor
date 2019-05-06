import csv
from utils import Indexer, get_indexer, get_data_from_csv, DataPoint
import numpy as np
from random import shuffle


class PerceptronClassifier():
    def __init__(self, label_indexer, feature_extractor):
        self.feature_extractor = feature_extractor
        self.num_classes = len(label_indexer)
        self.weight_vectors = []

    def predict(self, ex):
        feature_vector = self.feature_extractor.extract_features(ex)
        argmax, predicted_class = 0, 0

        for c in range(self.num_classes):
            current_activation = np.dot(feature_vector, self.weight_vectors[c])
            if current_activation >= argmax:

                argmax, predicted_class = current_activation, c

        return predicted_class

    def train(self, train_exs):
        for ex in train_exs:
            for word in ex.text:
                if not self.feature_extractor.indexer.contains(word):
                    self.feature_extractor.indexer.add_and_get_index(word)
        self.weight_vectors = [np.zeros(len(self.feature_extractor.indexer)) for _ in range(self.num_classes)]
        epochs = 5
        for i in range(epochs):
            print (i)
            for ex in train_exs:
                feature_vector = self.feature_extractor.extract_features(ex)
                predicted_class = self.predict(ex)

                if not (predicted_class == ex.label):
                    self.weight_vectors[predicted_class] -= feature_vector
                    self.weight_vectors[ex.label] += feature_vector

            shuffle(train_exs)

class FeatureExtractor():
    def __init__(self):
        self.indexer = Indexer()

    def get_indexer(self):
        return self.indexer

    def extract_features(self, ex):
        feature_vector = np.zeros(len(self.indexer))
        for word in ex.text:
            index = self.indexer.index_of(word)
            feature_vector[index] += 1
        return feature_vector


indexer = get_indexer()
dataset = get_data_from_csv()

train_exs = dataset[:1100000]
test_exs = dataset[1100000:]
# print (len(dataset))
p = PerceptronClassifier(indexer, FeatureExtractor())
p.train(train_exs)

num_correct = 0
num_total = 0
for ex in test_exs:
    num_total += 1
    if p.predict(ex) == ex.label:
        num_correct += 1

print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))

# train_perceptron(train_exs, FeatureExtractor())