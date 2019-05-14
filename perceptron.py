from utils import Indexer, get_indexer, get_train_data_from_csv, get_dev_data_from_csv, get_test_data_from_csv
import numpy as np
from random import shuffle
import sys

from sklearn.metrics import classification_report, accuracy_score

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
        epochs = 1
        for i in range(epochs):
            print ("epoch: i")
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


indexer = get_indexer('data/indexer_cont')
filename = sys.argv[1]

train_set = get_train_data_from_csv('data/train_' + filename)
dev_set = get_dev_data_from_csv('data/dev_' + filename)
test_set = get_test_data_from_csv('data/test_' + filename)

p = PerceptronClassifier(indexer, FeatureExtractor())
p.train(train_set)

y_pred = []
y_true = []
for ex in dev_set:
    y_true.append(ex.label)
    y_pred.append(p.predict(ex))

print ("Dev Set Results: ")
print ("Accuracy: ", accuracy_score(y_true, y_pred))
print (classification_report(y_true, y_pred))

y_pred = []
y_true = []
print ("Test Set Results: ")
for ex in dev_set:
    y_true.append(ex.label)
    y_pred.append(p.predict(ex))
print ("Accuracy: ", accuracy_score(y_true, y_pred))
print (classification_report(y_true, y_pred))

# print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))

