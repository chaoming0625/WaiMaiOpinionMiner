from collections import defaultdict
import numpy as np
import os
from WaiMaiMiner import common_lib


root_path = os.path.dirname(os.path.abspath(__file__))


class Corpus:
    def __init__(self):
        self._pos_doc_list = []
        self._pos_length = 1000

        with open(root_path + "/f_classifier/positive_corpus_v1.txt", encoding="utf-8") as f:
            i = 0
            while i < self._pos_length:
                self._pos_doc_list.append(common_lib.cut(f.readline().strip()))
                i += 1

        self._neg_doc_list = []
        self._neg_length = 1000
        with open(root_path + "/f_classifier/negative_corpus_v1.txt", encoding="utf-8") as f:
            i = 0
            while i < self._neg_length:
                self._neg_doc_list.append(common_lib.cut(f.readline().strip()))
                i += 1

        runout_content = "You are using the waimai f_classifier version 1.0.\n"
        runout_content += "I contains total %d positive and %d negative corpus." % \
                          (self._pos_length, self._neg_length)
        print(runout_content)

    def get_corpus(self, pos_num=1000, neg_num=1000):
        the_doc_list = self._pos_doc_list[:pos_num] + self._neg_doc_list[:neg_num]
        the_doc_labels = [1] * pos_num + [0] * neg_num
        return the_doc_list, the_doc_labels


class MaxEntClassifier:
    def __init__(self, max_iter=500):
        self._labels_filepath = root_path + "/f_classifier/labels.txt"
        self._feats_filepath = root_path + "/f_classifier/feats.txt"
        self._weights_filepath = root_path + "/f_classifier/weights.txt"

        self._max_iter = max_iter
        self._feats = defaultdict(int)
        self._labels = set()
        self._weights = []

        if not self._path_exists():
            corpus = Corpus()

            train_data, train_labels = corpus.get_corpus()

            from WaiMaiMiner.fe import ChiSquare
            chi = ChiSquare(train_data, train_labels)
            best_words = chi.best_words(3000)

            self.train(train_data, train_labels, best_words)
        else:
            self._init_from_file()

    def _path_exists(self):
        exists = True
        if not os.path.exists(self._labels_filepath):
            exists = False
        elif not os.path.exists(self._weights_filepath):
            exists = False
        elif not os.path.exists(self._feats_filepath):
            exists = False
        return exists

    def _init_from_file(self):
        with open(self._labels_filepath, encoding="utf-8") as f:
            for line in f:
                self._labels.add(eval(line.strip()))
        with open(self._weights_filepath, encoding="utf-8") as f:
            for line in f:
                self._weights.append(float(line.strip()))
        with open(self._feats_filepath, encoding="utf-8") as f:
            for line in f:
                splits = line.strip().split("\t")
                key = (splits[0], splits[1])
                value = eval(splits[2])
                self._feats[key] = value

    def _prob_weight(self, features, label):
        weight = 0.0
        for feature in features:
            if (label, feature) in self._feats:
                weight += self._weights[self._feats[(label, feature)]]
        return np.exp(weight)

    def _calculate_probability(self, features):
        weights = [(self._prob_weight(features, label), label) for label in self._labels]
        z = sum([weight for weight, label in weights])
        prob = [(weight / z, label) for weight, label in weights]
        return prob

    def _convergence(self, last_weight):
        for w1, w2 in zip(last_weight, self._weights):
            if abs(w1 - w2) >= 0.001:
                return False
        return True

    def _get_feats(self, data, label, best_words):
        if best_words is None:
            for word in set(data):
                self._feats[(label, word)] += 1
        else:
            for word in set(data):
                if word in best_words:
                    self._feats[(label, word)] += 1

    def train(self, train_data, train_labels, best_words):
        print("MaxEntClassifier is training ...... ")

        # init the parameters
        self._labels = set(train_labels)
        train_data_length = len(train_labels)
        for i in range(train_data_length):
            self._get_feats(train_data[i], train_labels[i], best_words)

        # the_max param for GIS training algorithm
        the_max = max([len(record) - 1 for record in train_data])
        # init weight for each feature
        self._weights = [0.0] * len(self._feats)
        # init the feature expectation on empirical distribution
        ep_empirical = [0.0] * len(self._feats)
        for i, f in enumerate(self._feats):
            # feature expectation on empirical distribution
            ep_empirical[i] = self._feats[f] / train_data_length
            # each feature function correspond to id
            self._feats[f] = i

        for i in range(self._max_iter):
            # feature expectation on model distribution
            ep_model = [0.0] * len(self._feats)
            for doc in train_data:
                # calculate p(y|x)
                prob = self._calculate_probability(doc)
                for feature in doc:
                    for weight, label in prob:
                        # only focus on features from training data.
                        if (label, feature) in self._feats:
                            # get feature id
                            idx = self._feats[(label, feature)]
                            # sum(1/N * f(y,x)*p(y|x)), p(x) = 1/N
                            ep_model[idx] += weight * (1.0 / train_data_length)

            last_weight = self._weights[:]
            for j, win in enumerate(self._weights):
                delta = 1.0 / the_max * np.log(ep_empirical[j] / ep_model[j])
                # update weight
                self._weights[j] += delta

            # test if the algorithm is convergence
            if self._convergence(last_weight):
                break

        # write the learning result into the file
        with open(self._weights_filepath, "w", encoding="utf-8") as f:
            for weight in self._weights:
                f.write("%f\n" % weight)

        with open(self._feats_filepath, "w", encoding="utf-8") as f:
            for key, value in self._feats.items():
                f.write("%s\t%s\t%d\n" % (key[0], key[1], value))

        with open(self._labels_filepath, "w", encoding="utf-8") as f:
            for label in self._labels:
                f.write("%s\n" % label)

        print("MaxEntClassifier trains over!")

    def classify(self, input_data):
        prob = self._calculate_probability(input_data)
        prob.sort(reverse=True)
        if prob[0][0] > prob[1][0]:
            return prob[0][1]
        else:
            return prob[1][1]


_classifier = MaxEntClassifier(max_iter=300)
classify = _classifier.classify
