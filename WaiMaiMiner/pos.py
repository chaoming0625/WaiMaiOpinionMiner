import os
import re
import math


root_filepath = os.path.dirname(os.path.abspath(__file__))


class POS:
    def __init__(self):
        # the import parameters
        self._tags = {}
        self._init_prob = {}
        self._emit_prob = {}
        self._transition_prob = {}

        # the filepath parameters
        self._infinitesimal = 1e-100
        self._tag_num_filepath = "f_pos/tag_num.txt"
        self._init_filepath = "f_pos/init_prob.txt"
        self._emit_filepath = "f_pos/emit_prob.txt"
        self._transition_filepath = "f_pos/transition_prob.txt"

        # check if there exists the init file
        self._check()

    def _check(self):
        prob_file_exist = True
        if not os.path.exists(self._tag_num_filepath):
            prob_file_exist = False
        if not os.path.exists(self._init_filepath):
            prob_file_exist = False
        if not os.path.exists(self._emit_filepath):
            prob_file_exist = False
        if not os.path.exists(self._transition_filepath):
            prob_file_exist = False
        if not prob_file_exist:
            self.train("f_pos/pos_corpus_RenMinRiBao199801.txt")
        else:
            self._init()

    def _init(self):
        with open(self._tag_num_filepath, encoding="utf-8") as f:
            for line in f:
                splits = line.strip().split("\t")
                word = splits[0]
                num = int(splits[1])
                self._tags[word] = num

        with open(self._init_filepath, encoding="utf-8") as f:
            for line in f:
                splits = line.strip().split("\t")
                tag = splits[0]
                prob = float(splits[1])
                self._init_prob[tag] = prob

        with open(self._emit_filepath, encoding="utf-8") as f:
            for line in f:
                splits = line.strip().split("\t")
                tag = splits[0]
                prob = float(splits[1])
                word = splits[2]
                if tag not in self._emit_prob:
                    self._emit_prob[tag] = {}
                self._emit_prob[tag][word] = prob

        with open(self._transition_filepath, encoding="utf-8") as f:
            for line in f:
                splits = line.strip().split("\t")
                tag1 = splits[0]
                prob = float(splits[1])
                tag2 = splits[2]
                if tag1 not in self._transition_prob:
                    self._transition_prob[tag1] = {}
                self._transition_prob[tag1][tag2] = prob

    def train(self, filepath):
        # declare some variables
        tags_num = {}  # record the number of each tag
        init_num = {}  # record the number of the initial number
        transition_num = {}  # record the number of the transition numbers between tags
        emit_num = {}  # record the number of the the emit numbers between word and tag

        # the pattern for split
        pattern = re.compile("\s+")

        # open the file, read each line one by one
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                # split the line into the several splits
                splits = pattern.split(line.strip())

                # establish two lists to record the word and the tag
                line_words = []
                line_tags = []

                for a_split in splits:
                    # split every previous split into word and tag
                    results = a_split.split("/")
                    line_words.append(results[0])
                    line_tags.append(results[1].lower())

                # get the length of two lists
                length = len(line_words)
                assert length == len(line_tags)

                # count the number of init, emit and transition

                # count the init
                tag = line_tags[0]
                init_num[tag] = init_num.get(tag, 0) + 1

                # count the transition
                for i in range(length - 1):
                    tag1 = line_tags[i]
                    tag2 = line_tags[i + 1]
                    if tag1 not in transition_num:
                        transition_num[tag1] = {}
                    transition_num[tag1][tag2] = transition_num[tag1].get(tag2, 0) + 1

                # count the emit and tag's number
                for i in range(length):
                    tag = line_tags[i]
                    word = line_words[i]
                    if tag not in emit_num:
                        emit_num[tag] = {}
                    emit_num[tag][word] = emit_num[tag].get(word, 0) + 1
                    tags_num[tag] = tags_num.get(tag, 0) + 1

        # count the probability of the self.init_prob, self.emit_prob, self.transition_prob
        # and write them into the file

        # write the tag num
        with open(self._tag_num_filepath, mode="w", encoding="utf-8") as f:
            self._tags = tags_num
            for tag, num in sorted(tags_num.items()):
                f.write("%s\t%d\n" % (tag, num))

        # write the init probability
        with open(self._init_filepath, mode="w", encoding="utf-8") as f:
            total = sum(init_num.values())
            for tag, num in sorted(init_num.items()):
                prob = num / total
                self._init_prob[tag] = prob
                f.write("%s\t%.60f\t%d\n" % (tag, prob, num))

        # write the transition probability
        with open(self._transition_filepath, mode="w", encoding="utf-8") as f:
            # get the tag and the transition tag
            for tag1 in sorted(transition_num.keys()):
                tag_dict = transition_num[tag1]
                total = sum(tag_dict.values())
                self._transition_prob[tag1] = {}
                for tag2 in sorted(tag_dict.keys()):
                    num = tag_dict[tag2]
                    prob = num / total
                    self._transition_prob[tag1][tag2] = prob
                    f.write("%s\t%.60f\t%s\t%d\n" % (tag1, prob, tag2, num))

        # write the emit probability
        with open(self._emit_filepath, mode="w", encoding="utf-8") as f:
            for tag in sorted(emit_num.keys()):
                tag_dict = emit_num[tag]
                total = sum(tag_dict.values())
                self._emit_prob[tag] = {}
                for word in sorted(tag_dict.keys()):
                    num = tag_dict[word]
                    prob = num / total
                    self._emit_prob[tag][word] = prob
                    f.write("%s\t%.60f\t%s\t%d\n" % (tag, prob, word, num))

    def _viterbi(self, observation):
        # record the first path and first probability
        prob_a = {}
        path_a = {}

        # initialize
        for tag in self._tags.keys():
            path_a[tag] = [tag]
            prob_a[tag] = math.log(self._init_prob.get(tag, self._infinitesimal)) + \
                math.log(self._emit_prob[tag].get(observation[0], self._infinitesimal))

        # traversal the observation
        for i in range(1, len(observation)):
            # copy the previous prob and path
            # and initialize the new prob and path
            prob_b = prob_a
            path_b = path_a

            path_a = {}
            prob_a = {}

            # get the previous max prob and corresponding tag
            for tag in self._tags.keys():
                max_prob, pre_tag = max(
                    [(pre_prob + math.log(self._transition_prob[pre_tag].get(tag, self._infinitesimal)) +
                      math.log(self._emit_prob[tag].get(observation[i], self._infinitesimal)),
                      pre_tag) for pre_tag, pre_prob in prob_b.items()])

                prob_a[tag] = max_prob
                path_a[tag] = path_b[pre_tag] + [tag]

        final_tag, final_prob = max(prob_a.items(), key=lambda a_item: a_item[1])
        return path_a[final_tag]

    def tag(self, sequence, tag_only=False):
        """
        This is viterbi algorithm
        """
        # judge whether the sequence is a list
        if not isinstance(sequence, list):
            print("Error. Not word list.")
        elif tag_only:
            return self._viterbi(sequence)
        else:
            return list(zip(sequence, self._viterbi(sequence)))


_pos = POS()
tag = _pos.tag

