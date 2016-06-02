import os
from math import log

from fgom import common_lib


class OpinionMinerHMM:
    """
    The supervised fine-grain (aspect) opinion mining
    """
    def __init__(self):
        # the import parameters
        self._tags = {}
        self._init_prob = {}
        self._emit_prob = {}
        self._transition_prob = {}

        # the filepath parameters
        self._infinitesimal = 1e-20
        self._tag_num_filepath = common_lib.miner_hmm_tag_num_filepath
        self._init_filepath = common_lib.miner_hmm_init_filepath
        self._emit_filepath = common_lib.miner_hmm_emit_filepath
        self._transition_filepath = common_lib.miner_hmm_transition_filepath
        self._hmm_train_corpus = common_lib.miner_hmm_train_corpus_filepath
        self._hmm_user_add_corpus = common_lib.miner_hmm_user_add_corpus_filepath

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
            print("Need Train OpinionMinerHMM")
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

    def train(self, corpus_filename=None):
        if corpus_filename is not None:
            self._hmm_train_corpus = os.path.join(os.getcwd(), corpus_filename)

        if self._hmm_train_corpus is None:
            raise ValueError("HMM need train.")

        # declare some variables
        tags_num = {}  # record the number of each tag
        init_num = {}  # record the number of the initial number
        transition_num = {}  # record the number of the transition numbers between tags
        emit_num = {}  # record the number of the the emit numbers between word and tag

        # open the file, read each line one by one
        for filepath in [self._hmm_train_corpus, self._hmm_user_add_corpus]:
            if not os.path.exists(filepath):
                continue

            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    # split the line into the several splits
                    splits = common_lib.re_space_split.split(line.strip())

                    # establish two lists to record the word and the tag
                    line_words = []
                    line_tags = []

                    for a_split in splits:
                        # split every previous split into word and tag
                        results = a_split.split("/")
                        line_words.append(results[0])
                        line_tags.append(results[1])

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
                f.write("%s\t%.100f\t%d\n" % (tag, prob, num))

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
                    f.write("%s\t%.100f\t%s\t%d\n" % (tag1, prob, tag2, num))

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
                    f.write("%s\t%.100f\t%s\t%d\n" % (tag, prob, word, num))

    def _viterbi(self, observation):
        # record the first path and first probability
        prob_a = {}
        path_a = {}

        # initialize
        for tag in self._tags.keys():
            path_a[tag] = [tag]
            prob_a[tag] = log(self._init_prob.get(tag, self._infinitesimal)) + \
                log(self._emit_prob[tag].get(observation[0], self._infinitesimal))

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
                    [(pre_prob + log(self._transition_prob[pre_tag].get(tag, self._infinitesimal)) +
                      log(self._emit_prob[tag].get(observation[i], self._infinitesimal)),
                      pre_tag) for pre_tag, pre_prob in prob_b.items()])

                prob_a[tag] = max_prob
                path_a[tag] = path_b[pre_tag] + [tag]

        final_tag, final_prob = max(prob_a.items(), key=lambda a_item: a_item[1])
        return path_a[final_tag]

    def _tag(self, sequence, tag_only=True):
        if not isinstance(sequence, list):
            print("Error. Not word list.")
        elif tag_only:
            return self._viterbi(sequence)
        else:
            return list(zip(sequence, self._viterbi(sequence)))

    def tag(self, sentence, tag_only=True):
        """
        This is viterbi algorithm
        """
        tags = []
        sentence = sentence.strip()
        if sentence:
            clauses = common_lib.re_clause_findall.findall(sentence)
            for clause in clauses:
                cuts = common_lib.cut(clause)
                tags += self._tag(cuts, tag_only)
        return tags

    def parse(self, sentence):
        analysis = {"entity": [], "pos1": [], "neg1": [], "pos2": [], "neg2": []}

        tags = self.tag(sentence, tag_only=False)

        cuts = [a[0] for a in tags]
        tags = [a[1] for a in tags]

        word = ""
        length = len(tags)
        for i in range(length):
            if "-" in tags[i]:
                splits = tags[i].split("-")
                type_ = splits[1]
                pos_ = splits[0]
            else:
                type_ = "OT"

            if type_ == "E":
                if pos_ == "I":
                    word = cuts[i]
                    analysis["entity"].append(word)
                    word = ""
                elif pos_ == "B":
                    word = cuts[i]
                elif pos_ == "E":
                    word += cuts[i]
                    analysis["entity"].append(word)
                    word = ""
                elif pos_ == "M":
                    if i + 1 < length and "-E" in tags[i + 1]:
                        word += cuts[i]
                    else:
                        word += cuts[i]
                        analysis["entity"].append(word)
                        word = ""
            elif type_ == "P1":
                if pos_ == "I":
                    word = cuts[i]
                    analysis["pos1"].append(word)
                    word = ""
                elif pos_ == "B":
                    word = cuts[i]
                elif pos_ == "E":
                    word += cuts[i]
                    analysis["pos1"].append(word)
                    word = ""
                elif pos_ == "M":
                    if i + 1 < length and "-P1" in tags[i + 1]:
                        word += cuts[i]
                    else:
                        word += cuts[i]
                        analysis["pos1"].append(word)
                        word = ""
            elif type_ == "P2":
                if pos_ == "I":
                    word = cuts[i]
                    analysis["pos2"].append(word)
                    word = ""
                elif pos_ == "B":
                    word = cuts[i]
                elif pos_ == "E":
                    word += cuts[i]
                    analysis["pos2"].append(word)
                    word = ""
                elif pos_ == "M":
                    if i + 1 < length and "-P2" in tags[i + 1]:
                        word += cuts[i]
                    else:
                        word += cuts[i]
                        analysis["pos2"].append(word)
                        word = ""
            elif type_ == "N1":
                if pos_ == "I":
                    word = cuts[i]
                    analysis["neg1"].append(word)
                    word = ""
                elif pos_ == "B":
                    word = cuts[i]
                elif pos_ == "E":
                    word += cuts[i]
                    analysis["neg1"].append(word)
                    word = ""
                elif pos_ == "M":
                    if i + 1 < length and "-N1" in tags[i + 1]:
                        word += cuts[i]
                    else:
                        word += cuts[i]
                        analysis["neg1"].append(word)
                        word = ""
            elif type_ == "N2":
                if pos_ == "I":
                    word = cuts[i]
                    analysis["neg2"].append(word)
                    word = ""
                elif pos_ == "B":
                    word = cuts[i]
                elif pos_ == "E":
                    word += cuts[i]
                    analysis["neg2"].append(word)
                    word = ""
                elif pos_ == "M":
                    if i + 1 < length and "-N2" in tags[i + 1]:
                        word += cuts[i]
                    else:
                        word += cuts[i]
                        analysis["neg2"].append(word)
                        word = ""
        return analysis


_hmm = OpinionMinerHMM()
parse = _hmm.parse
tag = _hmm.tag
train = _hmm.train

