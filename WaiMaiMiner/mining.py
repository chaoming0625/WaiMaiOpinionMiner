import math
import os
import re
from oujago.seg import cut

"""
The supervised fine-grain (aspect) opinion mining
"""


class OpinionMinerBasedOnHmm:
    def __init__(self):
        # the import parameters
        self._tags = {}
        self._init_prob = {}
        self._emit_prob = {}
        self._transition_prob = {}

        # the filepath parameters
        self._infinitesimal = 1e-100
        self._tag_num_filepath = root_filepath + "/f_hmm/tag_num.txt"
        self._init_filepath = root_filepath + "/f_hmm/init_prob.txt"
        self._emit_filepath = root_filepath + "/f_hmm/emit_prob.txt"
        self._transition_filepath = root_filepath + "/f_hmm/transition_prob.txt"
        self._hmm_train_corpus = root_filepath + "/f_hmm/hmm_train_corpus2.txt"
        self._hmm_add_corpus = root_filepath + "/f_hmm/hmm_add_corpus.txt"

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
            self.train()
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

    def train(self):
        # declare some variables
        tags_num = {}  # record the number of each tag
        init_num = {}  # record the number of the initial number
        transition_num = {}  # record the number of the transition numbers between tags
        emit_num = {}  # record the number of the the emit numbers between word and tag

        # the pattern for split
        pattern = re.compile("\s+")

        # open the file, read each line one by one
        for filepath in [self._hmm_train_corpus, self._hmm_add_corpus]:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    # split the line into the several splits
                    splits = pattern.split(line.strip())

                    # establish two lists to record the word and the tag
                    line_words = []
                    line_poses = []
                    line_tags = []

                    for a_split in splits:
                        # split every previous split into word and tag
                        results = a_split.split("/")
                        line_words.append(results[0])
                        line_poses.append(results[1])
                        line_tags.append(results[2])

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

    def parse(self, sentence):
        analysis = {"entity": [], "pos1": [], "neg1": [], "pos2": [], "neg2": []}

        cuts = seg.cut(sentence)
        tags = self.tag(cuts, tag_only=True)

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


_hmm = OpinionMinerBasedOnHmm()
parse = _hmm.parse
train = _hmm.train


def write_(sentence, which):
    cuts = seg.cut(sentence)
    tags = pos.tag(cuts, tag_only=True)
    output = ""

    if which == 1:
        tag = "E"
    elif which == 2:
        tag = "P1"
    elif which == 3:
        tag = "P2"
    elif which == 4:
        tag = "N1"
    elif which == 5:
        tag = "N2"
    elif which == 6:
        tag = "OT"
    else:
        return

    if which != 6:
        if len(cuts) == 1:
            output = "%s/%s/%s" % (cuts[0], tags[0], "I-" + tag)
        else:
            i = 0
            while True:
                if i == 0:
                    output = "%s/%s/%s\t" % (cuts[i], tags[i], "B-" + tag)
                elif i + 1 == len(cuts):
                    output += "%s/%s/%s\t" % (cuts[i], tags[i], "E-" + tag)
                    break
                else:
                    output += "%s/%s/%s\t" % (cuts[i], tags[i], "M-" + tag)
                i += 1
    else:
        for i in range(len(cuts)):
            output += "%s/%s/%s\t" % (cuts[i], tags[i], "OT")

    with open("f_hmm/hmm_add_corpus.txt", "a", encoding="utf-8") as f:
        f.write("%s\n" % output.strip())


def _test_bootstrapping():
    filepath = "f_hmm/hmm_train_corpus2.txt"

    # get the hmm corpus
    corpus = HMMCorpus(filepath)
    corpus.get_corpus()

    # bootstrapping
    master = BootstrappingMaster(filepath)
    master.bootstrapping()


def _test_hmm():
    # train hmm
    miner = OpinionMinerBasedOnHmm()

    # simple test
    sen = "味道不错，价格便宜，量又足"
    print(miner.parse(sen))
    cuts = seg.cut(sen)
    print(miner.tag(cuts))


if __name__ == "__main__":
    # _test_bootstrapping()
    _test_hmm()
