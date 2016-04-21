import math
import os
import re
from random import choice
from WaiMaiMiner import seg, pos


"""
The supervised fine-grain (aspect) opinion mining
"""


root_filepath = os.path.dirname(os.path.realpath(__file__))


class HMMCorpus:
    def __init__(self, runout_filepath):
        self.filepath = runout_filepath

    @staticmethod
    def _find_final_position(tags, tag, start):
        while start < len(tags):
            if tags[start] != tag:
                break
            start += 1
        return start - 1

    def get_corpus(self):
        """
            Purpose:
            Get the tagged corpus.

            Results:
            Get some wrong tagging. Needed manually get rid of them.
            The following are the wrong tags:
                I-2	3
                I-EN2	3
                I-E还	1
                I-N2…	1
                I-N2点	1
                I-PP2	1
                I-V	14
                I-[2	1
                I-快	1
            So we use a regex to find the wrong tags.
                ((2 )|(EN2)|(E还)|(N2…)|(N2点)|(PP2)|(V)|(\[2)|(快))
            """
        pattern = re.compile("\s+")

        root_filepath = "f_hmm/tag/"

        # list all the tagged corpus file
        with open(self.filepath, "w", encoding="utf-8") as train_f:
            for filepath in os.listdir(root_filepath):
                sentences = []
                with open(root_filepath + filepath, encoding="utf-8") as tagged_f:
                    # firstly, get the whole sentence
                    sentence = ""
                    for line in tagged_f:
                        if line != "\n":
                            sentence += "%s\t" % line.strip()
                        else:
                            if sentence != "":
                                sentences.append(sentence)
                            sentence = ""

                # parse each sentence and get tge word and the tag

                for sentence in sentences:
                    splits = pattern.split(sentence.strip())
                    words = []
                    tags = []
                    for a_split in splits:
                        if "/" in a_split:
                            results = a_split.split("/")
                            word = results[0]
                            tag = results[1]
                            words.append(word)
                            tags.append(tag)
                    poses = pos.tag(words, tag_only=True)

                    seq_tag = False
                    final_pos = -1
                    for i in range(len(words)):
                        if not seq_tag:
                            if tags[i] == "":
                                train_f.write("%s/%s/%s\t" % (words[i], poses[i], "OT"))
                            else:
                                final_pos = self._find_final_position(tags, tags[i], i)
                                if final_pos != i:
                                    train_f.write("%s/%s/%s\t" % (words[i], poses[i], "B-" + tags[i]))
                                    seq_tag = True
                                else:
                                    train_f.write("%s/%s/%s\t" % (words[i], poses[i], "I-" + tags[i]))
                        else:
                            if i == final_pos:
                                train_f.write("%s/%s/%s\t" % (words[i], poses[i], "E-" + tags[i]))
                                seq_tag = False
                            else:
                                train_f.write("%s/%s/%s\t" % (words[i], poses[i], "M-" + tags[i]))

                    # write a nwe line
                    train_f.write("\n")


class BootstrappingHmm:
    def __init__(self):
        # the import parameters
        self.__tags = {}
        self.__init_prob = {}
        self.__emit_prob = {}
        self.__transition_prob = {}

        # the filepath parameters
        self.__infinitesimal = 1e-100

    def train(self, filepath):
        print("I'm training ...")

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
        self.__tags = tags_num

        # write the init probability
        total = sum(init_num.values())
        for tag, num in sorted(init_num.items()):
            prob = num / total
            self.__init_prob[tag] = prob

        # write the transition probability
        # get the tag and the transition tag
        for tag1 in sorted(transition_num.keys()):
            tag_dict = transition_num[tag1]
            total = sum(tag_dict.values())
            self.__transition_prob[tag1] = {}
            for tag2 in sorted(tag_dict.keys()):
                num = tag_dict[tag2]
                prob = num / total
                self.__transition_prob[tag1][tag2] = prob
        for key in self.__tags:
            if key not in self.__transition_prob:
                self.__transition_prob[key] = {}

        # write the emit probability
        for tag in sorted(emit_num.keys()):
            tag_dict = emit_num[tag]
            total = sum(tag_dict.values())
            self.__emit_prob[tag] = {}
            for word in sorted(tag_dict.keys()):
                num = tag_dict[word]
                prob = num / total
                self.__emit_prob[tag][word] = prob
        for key in self.__tags:
            if key not in self.__emit_prob:
                self.__emit_prob[key] = {}

        print("Trains over ...")

    def __viterbi(self, observation):
        # record the first path and first probability
        prob_a = {}
        path_a = {}

        # initialize
        for tag in self.__tags.keys():
            path_a[tag] = [tag]
            prob_a[tag] = math.log(self.__init_prob.get(tag, self.__infinitesimal)) + \
                math.log(self.__emit_prob[tag].get(observation[0], self.__infinitesimal))

        # traversal the observation
        for i in range(1, len(observation)):
            # copy the previous prob and path
            # and initialize the new prob and path
            prob_b = prob_a
            path_b = path_a

            path_a = {}
            prob_a = {}

            # get the previous max prob and corresponding tag
            for tag in self.__tags.keys():
                max_prob, pre_tag = max(
                    [(pre_prob + math.log(self.__transition_prob[pre_tag].get(tag, self.__infinitesimal)) +
                      math.log(self.__emit_prob[tag].get(observation[i], self.__infinitesimal)), pre_tag)
                     for pre_tag, pre_prob in prob_b.items()])

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
            raise ValueError("Error. Not word list.")
        elif tag_only:
            return self.__viterbi(sequence)
        else:
            return list(zip(sequence, self.__viterbi(sequence)))


class BootstrappingMaster:
    """BootstrappingMaster"""
    def __init__(self, train_corpus_filepath):
        self.hmm1 = BootstrappingHmm()
        self.hmm2 = BootstrappingHmm()
        self.hmm3 = BootstrappingHmm()

        self.train_corpus_filepath = train_corpus_filepath
        self.bootstrapping_corpus_filepath = "f_hmm/hmm_bootstrapping_corpus.txt"
        self.hmm1_filepath = "f_hmm/hmm1_corpus.txt"
        self.hmm2_filepath = "f_hmm/hmm2_corpus.txt"
        self.hmm3_filepath = "f_hmm/hmm3_corpus.txt"

        self.bootstrap_contents = []
        self.added = False
        self.split_pattern = re.compile("\s+")
        self.load_bootstrap()

    def load_bootstrap(self):
        with open(self.bootstrapping_corpus_filepath, encoding="utf-8") as bootstrap_f:
            for line in bootstrap_f:
                self.bootstrap_contents.append(line.strip())

    @staticmethod
    def check_filepath(filepath):
        if os.path.exists(filepath):
            return True
        else:
            return False

    def distribute(self):
        """Distribute the tagged corpus."""
        if self.check_filepath(self.train_corpus_filepath):
            print("Distributing.")

            hmm1_file = open(self.hmm1_filepath, "w", encoding="utf-8")
            hmm2_file = open(self.hmm2_filepath, "w", encoding="utf-8")
            hmm3_file = open(self.hmm3_filepath, "w", encoding="utf-8")

            with open(self.train_corpus_filepath, encoding="utf-8") as f:
                contents = f.readlines()
            index_list = list(range(len(contents)))

            turn = 0

            while len(index_list) > 0:
                index = choice(index_list)
                index_list.remove(index)
                if turn == 0:
                    hmm1_file.write("%s" % contents[index])
                    turn += 1
                elif turn == 1:
                    hmm2_file.write("%s" % contents[index])
                    turn += 1
                else:
                    hmm3_file.write("%s" % contents[index])
                    turn = 0

            hmm1_file.close()
            hmm2_file.close()
            hmm3_file.close()
        else:
            print("Please check. Can not find the corpus path.")

    def bootstrapping(self):
        while True:
            # randomly get the train corpus and distribute to each hmm file
            self.distribute()
            # train each hmm
            self.hmm1.train(self.hmm1_filepath)
            self.hmm2.train(self.hmm2_filepath)
            self.hmm3.train(self.hmm3_filepath)

            # tag each test corpus
            # if two tags are equal, then add it into corpus file, and change the state of self.added
            with open(self.train_corpus_filepath, 'a', encoding="utf-8") as train_f:
                for line in self.bootstrap_contents:
                    segments = seg.cut(line)

                    hmm1_tags = self.hmm1.tag(segments, tag_only=True)
                    hmm2_tags = self.hmm2.tag(segments, tag_only=True)
                    hmm3_tags = self.hmm3.tag(segments, tag_only=True)

                    if hmm1_tags == hmm2_tags == hmm3_tags:
                        self.added = True
                        print("Add a new data.")
                        poses = pos.tag(segments, tag_only=True)
                        runout = ""
                        for i in range(len(poses)):
                            runout += "%s/%s/%s\t" % (segments[i], poses[i], hmm1_tags[i])
                        train_f.write("%s\n" % runout.strip())
                        self.bootstrap_contents.remove(line)

            print("Length of remaining corpus: %d" % len(self.bootstrap_contents))
            # check whether there are new data added
            if not self.added:
                break
            else:
                # change the sate of self.added
                self.added = False


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
        self._hmm_train_corpus = root_filepath + "/f_hmm/hmm_train_corpus.txt"
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


def write(sentence, which):
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
    filepath = "f_hmm/hmm_train_corpus.txt"

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
