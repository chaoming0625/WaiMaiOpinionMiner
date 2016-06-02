# -*- coding: <encoding name> -*-

import re
import os
import math
from random import choice

from fgom import common_lib

root_path = os.getcwd()


class GetToTagCorpus:
    def __init__(self, input_filename, output_filepath, start=0, end=-1, gap=10):
        """
        :param input_filename: 需要标记的语料文件 readf
        :param output_filepath: 输出文件夹
        :param start: readf从哪一行开始输出
        :param end: readf从哪一行结束输出
        :param gap: 多少语料数写进一个文件
        """
        self._input_filename = os.path.join(root_path, input_filename)

        out_path = os.path.join(root_path, output_filepath)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self._output_filepath = os.path.join(out_path, "to_tag_corpus-%s.txt")

        self._start = start
        self._end = end
        self._gap = gap

    def run(self):
        with open(self._input_filename, encoding="utf-8") as readf:
            i = 0
            for line in readf:
                if i >= self._start:
                    # get the clauses
                    clauses = common_lib.re_clause_findall.findall(line.strip())

                    # write the clause's segments
                    with open(self._output_filepath % ((i - self._start) // self._gap),
                              "a", encoding="utf-8") as writef:
                        for clause in clauses:
                            segments = common_lib.cut(clause)
                            writef.write("%s\n" % "\t".join([segment + "/" for segment in segments]))
                        writef.write("\n" * 2)

                i += 1
                if 0 < self._end == i:
                    break


class GetTaggedCorpus:
    def __init__(self, input_filepath, output_filename, default='OT'):
        self.input_filepath = os.path.join(root_path, input_filepath)
        self.output_filename = os.path.join(root_path, output_filename)

        self.default = default

    def run(self):
        with open(self.output_filename, "w", encoding='utf-8') as writef:
            for file in os.listdir(self.input_filepath):
                # get all lines
                with open(os.path.join(self.input_filepath, file), encoding="utf-8") as readf:
                    lines = [line.strip() for line in readf.readlines()]

                # analyse each line
                line_no = 0
                runout = ""
                while line_no < len(lines):
                    if lines[line_no] == "" and runout:
                        writef.write("%s\n" % runout.strip())
                        runout = ""

                    elif lines[line_no]:
                        # get words and tags
                        splits = common_lib.re_space_split.split(lines[line_no])
                        words, tags = [], []
                        for a_split in splits:
                            if "/" in a_split:
                                try:
                                    word, tag = a_split.split("/")
                                except ValueError:
                                    raise ValueError("Tag wrong! System cannot recognize it.")
                                words.append(word)
                                tags.append(tag.upper())

                        seq_tag = False
                        final_pos = -1
                        for i in range(len(words)):
                            if not seq_tag:
                                if tags[i] == "":
                                    runout += "%s/%s\t" % (words[i], self.default)
                                else:
                                    final_pos = common_lib.final_tag_position(tags, tags[i], i)
                                    if final_pos != i:
                                        runout += "%s/%s\t" % (words[i], "B-" + tags[i])
                                        seq_tag = True
                                    else:
                                        runout += "%s/%s\t" % (words[i], "I-" + tags[i])
                            else:
                                if i == final_pos:
                                    runout += "%s/%s\t" % (words[i], "E-" + tags[i])
                                    seq_tag = False
                                else:
                                    runout += "%s/%s\t" % (words[i], "M-" + tags[i])

                        if runout and not common_lib.re_han_match.match(words[-1]):
                            writef.write("%s\n" % runout.strip())
                            runout = ""

                    line_no += 1


class BootstrappingHMM:
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
                # line_poses = []
                line_tags = []

                for a_split in splits:
                    # split every previous split into word and tag
                    results = a_split.split("/")
                    line_words.append(results[0])
                    # line_poses.append(results[1])
                    line_tags.append(results[-1])

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

    def __init__(self, bootstrapping_filename, origin_tag_filename):
        self.hmm1 = BootstrappingHMM()
        self.hmm2 = BootstrappingHMM()

        self.train_corpus_filepath = os.path.join(root_path, origin_tag_filename)
        self.bootstrapping_corpus_filepath = os.path.join(root_path, bootstrapping_filename)
        self.hmm_path = 'f_hmm/'
        self.hmm1_filepath = "f_hmm/hmm1_corpus.txt"
        self.hmm2_filepath = "f_hmm/hmm2_corpus.txt"

        if not os.path.exists(self.hmm_path):
            os.makedirs(self.hmm_path)

        self.bootstrap_contents = []
        self.added = False
        self.load_bootstrap()

    def __del__(self):
        if os.path.exists(self.hmm1_filepath):
            os.remove(self.hmm1_filepath)
        if os.path.exists(self.hmm2_filepath):
            os.remove(self.hmm2_filepath)
        if os.path.exists(self.hmm_path):
            os.rmdir(self.hmm_path)

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
                else:
                    hmm2_file.write("%s" % contents[index])
                    turn = 0

            hmm1_file.close()
            hmm2_file.close()
        else:
            print("Please check. Can not find the corpus path.")

    def run(self):
        while True:
            # randomly get the train corpus and distribute to each hmm file
            self.distribute()
            # train each hmm
            self.hmm1.train(self.hmm1_filepath)
            self.hmm2.train(self.hmm2_filepath)

            # tag each test corpus
            # if two tags are equal, then add it into corpus file, and change the state of self.added
            with open(self.train_corpus_filepath, 'a', encoding="utf-8") as train_f:
                for line in self.bootstrap_contents:
                    hmm1_tags = []
                    hmm2_tags = []
                    clauses = []
                    for clause in common_lib.re_clause_findall.findall(line.strip()):
                        segments = common_lib.cut(clause)
                        clauses.append(segments)
                        hmm1_tags.append(self.hmm1.tag(segments, tag_only=True))
                        hmm2_tags.append(self.hmm2.tag(segments, tag_only=True))

                    if hmm1_tags == hmm2_tags:
                        self.added = True
                        print("Add a new data.")
                        runout = ""
                        for i in range(len(hmm1_tags)):
                            content = ""
                            for j in range(len(hmm1_tags[i])):
                                content += "%s/%s\t" % (clauses[i][j], hmm1_tags[i][j])
                            runout += "%s\n" % content.strip()
                        train_f.write("%s\n" % runout.strip())
                        self.bootstrap_contents.remove(line)

            print("Length of remaining corpus: %d" % len(self.bootstrap_contents))
            # check whether there are new data added
            if not self.added:
                break
            else:
                # change the sate of self.added
                self.added = False



