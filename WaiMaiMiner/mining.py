import os
import re
import math
import requests
import json
from WaiMaiMiner import common_lib


class Attr:
    def __init__(self, content=None, start=-1, end=-1):
        self.content = content
        self.start = start
        self.end = end

    def __str__(self):
        return self.content

    def __repr__(self):
        return "%s(%s, %s, %s)" % (self.__class__, self.content, self.start, self.end)


class Op:
    def __init__(self, content=None, start=-1, end=-1, orient=2):
        self.content = content
        self.start = start
        self.end = end
        self.orient = orient

    def __str__(self):
        return self.content

    def __repr__(self):
        return "%s(%s, %s, %s, %s)" % (
            self.__class__, self.content, self.start, self.end, self.orient)


class Pair:
    def __init__(self, attribution, opinion):
        self.attribution = attribution
        self.opinion = opinion

    def __str__(self):
        return "%s -> %s" % (self.attribution, self.opinion)

    def __repr__(self):
        return "%s(%s, %s)" % (
            self.__class__, self.attribution, self.opinion)


class Clause:
    def __init__(self, clause, attribution, opinion):
        self.clause = clause
        self.attribution = attribution
        self.opinion = opinion

    def __str__(self):
        return "%s (%s -> %s)" % (self.clause, self.attribution, self.opinion)

    def __repr__(self):
        return "%s(%s, %s, %s)" % (
            self.__class__, self.clause, self.attribution, self.opinion)


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
        for filepath in [self._hmm_train_corpus, self._hmm_user_add_corpus]:
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

    def tag(self, sequence, tag_only=True):
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

        cuts = common_lib.cut(sentence)
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


class Analysis:
    def __init__(self, url):
        self.base_url = "http://waimai.baidu.com/waimai/comment/getshop?display=" \
                        "json&shop_id=%s&page=%s&count=60"

        # get the shop id
        self._shop_id = ''
        self._get_shop_id(url)

        '''
        init some variables
        '''
        self.page_num = 1

        # contain all comments
        self.comments = []
        # contain all positive comment number
        self.positive = []
        # contain all negative comment number
        self.negative = []
        # contain all neutral comment number
        self.neutral = []

        # contain all service comments
        self.service = []
        # contain all delivery comments
        self.delivery = []
        # contain all taste comments
        self.taste = []
        # contain other comments
        self.others = []

        self.all = []

        self.average_score = {}
        self.score_detail = {}
        self.weeks_score = {}
        self.recommend_dishes = None
        self.comment_num = 0

        self.cost_time = []
        self.service_score = []
        self.dish_score = []
        self.sfrom = []
        self.score = []
        self.create_time = []
        self.arrive_time = []
        self.useful_comment_id = []
        self.rubbish_comment_id = []

        # start crawl the comments
        self.crawl()

    def _get_shop_id(self, url):
        if url is not None:
            shop_id = re.search("\d+", url)
            if shop_id is None:
                raise ValueError("Not found shop id")

            self.shop_id = shop_id.group()
        else:
            raise ValueError("Not found shop id")

    def crawl(self):
        # get the first page comment
        i = 0
        while i < self.page_num:
            self._get_json_request(self.base_url % (self.shop_id, i + 1))
            i += 1

        self._filter()

    def _get_json_request(self, url):
        try:
            result = requests.get(url)
        except requests.ConnectionError:
            raise ValueError("requests.ConnectionError")

        content = json.loads(result.text)
        result = content["result"]
        if self.page_num == 1:
            self._get_initial_info(result)

        content = result["content"]

        for a_json in content:
            self._get_a_json_info(a_json)

    def _get_initial_info(self, result):
        # get the page number
        self.page_num = result["comment_num"] // 60 + 1

        # get the average score information
        self.average_score["average_dish_score"] = float(result["average_dish_score"])
        self.average_score["average_service_score"] = float(result["average_service_score"])
        self.average_score["average_score"] = float(result["average_score"])

        # get the score detail
        self.score_detail = result["score_detail"]

        # get the weeks score
        for key, value in result["weeks_score"].items():
            self.weeks_score[key] = float(value)

        # get the recommend dished
        self.recommend_dishes = result["recommend_dishes"]

        # get the comment num
        self.comment_num = result['comment_num']

    def _get_a_json_info(self, a_json):
        self.comments.append(a_json["content"])
        self.cost_time.append(a_json["cost_time"])
        self.service_score.append(int(a_json["service_score"]))
        self.dish_score.append(int(a_json["dish_score"]))
        self.score.append(int(a_json["score"]))
        self.sfrom.append(a_json["sfrom"][3:] if "na-" in a_json["sfrom"] else a_json["sfrom"])
        self.create_time.append(a_json["create_time"])
        self.arrive_time.append(a_json["arrive_time"])

    def _filter(self):
        """
        get the useful comment, get rid of rubbish comment.

        We use the maximum entropy to classify the polarity of the comment.
        maxEntClassifier's parameters are wrote into the file
        """
        for i, sentence in enumerate(self.comments):
            rubbish_comment = False

            if self._is_english(sentence):
                rubbish_comment = True
            elif self._is_numeric(sentence):
                rubbish_comment = True
            elif self._is_too_short(sentence):
                rubbish_comment = True
            elif self._is_word_repeat(sentence):
                rubbish_comment = True

            if rubbish_comment:
                self.rubbish_comment_id.append(i)

            else:
                self.useful_comment_id.append(i)

                if self.score[i] >= 4:
                    self.positive.append(i)
                elif self.score[i] >= 2:
                    self.neutral.append(i)
                else:
                    self.negative.append(i)

    @staticmethod
    def _is_too_short(sentence):
        """judge if the f_waimai is too short or the chinese characters are few
            if True, write to the abandoned file
            :param sentence: a f_waimai
            :return: True or False
            """
        if len(sentence) < 5:
            return True
        if len(re.findall(r'[\u4e00-\u9fa5]', sentence)) <= len(sentence) * 0.4:
            return True
        return False

    @staticmethod
    def _is_numeric(sentence):
        """judge if the f_waimai's characters are all or almost numbers
            if True, write to the abandoned file
            :param sentence: a f_waimai
            :return: True or False
            """
        match = re.findall("\d+", sentence)
        if match is not None and sum([len(m) for m in match]) >= len(sentence) * 0.75:
            return True
        return False

    @staticmethod
    def _is_english(sentence):
        """judge if the f_waimai's characters are all English
            if True, write to the abandoned file
            :param sentence: a f_waimai
            :return: True or False
            """
        match = re.findall("[a-zA-Z]+", sentence)
        if match is not None and sum([len(m) for m in match]) >= len(sentence) * 0.75:
            return True
        return False

    @staticmethod
    def _is_word_repeat(sentence):
        """check if the f_waimai is always the repeat word
            :param sentence: a f_waimai
            :return: True or False
            """
        repeat_words, length = [], 0
        for word in sentence:
            times = sentence.count(word)
            if times >= 4 and word not in repeat_words:
                repeat_words.append(word)
                length += times
        if length > len(sentence) / 2:
            return True
        return False

    @staticmethod
    def find_pos(clause, word, start=0):
        try:
            index = clause.index(word, start)
        except ValueError:
            return -1, -1
        return index, index + len(word)

    # def get_pair(self, attributions, opinions):
    #     pass

    def match(self, clause, attributions, opinions):
        if len(attributions) == 1 and len(opinions) == 1:
            self.all.append(Clause(clause, attributions[0], opinions[0]))
        elif len(attributions) == 0 and len(opinions) == 1:
            self.all.append(Clause(clause, None, opinions[0]))

    def extract_from_sentence(self, sentence):
        clauses = common_lib.re_clause_findall.findall(sentence)

        start_position = 0
        for clause in clauses:
            position = 0
            attributions = []
            opinions = []

            cuts = common_lib.cut(clause)
            tags = hmm_tag(cuts)

            word = ""
            length = len(cuts)
            for i in range(length):
                if "-" in tags[i]:
                    prefix, type_ = tags[i].split("-")

                    if type_ == "E":
                        if prefix == "I":
                            word = cuts[i]
                            start, end = self.find_pos(clause, word, position)
                            # attributions.append(Attr(word, start + start_position, end + start_position))
                            attributions.append(Attr(word, start, end))
                            position = start
                            word = ""
                        elif prefix == "B":
                            word = cuts[i]
                        elif prefix == "E":
                            word += cuts[i]
                            start, end = self.find_pos(clause, word, position)
                            # attributions.append(Attr(word, start + start_position, end + start_position))
                            attributions.append(Attr(word, start, end))
                            position = start
                            word = ""
                        elif prefix == "M":
                            if i + 1 < length and "-E" in tags[i + 1]:
                                word += cuts[i]
                            else:
                                word += cuts[i]
                                start, end = self.find_pos(clause, word, position)
                                # attributions.append(Attr(word, start + start_position, end + start_position))
                                attributions.append(Attr(word, start, end))
                                position = start
                                word = ""
                    elif type_ == "P1":
                        if prefix == "I":
                            word = cuts[i]
                            start, end = self.find_pos(clause, word, position)
                            # opinions.append(Op(word, start + start_position, end + start_position, 1))
                            opinions.append(Op(word, start, end, 1))
                            position = start
                            word = ""
                        elif prefix == "B":
                            word = cuts[i]
                        elif prefix == "E":
                            word += cuts[i]
                            start, end = self.find_pos(clause, word, position)
                            # opinions.append(Op(word, start + start_position, end + start_position, 1))
                            opinions.append(Op(word, start, end, 1))
                            position = start
                            word = ""
                        elif prefix == "M":
                            if i + 1 < length and "-P1" in tags[i + 1]:
                                word += cuts[i]
                            else:
                                word += cuts[i]
                                start, end = self.find_pos(clause, word, position)
                                # opinions.append(Op(word, start + start_position, end + start_position, 1))
                                opinions.append(Op(word, start, end, 1))
                                position = start
                                word = ""
                    elif type_ == "N1":
                        if prefix == "I":
                            word = cuts[i]
                            start, end = self.find_pos(clause, word, position)
                            # opinions.append(Op(word, start + start_position, end + start_position, 0))
                            opinions.append(Op(word, start, end, 0))
                            position = start
                            word = ""
                        elif prefix == "B":
                            word = cuts[i]
                        elif prefix == "E":
                            word += cuts[i]
                            start, end = self.find_pos(clause, word, position)
                            # opinions.append(Op(word, start + start_position, end + start_position, 0))
                            opinions.append(Op(word, start, end, 0))
                            position = start
                            word = ""
                        elif prefix == "M":
                            if i + 1 < length and "-N1" in tags[i + 1]:
                                word += cuts[i]
                            else:
                                word += cuts[i]
                                start, end = self.find_pos(clause, word, position)
                                # opinions.append(Op(word, start + start_position, end + start_position, 0))
                                opinions.append(Op(word, start, end, 0))
                                position = start
                                word = ""

            start_position += len(clause)
            self.match(clause, attributions, opinions)

    def extract_all(self):
        for i in self.useful_comment_id:
            self.extract_from_sentence(self.comments[i])


_hmm = OpinionMinerHMM()
parse = _hmm.parse
hmm_tag = _hmm.tag
train = _hmm.train


def write_(sentence, which):
    cuts = common_lib.cut(sentence)
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
            output = "%s/%s" % (cuts[0], "I-" + tag)
        else:
            i = 0
            while True:
                if i == 0:
                    output = "%s/%s\t" % (cuts[i], "B-" + tag)
                elif i + 1 == len(cuts):
                    output += "%s/%s\t" % (cuts[i], "E-" + tag)
                    break
                else:
                    output += "%s/%s\t" % (cuts[i], "M-" + tag)
                i += 1
    else:
        for i in range(len(cuts)):
            output += "%s/%s\t" % (cuts[i], "OT")

    with open("f_hmm/hmm_user_add_corpus.txt", "a", encoding="utf-8") as f:
        f.write("%s\n" % output.strip())


def find_pos(clause, word, start=0):
    try:
        index = clause.index(word, start)
    except ValueError:
        return -1, -1
    return index, index + len(word)


def match(attributions, opinions):
    if len(attributions) == 1 and len(opinions) == 1:
        return Pair(attributions[0], opinions[0])
    else:
        return None


def analyse(sentence, f):
    results = []
    clauses = common_lib.re_clause_findall.findall(sentence)

    start_position = 0
    for clause in clauses:
        position = 0
        attributions = []
        opinions = []

        cuts = common_lib.cut(clause)
        tags = hmm_tag(cuts)

        f.write("%s\n" % cuts)
        f.write("%s\n" % tags)

        word = ""
        length = len(cuts)
        for i in range(length):
            if "-" in tags[i]:
                prefix, type_ = tags[i].split("-")

                if type_ == "E":
                    if prefix == "I":
                        word = cuts[i]
                        start, end = find_pos(clause, word, position)
                        attributions.append(Attr(word, start + start_position, end + start_position))
                        position = start
                        word = ""
                    elif prefix == "B":
                        word = cuts[i]
                    elif prefix == "E":
                        word += cuts[i]
                        start, end = find_pos(clause, word, position)
                        attributions.append(Attr(word, start + start_position, end + start_position))
                        position = start
                        word = ""
                    elif prefix == "M":
                        if i + 1 < length and "-E" in tags[i + 1]:
                            word += cuts[i]
                        else:
                            word += cuts[i]
                            start, end = find_pos(clause, word, position)
                            attributions.append(Attr(word, start + start_position, end + start_position))
                            position = start
                            word = ""
                elif type_ == "P1":
                    if prefix == "I":
                        word = cuts[i]
                        start, end = find_pos(clause, word, position)
                        opinions.append(Op(word, start + start_position, end + start_position, 1))
                        position = start
                        word = ""
                    elif prefix == "B":
                        word = cuts[i]
                    elif prefix == "E":
                        word += cuts[i]
                        start, end = find_pos(clause, word, position)
                        opinions.append(Op(word, start + start_position, end + start_position, 1))
                        position = start
                        word = ""
                    elif prefix == "M":
                        if i + 1 < length and "-P1" in tags[i + 1]:
                            word += cuts[i]
                        else:
                            word += cuts[i]
                            start, end = find_pos(clause, word, position)
                            opinions.append(Op(word, start + start_position, end + start_position, 1))
                            position = start
                            word = ""
                elif type_ == "N1":
                    if prefix == "I":
                        word = cuts[i]
                        start, end = find_pos(clause, word, position)
                        opinions.append(Op(word, start + start_position, end + start_position, 0))
                        position = start
                        word = ""
                    elif prefix == "B":
                        word = cuts[i]
                    elif prefix == "E":
                        word += cuts[i]
                        start, end = find_pos(clause, word, position)
                        opinions.append(Op(word, start + start_position, end + start_position, 0))
                        position = start
                        word = ""
                    elif prefix == "M":
                        if i + 1 < length and "-N1" in tags[i + 1]:
                            word += cuts[i]
                        else:
                            word += cuts[i]
                            start, end = find_pos(clause, word, position)
                            opinions.append(Op(word, start + start_position, end + start_position, 0))
                            position = start
                            word = ""

        start_position += len(clause)

        results.append(Clause(clause, attributions, opinions))

    return results


def _test_hmm():
    # simple test
    sen = "味道不错，价格便宜，量又足"
    print(parse(sen))
    cuts = common_lib.cut(sen)
    print(_hmm.tag(cuts))


def _test_analyse():

    origin_files = ["D:\\My Data\\NLP\\SA\\waimai\\positive_corpus_v1.txt",
                    "D:\\My Data\\NLP\\SA\\waimai\\negative_corpus_v1.txt"]

    runout_file = ['f_runout/test_analyse_positive.txt',
                   'f_runout/test_analyse_negative.txt']

    for k in range(2):
        j = 0
        with open(origin_files[k], encoding="utf-8") as readf:
            with open(runout_file[k], "w", encoding="utf-8") as writef:
                for line in readf:
                    results = analyse(line.strip(), writef)

                    writef.write("Origin :\t%s\n" % line)

                    for i, pair in enumerate(results):
                        writef.write("Clause %d: %s\n" % (i, pair.sentence))
                        writef.write("Attributions:\n")
                        for attr in pair.attributions:
                            writef.write("\t%s\t%d\t%d\n" % (attr.content, attr.start, attr.end))
                        writef.write("Opinions:\n")
                        for op in pair.opinions:
                            writef.write("\t%s\t%d\t%d\t%s\n" % (
                                op.content, op.start, op.end, "positive" if op.orient else 'negative'))
                        writef.write("\n")
                    writef.write("\n\n\n")

                    j += 1
                    if j == 100:
                        break

if __name__ == "__main__":
    pass
    # _test_hmm()

    # analyse("还可以~~而且有优惠很好哈哈~")
    _test_analyse()
