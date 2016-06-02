import os
import re
import jieba

__init_seg = False


def __init():
    user_dict_path = os.path.join(root_filepath, "f_seg/user_dict.txt")
    jieba.load_userdict(user_dict_path)
    jieba.add_word("快递", 10000)
    jieba.suggest_freq(("面", "太厚"))
    jieba.suggest_freq(("价格", "便宜"))
    jieba.suggest_freq(("服务", "周到"))
    jieba.suggest_freq(("速度", "快"))


def cut(sentence):
    if not __init_seg:
        __init()
    return jieba.lcut(sentence)

root_filepath = os.path.dirname(os.path.realpath(__file__))
miner_hmm_tag_num_filepath = os.path.normpath(os.path.join(
    root_filepath, "f_hmm/tag_num.txt"))
miner_hmm_init_filepath = os.path.normpath(os.path.join(
    root_filepath, "f_hmm/init_prob.txt"))
miner_hmm_emit_filepath = os.path.normpath(os.path.join(
    root_filepath, "f_hmm/emit_prob.txt"))
miner_hmm_transition_filepath = os.path.normpath(os.path.join(
    root_filepath, "f_hmm/transition_prob.txt"))
miner_hmm_train_corpus_filepath = os.path.normpath(os.path.join(
    root_filepath, "f_hmm/hmm_train_corpus.txt"))
miner_hmm_user_add_corpus_filepath = os.path.normpath(os.path.join(
    root_filepath, "f_hmm/hmm_user_add_corpus.txt"))

re_clause_findall = re.compile(r"([a-zA-Z0-9:\u4e00-\u9fa5]+[，。%、！!？?,；～~.… ]*)")
re_space_split = re.compile(r"\s+")
re_han_match = re.compile('[\s\u4e00-\u9fa5]+')


def final_tag_position(tags, tag, start):
    while start < len(tags):
        if tags[start] != tag:
            break
        start += 1
    return start - 1


def write_(sentence, which):
    cuts = cut(sentence)
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


