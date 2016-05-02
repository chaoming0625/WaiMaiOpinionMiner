import os
import re
from oujago import seg


init_seg = False


def init():
    seg.load_userdict(os.path.join(os.path.dirname(os.path.realpath(__file__)), "f_seg/user_dict.txt"))
    seg.add_word("快递", 10000)
    seg.suggest_freq(("面", "太厚"))
    seg.suggest_freq(("价格", "便宜"))
    seg.suggest_freq(("服务", "周到"))
    seg.suggest_freq(("速度", "快"))


def cut(sentence):
    if not init_seg:
        init()
    return list(seg.cut(sentence))

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

